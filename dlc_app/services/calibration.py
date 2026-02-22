"""Stereo calibration loading for the DLC app.

Loads camera_assignments.yaml to map subjects to camera calibration files,
then loads OpenCV YAML calibration files. Adapted from preprocess/calibration.py.
"""

import logging
import os
import re

import cv2
import numpy as np
import yaml

from ..config import get_settings, PROJECT_DIR

logger = logging.getLogger(__name__)

# Cache for camera assignments YAML and loaded calibration dicts
_camera_assignments = None
_calib_cache = {}


def load_calibration(yaml_path: str) -> dict:
    """Load stereo calibration from OpenCV YAML file.

    Returns dict with K1, K2, dist1, dist2, R, T, P1, P2.
    """
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    K1 = fs.getNode('K1').mat()
    dist1 = fs.getNode('dist1').mat()
    K2 = fs.getNode('K2').mat()
    dist2 = fs.getNode('dist2').mat()
    R = fs.getNode('R').mat()
    T = fs.getNode('T').mat()
    fs.release()

    P1 = np.hstack((K1, np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T.reshape(3, 1)))

    return {
        'K1': K1, 'K2': K2,
        'dist1': dist1, 'dist2': dist2,
        'R': R, 'T': T,
        'P1': P1, 'P2': P2,
    }


def _load_camera_assignments() -> dict:
    """Load calibration/camera_assignments.yaml (cached)."""
    global _camera_assignments
    if _camera_assignments is not None:
        return _camera_assignments

    settings = get_settings()
    calib_dir = settings.calibration_dir
    if not calib_dir:
        calib_dir = str(PROJECT_DIR / "calibration")

    yaml_path = os.path.join(calib_dir, 'camera_assignments.yaml')
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"camera_assignments.yaml not found at {yaml_path}. "
            "Configure calibration_dir in Settings."
        )
    with open(yaml_path) as f:
        _camera_assignments = yaml.safe_load(f)
    return _camera_assignments


def get_calibration_for_subject(subject_name: str) -> dict | None:
    """Load the correct stereo calibration for a subject.

    Looks up calibration/camera_assignments.yaml:
      1. Check subject_overrides (e.g. MSA10 -> dysaut_cam2)
      2. Fall back to prefix_defaults (e.g. PD -> camera1)

    Returns calibration dict or None if not available.
    """
    try:
        assignments = _load_camera_assignments()
    except FileNotFoundError:
        logger.warning("No camera_assignments.yaml found")
        return None

    settings = get_settings()
    calib_dir = settings.calibration_dir
    if not calib_dir:
        calib_dir = str(PROJECT_DIR / "calibration")

    # Extract subject prefix + number (e.g. "MSA10" from "MSA10")
    match = re.match(r'([A-Za-z]+\d*)', subject_name)
    if not match:
        logger.warning(f"Cannot parse subject name: {subject_name}")
        return None
    subject = match.group(1)

    # Check subject overrides first
    camera_name = assignments.get('subject_overrides', {}).get(subject)

    # Fall back to prefix defaults
    if camera_name is None:
        prefix_match = re.match(r'([A-Za-z]+)', subject)
        if prefix_match:
            prefix = prefix_match.group(1)
            camera_name = assignments.get('prefix_defaults', {}).get(prefix)

    if camera_name is None:
        logger.warning(f"No camera assignment for subject '{subject_name}'")
        return None

    # Load calibration (cached by camera name)
    if camera_name not in _calib_cache:
        cameras = assignments.get('cameras', {})
        if camera_name not in cameras:
            logger.warning(f"Camera '{camera_name}' not in cameras section")
            return None
        calib_rel_path = cameras[camera_name]['calibration']
        # Resolve relative to project dir (calibration paths are relative to project root)
        calib_path = os.path.join(str(PROJECT_DIR), calib_rel_path)
        if not os.path.exists(calib_path):
            logger.warning(f"Calibration file not found: {calib_path}")
            return None
        _calib_cache[camera_name] = load_calibration(calib_path)
        logger.info(f'Loaded calibration for {camera_name}: {calib_path}')

    return _calib_cache[camera_name]


def triangulate_points(pts_L: np.ndarray, pts_R: np.ndarray,
                       calib: dict) -> np.ndarray:
    """Triangulate 2D point pairs from stereo cameras to 3D.

    Args:
        pts_L: (N, 2) left camera points in pixels
        pts_R: (N, 2) right camera points in pixels
        calib: calibration dict with K1, K2, dist1, dist2, P1, P2

    Returns:
        (N, 3) 3D points. NaN for invalid pairs.
    """
    N = len(pts_L)
    pts_3d = np.full((N, 3), np.nan)

    # Undistort points
    valid = ~np.isnan(pts_L[:, 0]) & ~np.isnan(pts_R[:, 0])
    if not np.any(valid):
        return pts_3d

    valid_idx = np.where(valid)[0]
    pL = pts_L[valid_idx].reshape(-1, 1, 2).astype(np.float64)
    pR = pts_R[valid_idx].reshape(-1, 1, 2).astype(np.float64)

    pL_undist = cv2.undistortPoints(pL, calib['K1'], calib['dist1'], P=calib['P1'])
    pR_undist = cv2.undistortPoints(pR, calib['K2'], calib['dist2'], P=calib['P2'])

    # Triangulate each point
    for i, idx in enumerate(valid_idx):
        pt4d = cv2.triangulatePoints(
            calib['P1'], calib['P2'],
            pL_undist[i].T, pR_undist[i].T,
        )
        pt3d = (pt4d[:3] / pt4d[3]).flatten()
        pts_3d[idx] = pt3d

    return pts_3d


def compute_3d_distance(thumb_L, thumb_R, index_L, index_R, calib):
    """Compute 3D Euclidean distance between thumb and index tips.

    Args:
        thumb_L, thumb_R: (2,) pixel coords for thumb in left/right camera
        index_L, index_R: (2,) pixel coords for index in left/right camera
        calib: calibration dict

    Returns:
        float distance in mm, or NaN if any point is missing
    """
    pts_L = np.array([thumb_L, index_L])
    pts_R = np.array([thumb_R, index_R])

    if np.any(np.isnan(pts_L)) or np.any(np.isnan(pts_R)):
        return np.nan

    pts_3d = triangulate_points(pts_L, pts_R, calib)

    if np.any(np.isnan(pts_3d)):
        return np.nan

    return float(np.linalg.norm(pts_3d[0] - pts_3d[1]))
