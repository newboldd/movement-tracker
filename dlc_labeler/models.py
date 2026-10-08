"""Pydantic request/response models."""
from __future__ import annotations

from pydantic import BaseModel
from typing import Dict, List, Optional


# ── Subject stages (ordered) ──────────────────────────────────────────────
# The DeepLabCut loop, in order: create the project, label frames, commit
# them, train, analyze, correct the predictions, refine with the
# corrections.  A subject's stage is inferred from what is on disk
# (services/discovery.py), so it stays right even if the database is lost.
STAGES = [
    "created",
    "labeling",
    "committed",
    "training",
    "trained",
    "analyzed",
    "corrected",
    "refined",
]

STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}


# ── Subjects ──────────────────────────────────────────────────────────────
class SubjectCreate(BaseModel):
    name: str
    video_pattern: Optional[str] = None
    camera_mode: Optional[str] = None


class SubjectUpdate(BaseModel):
    name: Optional[str] = None
    camera_mode: Optional[str] = None
    camera_name: Optional[str] = None
    no_face_videos: Optional[List[str]] = None
    notes: Optional[str] = None


class SubjectResponse(BaseModel):
    id: int
    name: str
    stage: str
    stage_idx: int
    iteration: int
    camera_mode: str = "stereo"
    camera_name: Optional[str] = None
    dlc_dir: Optional[str] = None
    notes: Optional[str] = None
    video_count: int = 0
    has_snapshots: bool = False
    has_labels: bool = False
    labeled_frame_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SubjectDetail(SubjectResponse):
    videos: List[str] = []
    trials: List[str] = []
    no_face_videos: List[str] = []
    jobs: List[dict] = []
    label_sessions: List[dict] = []


# ── Jobs ──────────────────────────────────────────────────────────────────
class JobLaunch(BaseModel):
    job_type: str
    subjects: List[str] = []
    gpu_index: int = 0
    # MediaPipe pass options
    reverse: bool = False
    static_image_mode: bool = False
    use_bbox: bool = True
    trial_idx: Optional[int] = None


class JobResponse(BaseModel):
    id: int
    subject_id: int
    job_type: str
    status: str
    progress_pct: float
    error_msg: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


# ── Labeling ──────────────────────────────────────────────────────────────
class LabelData(BaseModel):
    frame_num: int
    trial_idx: int = 0
    side: str = "OS"
    keypoints: Dict[str, List[Optional[float]]] = {}


class LabelBatchSave(BaseModel):
    labels: List[LabelData]


class SessionCreate(BaseModel):
    # "initial"     — place labels on raw video to build the first training set
    # "corrections" — edit model predictions; saved to corrections/ CSVs
    # "refine"      — promote chosen corrections into a new training round
    session_type: str = "initial"


class SessionResponse(BaseModel):
    id: int
    subject_id: int
    iteration: int
    session_type: str
    status: str
    label_count: int = 0
    created_at: Optional[str] = None


class TrainFrame(BaseModel):
    frame_num: int
    side: str


class CommitRequest(BaseModel):
    # Refine mode: the corrected frames to add to the DLC training set.
    # Chosen on the page from the corrections-vs-predictions diff, so a
    # frame the model already gets right doesn't bloat the training set.
    train_frames: List[TrainFrame] = []
