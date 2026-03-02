"""Pydantic request/response models."""
from __future__ import annotations

from pydantic import BaseModel
from typing import Dict, List, Optional


# ── Subject stages (ordered) ──────────────────────────────────────────────
STAGES = [
    "created",
    "videos_linked",
    "prelabeled",
    "labeling",
    "labeled",
    "committed",
    "training",
    "trained",
    "analyzed",
    "refined",
    "corrected",
]

STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}


# ── Subjects ──────────────────────────────────────────────────────────────
class SubjectCreate(BaseModel):
    name: str
    video_pattern: Optional[str] = None


class SubjectUpdate(BaseModel):
    camera_name: Optional[str] = None
    no_face_videos: Optional[List[str]] = None


class SubjectResponse(BaseModel):
    id: int
    name: str
    stage: str
    stage_idx: int
    iteration: int
    camera_name: Optional[str]
    dlc_dir: Optional[str]
    video_count: int = 0
    has_snapshots: bool = False
    has_labels: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SubjectDetail(SubjectResponse):
    videos: List[str] = []
    trials: List[str] = []
    no_face_videos: List[str] = []
    jobs: List[dict] = []
    label_sessions: List[dict] = []


# ── Jobs ──────────────────────────────────────────────────────────────────
class RunStepRequest(BaseModel):
    step: str


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
    session_type: str = "initial"


class SessionResponse(BaseModel):
    id: int
    subject_id: int
    iteration: int
    session_type: str
    status: str
    label_count: int = 0
    created_at: Optional[str] = None
