"""Pydantic request/response models."""

from __future__ import annotations

from pydantic import BaseModel
from typing import Optional


# ── Subject stages (ordered) ──────────────────────────────────────────────
STAGES = [
    "created",
    "videos_linked",
    "labeling",
    "labeled",
    "committed",
    "training_dataset_created",
    "training",
    "trained",
    "cropping",
    "cropped",
    "analyzing",
    "analyzed",
    "triangulating",
    "triangulated",
    "error_detection",
    "errors_detected",
    "error_labeling",
    "error_labeled",
    "retraining",
    "retrained",
    "complete",
]

STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}


# ── Subjects ──────────────────────────────────────────────────────────────
class SubjectCreate(BaseModel):
    name: str
    video_pattern: Optional[str] = None


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
    videos: list[str] = []
    trials: list[str] = []
    jobs: list[dict] = []
    label_sessions: list[dict] = []


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
    keypoints: dict[str, list[Optional[float]]] = {}


class LabelBatchSave(BaseModel):
    labels: list[LabelData]


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
