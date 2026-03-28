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
    "events_partial",
    "events_complete",
]

STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}


# ── Subjects ──────────────────────────────────────────────────────────────
class SubjectCreate(BaseModel):
    name: str
    video_pattern: Optional[str] = None
    diagnosis: str = "Control"


class SubjectUpdate(BaseModel):
    name: Optional[str] = None
    camera_mode: Optional[str] = None
    camera_name: Optional[str] = None
    no_face_videos: Optional[List[str]] = None
    diagnosis: Optional[str] = None
    # Clinical data
    age: Optional[int] = None
    sex: Optional[str] = None
    laterality: Optional[str] = None
    disease_duration: Optional[float] = None
    levodopa: Optional[str] = None
    last_dose: Optional[str] = None
    dbs: Optional[str] = None
    fluctuations: Optional[str] = None
    tremor: Optional[str] = None
    dysmetria: Optional[str] = None
    myoclonus: Optional[str] = None


class SubjectResponse(BaseModel):
    id: int
    name: str
    stage: str
    stage_idx: int
    iteration: int
    camera_mode: str = "stereo"
    camera_name: Optional[str]
    dlc_dir: Optional[str]
    diagnosis: str = "Control"
    video_count: int = 0
    has_snapshots: bool = False
    has_labels: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SubjectDetail(SubjectResponse):
    videos: List[str] = []
    trials: List[str] = []
    segments: List[dict] = []
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


class V2Frame(BaseModel):
    frame_num: int
    side: str


class CommitRequest(BaseModel):
    # Refine mode: explicit list of correction frames to include in DLC training.
    # Derived from corrections CSV vs DLC CSV diff on the frontend.
    v2_train_frames: List[V2Frame] = []
