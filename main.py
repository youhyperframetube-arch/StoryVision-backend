from __future__ import annotations

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(title="StoryVision AI Backend - MVP (Job + Polling)")

# CORS (일단 전체 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Models
# -------------------------

class GeneratePromptsRequest(BaseModel):
    lines: List[str]


class PromptItem(BaseModel):
    id: int
    line_ko: str
    prompt_en: str


class GeneratePromptsResponse(BaseModel):
    items: List[PromptItem]


class SceneMeta(BaseModel):
    duration: Optional[float] = None
    resolution: Optional[str] = None
    model: Optional[str] = None
    version: Optional[str] = None
    original_prompt: Optional[str] = None


SceneStatus = Literal["pending", "running", "succeeded", "failed"]


class SceneState(BaseModel):
    id: int
    line_ko: str
    prompt_en: str

    status: SceneStatus = "pending"
    progress: int = 0  # 0~100

    # 결과
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None

    # 메타
    meta: SceneMeta = Field(default_factory=SceneMeta)

    # 에러
    error: Optional[str] = None


JobStatus = Literal["queued", "running", "succeeded", "failed"]


class CreateJobRequest(BaseModel):
    items: List[PromptItem]


class CreateJobResponse(BaseModel):
    job_id: str


class JobState(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float

    total: int
    done: int
    progress: int  # 0~100

    scenes: List[SceneState]


# -------------------------
# In-memory store (DB 없음)
# -------------------------
JOBS: Dict[str, JobState] = {}
JOB_TASKS: Dict[str, asyncio.Task] = {}

# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------
# 1) Prompt generation (demo)
# -------------------------
@app.post("/generate-prompts", response_model=GeneratePromptsResponse)
def generate_prompts(body: GeneratePromptsRequest):
    if not body.lines:
        raise HTTPException(status_code=400, detail="lines가 비어 있습니다.")

    cleaned = [ln.strip() for ln in body.lines if ln.strip()]
    if not cleaned:
        raise HTTPException(status_code=400, detail="유효한 문장이 없습니다.")

    items: List[PromptItem] = []
    for idx, line in enumerate(cleaned, start=1):
        fake_prompt = (
            f'Documentary-style cinematic shot for: "{line[:40]}..." '
            f"in a dark, mysterious atmosphere, subtle camera movement, gloomy lighting."
        )
        items.append(PromptItem(id=idx, line_ko=line, prompt_en=fake_prompt))

    return GeneratePromptsResponse(items=items)


# -------------------------
# 2) Create Job
# -------------------------
@app.post("/jobs", response_model=CreateJobResponse)
async def create_job(body: CreateJobRequest):
    if not body.items:
        raise HTTPException(status_code=400, detail="items가 비어 있습니다.")

    job_id = uuid.uuid4().hex[:12]
    now = time.time()

    scenes = [
        SceneState(
            id=item.id,
            line_ko=item.line_ko,
            prompt_en=item.prompt_en,
            status="pending",
            progress=0,
            meta=SceneMeta(
                model=None,
                version=None,
                original_prompt=item.prompt_en,
            ),
        )
        for item in body.items
    ]

    job = JobState(
        job_id=job_id,
        status="queued",
        created_at=now,
        updated_at=now,
        total=len(scenes),
        done=0,
        progress=0,
        scenes=scenes,
    )

    JOBS[job_id] = job

    # 백그라운드 렌더 시작 (DB 없이 메모리에 상태 갱신)
    task = asyncio.create_task(run_job(job_id))
    JOB_TASKS[job_id] = task

    return CreateJobResponse(job_id=job_id)


# -------------------------
# 3) Poll Job
# -------------------------
@app.get("/jobs/{job_id}", response_model=JobState)
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id를 찾을 수 없습니다.")
    return job


# -------------------------
# 4) Retry one scene
# -------------------------
@app.post("/jobs/{job_id}/scenes/{scene_id}/retry")
async def retry_scene(job_id: str, scene_id: int):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id를 찾을 수 없습니다.")

    scene = next((s for s in job.scenes if s.id == scene_id), None)
    if not scene:
        raise HTTPException(status_code=404, detail="scene_id를 찾을 수 없습니다.")

    # 상태 초기화
    scene.status = "pending"
    scene.progress = 0
    scene.video_url = None
    scene.thumbnail_url = None
    scene.error = None
    job.updated_at = time.time()

    # job이 이미 완료였다면 running으로 되돌림
    job.status = "running"

    # 해당 scene만 다시 돌리는 태스크를 붙임
    asyncio.create_task(run_one_scene(job_id, scene_id))

    return {"ok": True}


# -------------------------
# Job runner (현재는 더미)
# 나중에 여기만 Minimax 호출로 교체하면 됨
# -------------------------
async def run_job(job_id: str):
    job = JOBS[job_id]
    job.status = "running"
    job.updated_at = time.time()

    for scene in job.scenes:
        await run_scene_object(job, scene)

        # 진행률 갱신
        done = sum(1 for s in job.scenes if s.status == "succeeded")
        job.done = done
        job.progress = int(done / job.total * 100)
        job.updated_at = time.time()

    # 최종 상태
    if any(s.status == "failed" for s in job.scenes):
        job.status = "failed"
    else:
        job.status = "succeeded"
        job.progress = 100
    job.updated_at = time.time()


async def run_one_scene(job_id: str, scene_id: int):
    job = JOBS[job_id]
    scene = next((s for s in job.scenes if s.id == scene_id), None)
    if not scene:
        return

    await run_scene_object(job, scene)

    done = sum(1 for s in job.scenes if s.status == "succeeded")
    job.done = done
    job.progress = int(done / job.total * 100)
    job.updated_at = time.time()

    if done == job.total and not any(s.status == "failed" for s in job.scenes):
        job.status = "succeeded"
        job.progress = 100
        job.updated_at = time.time()


async def run_scene_object(job: JobState, scene: SceneState):
    # 더미 렌더: 0->100까지 진행률 올리고 결과 채움
    scene.status = "running"
    scene.progress = 0
    job.updated_at = time.time()

    try:
        for p in range(0, 101, 10):
            scene.progress = p
            job.updated_at = time.time()
            await asyncio.sleep(0.25)  # 렌더 시간이 걸리는 척

        scene.status = "succeeded"
        # 더미 결과 (나중엔 Minimax 결과로 교체)
        scene.video_url = f"https://example.com/video_{scene.id:03d}.mp4"
        scene.thumbnail_url = f"https://picsum.photos/seed/storyvision{scene.id}/200/120"
        scene.meta.duration = 10 + (scene.id % 7)
        scene.meta.resolution = "1080p"
        scene.meta.model = "t2v-beta"
        scene.meta.version = "v1"
        scene.meta.original_prompt = scene.prompt_en
        scene.error = None

    except Exception as e:
        scene.status = "failed"
        scene.error = str(e)
        scene.progress = 0
