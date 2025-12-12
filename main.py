from __future__ import annotations

import os
import time
import uuid
import random
from typing import Dict, Optional, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTasks
from pydantic import BaseModel, Field

from openai import OpenAI


# =========================================================
# Fixed SYSTEM PROMPT (서버 고정 기준)
# =========================================================
SYSTEM_PROMPT = """You are creating a video-generation prompt optimized for Minimax (Hailuo).

The user provides a single Korean sentence.
Assume this sentence is being read aloud as narration in a documentary or real-case introduction program.

Your task is to generate an AI video prompt that visually supports the sentence,
so that when the narration and the video are experienced together,
the scene feels natural, understandable, and emotionally coherent,
even if the sentence is abstract, brief, or incomplete.

Core requirements (must always apply):

- The video should generally align with documentary or real-incident storytelling,
  but may adopt restrained cinematic or dramatic visual language when it better supports the sentence.
- The tone must remain grounded and serious, avoiding fantasy, science fiction, or exaggerated symbolism.
- The camera represents a neutral observer, not a character,
  but framing, pacing, and lighting may be expressive if appropriate.
- Visually suggest the situation implied by the sentence rather than literally illustrating it.
- Do not introduce story elements that contradict the sentence.
- When details are missing, choose realistic, broadly believable environments
  suitable for documentary or dramatized reenactment contexts.
- Convey human emotion through atmosphere, distance, movement, lighting, or absence,
  never through explicit acting, facial close-ups, or melodrama.
- Camera movement should be minimal, controlled, and intentional
  (static shots, slow push-ins, gentle pans when needed).
- The scene should feel like a single continuous observational moment.

Length constraint (critical):

- The final prompt must be no more than 1800 characters.
- If the prompt risks exceeding this limit, compress aggressively:
  - Remove non-essential adjectives first
  - Prefer one environment, one mood, and one primary camera behavior
  - Preserve clarity, tone, and visual coherence over detail richness

Output rules:

- Write a single cohesive English prompt for AI video generation.
- Focus only on visual composition, environment, atmosphere, lighting, and camera behavior.
- Do not explain the prompt.
- Do not include narration text, subtitles, dialogue, or on-screen text.
- Do not mention the words “narration” or “voiceover”.
"""

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Minimax config
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = "https://api.minimax.io"
DEFAULT_MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-Hailuo-2.3")


# =========================================================
# OpenAI Client Factory (환경변수 안전 처리)
# =========================================================
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)


def require_minimax_key() -> None:
    if not MINIMAX_API_KEY:
        raise RuntimeError("MINIMAX_API_KEY environment variable is not set")


# =========================================================
# Helpers
# =========================================================
def now_ts() -> float:
    return time.time()


MAX_PROMPT_CHARS = 1800  # Minimax prompt 2000 chars 제한 대비 안전마진


def clamp_prompt(text: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """Minimax prompt length safety clamp."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    last_break = max(cut.rfind("."), cut.rfind(";"), cut.rfind(","), cut.rfind("\n"))
    if last_break >= int(max_chars * 0.7):
        cut = cut[: last_break + 1]
    return cut.strip()


# =========================================================
# FastAPI
# =========================================================
app = FastAPI(title="StoryVision AI Backend - MVP (Prompt + Video)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# In-memory job stores (DB 없음)
# =========================================================
JobStatus = Literal["PENDING", "PROCESSING", "DONE", "ERROR", "EXPIRED"]

# Prompt jobs
class PromptJob(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float

    user_sentence: str
    video_prompt: Optional[str] = None
    error: Optional[str] = None


PROMPT_JOBS: Dict[str, PromptJob] = {}
PROMPT_JOB_TTL_SECONDS = 60 * 30  # 30분

# Video jobs
class VideoJob(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float

    prompt: str
    model: str
    duration: int
    resolution: str
    prompt_optimizer: bool = True
    fast_pretreatment: bool = False

    minimax_task_id: Optional[str] = None
    minimax_status: Optional[str] = None  # Preparing/Queueing/Processing/Success/Fail
    file_id: Optional[str] = None

    video_width: Optional[int] = None
    video_height: Optional[int] = None

    # Minimax files/retrieve 결과
    download_url: Optional[str] = None
    filename: Optional[str] = None
    bytes: Optional[int] = None

    error: Optional[str] = None

    # 내부 제어용(폴링 과다 방지)
    _last_poll: float = 0.0


VIDEO_JOBS: Dict[str, VideoJob] = {}
VIDEO_JOB_TTL_SECONDS = 60 * 60 * 6  # 6시간
MIN_POLL_INTERVAL_SEC = 1.5


def gc_prompt_jobs() -> None:
    t = now_ts()
    expired = [job_id for job_id, job in PROMPT_JOBS.items() if t - job.created_at > PROMPT_JOB_TTL_SECONDS]
    for job_id in expired:
        PROMPT_JOBS.pop(job_id, None)


def gc_video_jobs() -> None:
    t = now_ts()
    expired = [job_id for job_id, job in VIDEO_JOBS.items() if t - job.created_at > VIDEO_JOB_TTL_SECONDS]
    for job_id in expired:
        VIDEO_JOBS.pop(job_id, None)


# =========================================================
# API Models (Prompt Jobs)
# =========================================================
class CreatePromptJobRequest(BaseModel):
    sentence: str = Field(..., min_length=1)


class CreatePromptJobResponse(BaseModel):
    job_id: str
    status: JobStatus


class GetPromptJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float

    user_sentence: str
    video_prompt: Optional[str] = None
    error: Optional[str] = None


# =========================================================
# GPT 호출 로직
# =========================================================
def call_gpt_make_video_prompt(user_sentence: str) -> str:
    user_sentence = user_sentence.strip()

    user_message = f'''User sentence:
"{user_sentence}"

Return ONLY the final English video-generation prompt.'''

    client = get_openai_client()

    max_tries = 4
    base_delay = 0.8
    last_error = None

    for i in range(max_tries):
        try:
            res = client.responses.create(
                model=DEFAULT_MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_output_tokens=260,
            )

            output = (res.output_text or "").strip()
            if not output:
                raise RuntimeError("Empty response from OpenAI")

            output = clamp_prompt(output, MAX_PROMPT_CHARS)
            return output

        except Exception as e:
            last_error = e
            time.sleep(base_delay * (2 ** i) + random.uniform(0, 0.3))

    raise RuntimeError(f"OpenAI request failed: {last_error}")


def run_prompt_job(job_id: str, sentence: str) -> None:
    try:
        job = PROMPT_JOBS.get(job_id)
        if not job:
            return

        prompt = call_gpt_make_video_prompt(sentence)
        job.video_prompt = prompt
        job.status = "DONE"
        job.updated_at = now_ts()

    except Exception as e:
        job = PROMPT_JOBS.get(job_id)
        if job:
            job.status = "ERROR"
            job.error = str(e)
            job.updated_at = now_ts()


# =========================================================
# Minimax API Calls
# =========================================================
async def minimax_create_task(model: str, prompt: str, duration: int, resolution: str,
                             prompt_optimizer: bool, fast_pretreatment: bool) -> dict:
    require_minimax_key()
    url = f"{MINIMAX_BASE_URL}/v1/video_generation"
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "duration": duration,
        "resolution": resolution,
        "prompt_optimizer": prompt_optimizer,
        "fast_pretreatment": fast_pretreatment,
    }

    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(url, headers=headers, json=payload)

    data = r.json() if r.content else {}
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Minimax HTTP {r.status_code}: {data}")

    base = data.get("base_resp") or {}
    if base.get("status_code") != 0:
        raise HTTPException(status_code=400, detail=f"Minimax error {base.get('status_code')}: {base.get('status_msg')}")

    task_id = data.get("task_id")
    if not task_id:
        raise HTTPException(status_code=502, detail=f"Minimax response missing task_id: {data}")

    return data


async def minimax_query_task(task_id: str) -> dict:
    require_minimax_key()
    url = f"{MINIMAX_BASE_URL}/v1/query/video_generation"
    headers = {"Authorization": f"Bearer {MINIMAX_API_KEY}"}
    params = {"task_id": task_id}

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url, headers=headers, params=params)

    data = r.json() if r.content else {}
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Minimax HTTP {r.status_code}: {data}")

    base = data.get("base_resp") or {}
    if base.get("status_code") != 0:
        raise HTTPException(status_code=400, detail=f"Minimax error {base.get('status_code')}: {base.get('status_msg')}")

    return data


async def minimax_retrieve_file(file_id: str) -> dict:
    require_minimax_key()
    url = f"{MINIMAX_BASE_URL}/v1/files/retrieve"
    headers = {"Authorization": f"Bearer {MINIMAX_API_KEY}"}
    params = {"file_id": file_id}

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url, headers=headers, params=params)

    data = r.json() if r.content else {}
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Minimax HTTP {r.status_code}: {data}")

    base = data.get("base_resp") or {}
    if base.get("status_code") != 0:
        raise HTTPException(status_code=400, detail=f"Minimax error {base.get('status_code')}: {base.get('status_msg')}")

    return data


# =========================================================
# API Models (Video Jobs)
# =========================================================
class CreateVideoJobRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="English prompt (<=2000 chars, we clamp to 1800)")
    model: str = Field(default=DEFAULT_MINIMAX_MODEL)
    duration: int = Field(default=6)
    resolution: str = Field(default="768P")
    prompt_optimizer: bool = Field(default=True)
    fast_pretreatment: bool = Field(default=False)


class CreateVideoJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    minimax_task_id: Optional[str] = None


class GetVideoJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float

    prompt: str
    model: str
    duration: int
    resolution: str
    prompt_optimizer: bool
    fast_pretreatment: bool

    minimax_task_id: Optional[str] = None
    minimax_status: Optional[str] = None
    file_id: Optional[str] = None

    video_width: Optional[int] = None
    video_height: Optional[int] = None

    # ✅ 프론트 호환 필드
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None  # (지금은 미지원이면 None)
    original_prompt: Optional[str] = None

    # 기존 유지(디버깅)
    download_url: Optional[str] = None
    filename: Optional[str] = None
    bytes: Optional[int] = None

    error: Optional[str] = None


# =========================================================
# Routes
# =========================================================
@app.get("/health")
def health():
    gc_prompt_jobs()
    gc_video_jobs()
    return {"ok": True}


# ---------- Prompt Jobs ----------
@app.post("/prompt-jobs", response_model=CreatePromptJobResponse)
def create_prompt_job(req: CreatePromptJobRequest, bg: BackgroundTasks):
    gc_prompt_jobs()

    sentence = req.sentence.strip()
    if not sentence:
        raise HTTPException(status_code=400, detail="sentence is empty")

    job_id = uuid.uuid4().hex
    t = now_ts()

    PROMPT_JOBS[job_id] = PromptJob(
        job_id=job_id,
        status="PENDING",
        created_at=t,
        updated_at=t,
        user_sentence=sentence,
    )

    bg.add_task(run_prompt_job, job_id, sentence)
    return CreatePromptJobResponse(job_id=job_id, status="PENDING")


@app.get("/prompt-jobs/{job_id}", response_model=GetPromptJobResponse)
def get_prompt_job(job_id: str):
    gc_prompt_jobs()

    job = PROMPT_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found (or expired)")

    return GetPromptJobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        user_sentence=job.user_sentence,
        video_prompt=job.video_prompt,
        error=job.error,
    )


# ---------- Video Jobs ----------
@app.post("/video-jobs", response_model=CreateVideoJobResponse)
async def create_video_job(req: CreateVideoJobRequest):
    gc_video_jobs()

    prompt = clamp_prompt(req.prompt, MAX_PROMPT_CHARS)
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is empty after trimming")

    job_id = uuid.uuid4().hex
    t = now_ts()

    VIDEO_JOBS[job_id] = VideoJob(
        job_id=job_id,
        status="PENDING",
        created_at=t,
        updated_at=t,
        prompt=prompt,
        model=req.model,
        duration=req.duration,
        resolution=req.resolution,
        prompt_optimizer=req.prompt_optimizer,
        fast_pretreatment=req.fast_pretreatment,
    )

    try:
        mm = await minimax_create_task(
            model=req.model,
            prompt=prompt,
            duration=req.duration,
            resolution=req.resolution,
            prompt_optimizer=req.prompt_optimizer,
            fast_pretreatment=req.fast_pretreatment,
        )

        job = VIDEO_JOBS[job_id]
        job.minimax_task_id = mm["task_id"]
        job.status = "PROCESSING"
        job.updated_at = now_ts()

        # ✅ 상태 일치
        return CreateVideoJobResponse(job_id=job_id, status="PROCESSING", minimax_task_id=job.minimax_task_id)

    except Exception as e:
        job = VIDEO_JOBS.get(job_id)
        if job:
            job.status = "ERROR"
            job.error = str(e)
            job.updated_at = now_ts()
        raise


@app.get("/video-jobs/{job_id}", response_model=GetVideoJobResponse)
async def get_video_job(job_id: str):
    gc_video_jobs()

    job = VIDEO_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found (or expired)")

    # DONE/ERROR면 그대로 리턴(프론트 호환 필드 포함)
    if job.status in ("DONE", "ERROR"):
        base = job.model_dump(exclude={"_last_poll"})
        return GetVideoJobResponse(
            **base,
            video_url=job.download_url,
            thumbnail_url=None,
            original_prompt=job.prompt,
        )

    # 폴링 과다 방지
    now = now_ts()
    if now - job._last_poll < MIN_POLL_INTERVAL_SEC:
        base = job.model_dump(exclude={"_last_poll"})
        return GetVideoJobResponse(
            **base,
            video_url=job.download_url,
            thumbnail_url=None,
            original_prompt=job.prompt,
        )
    job._last_poll = now

    try:
        if not job.minimax_task_id:
            raise HTTPException(status_code=400, detail="minimax_task_id missing")

        st = await minimax_query_task(job.minimax_task_id)

        mm_status = st.get("status")
        job.minimax_status = mm_status
        job.updated_at = now_ts()

        if mm_status == "Success":
            job.file_id = st.get("file_id")
            job.video_width = st.get("video_width")
            job.video_height = st.get("video_height")

            if job.file_id:
                fdata = await minimax_retrieve_file(job.file_id)
                f = (fdata.get("file") or {})
                job.download_url = f.get("download_url")
                job.filename = f.get("filename")
                job.bytes = f.get("bytes")

            job.status = "DONE"
            job.updated_at = now_ts()

        elif mm_status == "Fail":
            job.status = "ERROR"
            job.error = "Minimax task failed"
            job.updated_at = now_ts()

        else:
            job.status = "PROCESSING"
            job.updated_at = now_ts()

        base = job.model_dump(exclude={"_last_poll"})
        return GetVideoJobResponse(
            **base,
            video_url=job.download_url,
            thumbnail_url=None,
            original_prompt=job.prompt,
        )

    except Exception as e:
        job.status = "ERROR"
        job.error = str(e)
        job.updated_at = now_ts()

        base = job.model_dump(exclude={"_last_poll"})
        return GetVideoJobResponse(
            **base,
            video_url=job.download_url,
            thumbnail_url=None,
            original_prompt=job.prompt,
        )
