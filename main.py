from __future__ import annotations

import os
import time
import uuid
import random
from typing import Dict, Optional, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTasks
from pydantic import BaseModel, Field

from openai import OpenAI


# =========================================================
# OpenAI Client (키는 환경변수로만)
# - Render / 로컬 환경변수: OPENAI_API_KEY
# - 예: OPENAI_API_KEY="MyKey"  (실제 값은 절대 코드에 하드코딩 금지)
# =========================================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================================================
# Fixed SYSTEM PROMPT (서버 고정 기준)
# =========================================================
SYSTEM_PROMPT = """You are creating a video-generation prompt optimized for Minimax (Hailuo).

The user provides a single Korean sentence.
Assume this sentence is being read aloud as narration in a documentary or real-case introduction program.

Your task is to generate an AI video prompt that visually supports the narration,
so that when the narration and the video are played together,
the scene feels natural, understandable, and emotionally coherent,
even if the sentence is abstract, brief, or incomplete.

Core requirements (must always apply):

- The video should generally align with documentary or real-incident storytelling,
  but may adopt cinematic or dramatic visual language when it better supports the sentence.
- The tone should feel grounded and serious, avoiding fantasy or exaggerated symbolism,
  while allowing mood, tension, or emotional weight through visual composition.
- The camera represents a neutral observer, not a character,
  but framing and lighting may be expressive if appropriate.
- The scene should visually interpret the situation implied by the sentence,
  rather than literally illustrating or explaining the text.
- Do not add story details that clearly contradict the sentence,
  but allow reasonable visual interpretation when details are abstract or emotional.
- If specific details are missing, choose realistic or broadly believable environments
  that feel appropriate for documentary or dramatized reenactment contexts.
- Human emotion should be conveyed through atmosphere, movement, distance,
  lighting, or absence, rather than explicit acting or facial close-ups.
- Avoid fantasy, science fiction, or overtly stylized visuals,
  but subtle cinematic tension and drama are allowed.
- Lighting may be naturalistic or deliberately moody, depending on the sentence.
- Camera movement should be controlled and intentional,
  favoring stable or slow cinematic motion over chaotic movement.
- The video should feel like a single, continuous observational moment,
  even if the visual tone is dramatic.

Output rules:

- Write a single cohesive English prompt for AI video generation.
- Focus only on visual composition, atmosphere, environment, lighting, and camera behavior.
- Do not explain the prompt.
- Do not include narration text, subtitles, or dialogue.
- Do not mention the word “narration” or “voiceover” in the output.
"""

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


# =========================================================
# FastAPI
# =========================================================
app = FastAPI(title="StoryVision AI Backend - Prompt Generator (MVP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요시 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# In-memory job store (DB 0)
# =========================================================
JobStatus = Literal["PENDING", "DONE", "ERROR", "EXPIRED"]

class PromptJob(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float

    user_sentence: str
    video_prompt: Optional[str] = None
    error: Optional[str] = None

JOBS: Dict[str, PromptJob] = {}
JOB_TTL_SECONDS = 60 * 30  # 30분


def now_ts() -> float:
    return time.time()


def gc_jobs() -> None:
    t = now_ts()
    expired_ids = []
    for job_id, job in JOBS.items():
        if t - job.created_at > JOB_TTL_SECONDS:
            expired_ids.append(job_id)
    for job_id in expired_ids:
        JOBS.pop(job_id, None)


# =========================================================
# API Models
# =========================================================
class CreatePromptJobRequest(BaseModel):
    sentence: str = Field(..., min_length=1, description="사용자 한글 문장")


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
# OpenAI call with simple retry (rate limit/network 대비)
# =========================================================
def call_gpt_make_video_prompt(user_sentence: str) -> str:
    """
    Returns: Minimax(Hailuo)용 영어 1문단 프롬프트
    """
    # 안전하게 따옴표/개행 정리
    user_sentence = user_sentence.strip()

    user_message = f'User sentence:\n"{user_sentence}"\n\nReturn ONLY the final English video-generation prompt.'

    # 지수 백오프 재시도 (429/일시 장애)
    max_tries = 4
    base = 0.8

    last_err = None
    for i in range(max_tries):
        try:
            res = client.responses.create(
                model=DEFAULT_MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                # 프롬프트 길이 과도 방지 (대략)
                max_output_tokens=220,
            )

            text = (res.output_text or "").strip()

            # 마지막 방어: 빈 문자열이면 에러 처리
            if not text:
                raise RuntimeError("Empty output from model")

            # 혹시 모델이 따옴표/설명을 붙이면 최소 정리(설명문이 길게 섞이면 다시 보완 가능)
            # 여기서는 '한 문단 영어 프롬프트'를 기대.
            return text

        except Exception as e:
            last_err = e
            # backoff + jitter
            sleep_s = (base * (2 ** i)) + random.uniform(0, 0.25)
            time.sleep(sleep_s)

    raise RuntimeError(f"OpenAI call failed after retries: {last_err}")


def run_prompt_job(job_id: str, sentence: str) -> None:
    try:
        job = JOBS.get(job_id)
        if not job:
            return

        video_prompt = call_gpt_make_video_prompt(sentence)

        job.video_prompt = video_prompt
        job.status = "DONE"
        job.updated_at = now_ts()
        JOBS[job_id] = job

    except Exception as e:
        job = JOBS.get(job_id)
        if job:
            job.status = "ERROR"
            job.error = str(e)
            job.updated_at = now_ts()
            JOBS[job_id] = job


# =========================================================
# Routes
# =========================================================
@app.get("/health")
def health():
    gc_jobs()
    return {"ok": True}


@app.post("/prompt-jobs", response_model=CreatePromptJobResponse)
def create_prompt_job(req: CreatePromptJobRequest, bg: BackgroundTasks):
    gc_jobs()

    sentence = req.sentence.strip()
    if not sentence:
        raise HTTPException(status_code=400, detail="sentence is empty")

    job_id = uuid.uuid4().hex
    t = now_ts()

    JOBS[job_id] = PromptJob(
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
    gc_jobs()

    job = JOBS.get(job_id)
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
