from __future__ import annotations

import os
import time
import uuid
import random
from typing import Dict, Optional, Literal

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
# OpenAI Client Factory (환경변수 안전 처리)
# =========================================================
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)


# =========================================================
# FastAPI
# =========================================================
app = FastAPI(title="StoryVision AI Backend - Prompt Generator (MVP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# In-memory job store (DB 없음)
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
    expired = [
        job_id for job_id, job in JOBS.items()
        if t - job.created_at > JOB_TTL_SECONDS
    ]
    for job_id in expired:
        JOBS.pop(job_id, None)


# =========================================================
# API Models
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
                max_output_tokens=220,
            )

            output = (res.output_text or "").strip()
            if not output:
                raise RuntimeError("Empty response from OpenAI")

            return output

        except Exception as e:
            last_error = e
            time.sleep(base_delay * (2 ** i) + random.uniform(0, 0.3))

    raise RuntimeError(f"OpenAI request failed: {last_error}")


def run_prompt_job(job_id: str, sentence: str) -> None:
    try:
        job = JOBS.get(job_id)
        if not job:
            return

        prompt = call_gpt_make_video_prompt(sentence)

        job.video_prompt = prompt
        job.status = "DONE"
        job.updated_at = now_ts()

    except Exception as e:
        job = JOBS.get(job_id)
        if job:
            job.status = "ERROR"
            job.error = str(e)
            job.updated_at = now_ts()


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
