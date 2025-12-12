from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="StoryVision AI Backend - MVP")

# CORS: 일단은 다 열어두기 (나중에 도메인 제한 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 추후 "https://youhyperframetube-arch.github.io" 로 좁혀도 됨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === 요청/응답 모델 ===

class GeneratePromptsRequest(BaseModel):
    lines: List[str]


class PromptItem(BaseModel):
    id: int
    line_ko: str
    prompt_en: str


class GeneratePromptsResponse(BaseModel):
    items: List[PromptItem]


# === 헬스 체크용 ===

@app.get("/health")
def health():
    return {"status": "ok"}


# === 프롬프트 생성 (지금은 "가짜" 버전, GPT 없이) ===

@app.post("/generate-prompts", response_model=GeneratePromptsResponse)
def generate_prompts(body: GeneratePromptsRequest):
    """
    입력: 한국어 문장 리스트
    출력: 각 문장에 대응하는 '영상 프롬프트' (지금은 데모용 가짜)
    """
    if not body.lines:
        raise HTTPException(status_code=400, detail="lines가 비어 있습니다.")

    cleaned_lines = [ln.strip() for ln in body.lines if ln.strip()]
    if not cleaned_lines:
        raise HTTPException(status_code=400, detail="유효한 문장이 없습니다.")

    items: List[PromptItem] = []
    for idx, line in enumerate(cleaned_lines, start=1):
        fake_prompt = (
            f"Documentary-style cinematic shot for: \"{line[:40]}...\" "
            f"in a dark, mysterious atmosphere, subtle camera movement, gloomy lighting."
        )
        items.append(
            PromptItem(
                id=idx,
                line_ko=line,
                prompt_en=fake_prompt,
            )
        )

    return GeneratePromptsResponse(items=items)
