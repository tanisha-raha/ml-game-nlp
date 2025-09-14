# src/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uuid

from .queue import InMemoryQueue, Job
from .generator import GenConfig, build_prompt, generate_text
from .prompts import TEMPLATES

app = FastAPI(title="ML Game NLP API")

# global queue
job_queue = InMemoryQueue()

class ContentParams(BaseModel):
    place: str = "Ebonridge Keep"
    tone: str = "somber"
    level: int = 8
    biome: str = "Crystal Marsh"
    name: str = "Aetherglass Dagger"
    n: int = 8
    faction: str = "Verdant Wardens"

class GenParams(BaseModel):
    max_new_tokens: int = 120
    temperature: float = 0.9
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    seed: int = 42
    model: str = "distilgpt2"

class GenerateRequest(BaseModel):
    type: str = Field(pattern="^(lore|quest|item|names)$")
    params: ContentParams = ContentParams()
    gen: GenParams = GenParams()

def handle_job(payload: dict) -> dict:
    req = GenerateRequest(**payload)
    prompt = build_prompt(req.type, req.params.dict())
    cfg = GenConfig(
        max_new_tokens=req.gen.max_new_tokens,
        temperature=req.gen.temperature,
        top_p=req.gen.top_p,
        top_k=req.gen.top_k,
        repetition_penalty=req.gen.repetition_penalty,
        seed=req.gen.seed,
    )
    text = generate_text(req.gen.model, prompt, cfg)
    return {
        "type": req.type,
        "content_params": req.params.dict(),
        "gen_params": req.gen.dict(),
        "prompt": prompt,
        "text": text.strip(),
    }

# start background worker
job_queue.run_worker(handle_job)

@app.post("/generate")
def generate(req: GenerateRequest):
    job_id = str(uuid.uuid4())
    job = Job(id=job_id, payload=req.dict())
    job_queue.submit(job)
    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
def status(job_id: str):
    job = job_queue.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }

