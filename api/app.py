from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    CreateRunResponse,
    HealthResponse,
    RunListResponse,
    RunSummaryResponse,
)
from api.services import (
    get_run_summary,
    list_runs,
    prepare_effective_config,
    run_pipeline,
)

app = FastAPI(title="Bias Hunter API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/runs", response_model=RunListResponse)
def get_runs() -> RunListResponse:
    return RunListResponse(runs=list_runs())


@app.get("/runs/{run_id}", response_model=RunSummaryResponse)
def get_run(run_id: str) -> RunSummaryResponse:
    try:
        return RunSummaryResponse(**get_run_summary(run_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/runs", response_model=CreateRunResponse)
async def create_run(
    config_file: UploadFile = File(...),
    dataset_file: Optional[UploadFile] = File(default=None),
) -> CreateRunResponse:
    if not config_file.filename:
        raise HTTPException(status_code=400, detail="Config file must have a filename.")

    config_bytes = await config_file.read()
    dataset_bytes = await dataset_file.read() if dataset_file is not None else None

    try:
        effective_config_path = prepare_effective_config(
            config_bytes=config_bytes,
            config_filename=config_file.filename,
            dataset_bytes=dataset_bytes,
            dataset_filename=dataset_file.filename if dataset_file is not None else None,
        )
        result = run_pipeline(effective_config_path)
        return CreateRunResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
