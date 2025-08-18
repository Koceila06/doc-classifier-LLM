# -*- coding: utf-8 -*-

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from utils import save_upload_file_tmp, classify_image, load_prompt
import os

app = FastAPI(title="DocClassifier API - POC")
API_KEY = os.getenv("OPENAI_API_KEY")

async def _generic_classify_endpoint(file: UploadFile, doc_key: str):
    prompt_text = load_prompt(doc_key)
    if not prompt_text:
        raise HTTPException(status_code=400, detail="Type de document non supporté")

    tmp_path = save_upload_file_tmp(file)
    try:
        label = classify_image(tmp_path, prompt_text, API_KEY)
        return JSONResponse({"category": label})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/classify/{doc_key}")
async def classify(file: UploadFile = File(...), doc_key: str = "permis"):
    return await _generic_classify_endpoint(file, doc_key)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
