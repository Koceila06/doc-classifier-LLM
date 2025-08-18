# -*- coding: utf-8 -*-

import openai
import base64
import tempfile
import cv2
from PIL import Image, UnidentifiedImageError
import unicodedata
import re
import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from openai import OpenAI
import yaml

openai.api_key = os.getenv("OPENAI_API_KEY")
RETRY_COUNT = 3
RETRY_DELAY = 2
app = FastAPI(title="DocClassifier API - POC")


def save_upload_file_tmp(upload_file: UploadFile) -> str:
    suffix = ""
    if upload_file.filename and "." in upload_file.filename:
        suffix = "." + upload_file.filename.rsplit(".", 1)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = upload_file.file.read()
        tmp.write(content)
        return tmp.name

def remove_accents(input_str):
    return ''.join(
        c for c in unicodedata.normalize('NFD', input_str)
        if unicodedata.category(c) != 'Mn'
    )

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def normalize_classe(predicted_label):
    label = predicted_label.strip().lower().strip(' "\'`')
    label = ''.join(c for c in unicodedata.normalize('NFD', label) if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^\w]', '', label)



def classify_image(image_path, prompt_text, api_key):
    try:
        with Image.open(image_path) as img:
            img.verify()
    except (UnidentifiedImageError, IOError):
        return "error"

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image introuvable ou illisible.")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, image)

    # Encodage base64
    base64_image = encode_image(temp_path)

    # Suppression fichier temporaire
    os.remove(temp_path)
    client = OpenAI(api_key = api_key )    # Envoi à l’API avec gestion des erreurs
    for attempt in range(RETRY_COUNT):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"}}]
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            return normalize_classe(response.choices[0].message.content.strip().lower())

        except openai.RateLimitError:
            # print("Rate limit atteint. Pause 1 seconde...")
            time.sleep(5)

        except Exception as e:
            print(f"Erreur API : {e}")
            return "error"


def load_prompt(doc_key):
    yaml_path = f"prompts/{doc_key}.yaml"
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    return content.get("text", "")
