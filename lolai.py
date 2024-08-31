import os.path

from model import SenseVoiceSmall
import librosa.feature
import numpy as np
from dotenv import load_dotenv

load_dotenv(".env")

model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
m.eval()

def convert_range(x):
    x = np.clip(x, 0.018, 0.3)
    a, b = (0.018, 0.3)
    c, d = (np.pi/20, np.pi/2)
    return c + (x - a) * (d - c) / (b - a)

def evaluate_internal(file_or_uri):
    results, meta_data = m.inference(file_or_uri, language="auto", use_itn=False, ban_emo_unk=True, **kwargs)
    res = results[0]["text"]
    score = np.int32(0)
    if "<|HAPPY|><|Laughter|>" in str(res) or "<|HAPPY|><|Speech|>" in str(res):
        rms = librosa.feature.rms(y=meta_data["audio_sample"])[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        if rms_mean > 0.018 and rms_std > 0.015:
            mean = np.mean([np.sin(convert_range(rms_mean)), np.sin(convert_range(rms_std))])
            score = (np.round(mean, 2) * 100).astype(int)

    return res, score

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Payload(BaseModel):
    filename: str

class ScoreResponse(BaseModel):
    code: int
    data: dict

@app.post("/evaluate", response_model=ScoreResponse)
def evaluate(payload: Payload):
    uri = os.getenv("audio_url") + payload.filename
    _, score = evaluate_internal(uri)
    return {"code": 200, "data": {"score": score.item()}}

# test locally
# for root, dirs, files in os.walk(f"{os.curdir}/happy"):
#     for file in files:
#         file_path = os.path.join(root, file)
#         res, score = _evaluate(file_path)
#         print("{:<3}".format(score), file, res)

uvicorn.run(app, host="0.0.0.0", port=3000)
