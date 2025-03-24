# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Union
import os
import sys
import argparse
import logging
import re
from starlette.routing import Mount
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, Form, File, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import torch
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

logger = logging.getLogger("cosyvoice2")
logger.setLevel(logging.INFO)

LONG_MIN = torch.iinfo(torch.long).min
LONG_MAX = torch.iinfo(torch.long).max

class CompletionRequest(BaseModel):
    model: Optional[str] = "cosyvoice2"
    frequency_penalty: Optional[float] = 0.0
    max_tokens: Optional[int] = 1500
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = Field(None, ge=LONG_MIN, le=LONG_MAX)
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0

    use_beam_search: bool = False
    top_k: Optional[int] = -1
    min_p: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    length_penalty: float = 1.0
    min_tokens: int = 0


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(),
    tts_language: Optional[str] = Form(None),
    prompt_text: str = Form(),
    prompt_language: Optional[str] = Form(None),
    prompt_wav: UploadFile = File(),
    llm_request: CompletionRequest = Depends(CompletionRequest),
):
    try:
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)
        llm_request = dict(llm_request)

        if tts_language:
            tts_text = f"<|{tts_language}|>{tts_text}"

        if prompt_language:
            prompt_text = f"<|{prompt_language}|>{prompt_text}"

        model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, llm_request, text_frontend=False)
        return StreamingResponse(generate_data(model_output))
    except RuntimeError as e:
        logger.error(f"RuntimeError in inference_zero_shot: {str(e)}")
        # Return a proper error response with status code
        return JSONResponse(
            status_code=500,
            content={"error": "Runtime error during inference", "message": str(e)}
        )


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))

@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


def mount_metrics(app: FastAPI):
    from prometheus_client import (CollectorRegistry, make_asgi_app,
                                   multiprocess)
    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app())

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    parser.add_argument('--tokenizer_model_dir',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    parser.add_argument('--vllm_endpoint',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    mount_metrics(app)
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.vllm_endpoint, args.tokenizer_model_dir, args.model_dir)
        except Exception:
            raise TypeError('no valid model_type!')
    uvicorn.run(app, host="0.0.0.0", port=args.port)
