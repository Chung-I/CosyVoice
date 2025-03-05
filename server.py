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
import os
import sys
import signal
import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from transformers.models.qwen2 import Qwen2ForCausalLM

import anyio
import uvloop
from vllm.entrypoints.launcher import serve_http
from vllm.utils import (
    FlexibleArgumentParser,
    set_ulimit,
    is_valid_ipv6_address,
)
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client,
    build_app,
    create_server_socket,
    init_app_state,
)
from vllm.version import __version__ as VLLM_VERSION

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = logging.getLogger("uvicorn")


async def generate_data(model_output):
    async for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio

def add_endpoints(app):
    @app.get("/inference_sft")
    @app.post("/inference_sft")
    async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
        model_output = app.state.model.inference_sft(tts_text, spk_id)
        return StreamingResponse(generate_data(model_output))


    @app.get("/inference_zero_shot")
    @app.post("/inference_zero_shot")
    async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)
        model_output = app.state.model.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, text_frontend=False)
        return StreamingResponse(generate_data(model_output))


    @app.get("/inference_cross_lingual")
    @app.post("/inference_cross_lingual")
    async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)
        model_output = app.state.model.inference_cross_lingual(tts_text, prompt_speech_16k)
        return StreamingResponse(generate_data(model_output))


    @app.get("/inference_instruct")
    @app.post("/inference_instruct")
    async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
        model_output = app.state.model.inference_instruct(tts_text, spk_id, instruct_text)
        return StreamingResponse(generate_data(model_output))

    @app.get("/inference_instruct2")
    @app.post("/inference_instruct2")
    async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)
        model_output = app.state.model.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
        return StreamingResponse(generate_data(model_output))


async def run_server(args, **uvicorn_kwargs) -> None:
    global app
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    nn_pool_size = int(os.environ.get("NN_POOL_SIZE", 100))

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)

        add_endpoints(app)

        app.state.model = CosyVoice2(engine_client, args.model_dir)

        model_config = await engine_client.get_model_config()
        await init_app_state(engine_client, model_config, app.state, args)

        def _listen_addr(a: str) -> str:
            if is_valid_ipv6_address(a):
                return '[' + a + ']'
            return a or "0.0.0.0"

        logger.info("Starting vLLM API server on http://%s:%d",
                    _listen_addr(sock_addr[0]), sock_addr[1])

        shutdown_task = await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = nn_pool_size

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task

    sock.close()


async def run_server_hf(args, **uvicorn_kwargs) -> None:
    global app
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    nn_pool_size = int(os.environ.get("NN_POOL_SIZE", 100))

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    app = FastAPI()

    add_endpoints(app)

    llm_model = Qwen2ForCausalLM.from_pretrained(args.model)

    app.state.model = CosyVoice2(llm_model, args.model_dir)

    # model_config = await engine_client.get_model_config()
    # await init_app_state(engine_client, model_config, app.state, args)

    def _listen_addr(a: str) -> str:
        if is_valid_ipv6_address(a):
            return '[' + a + ']'
        return a or "0.0.0.0"

    logger.info("Starting vLLM API server on http://%s:%d",
                _listen_addr(sock_addr[0]), sock_addr[1])

    shutdown_task = await serve_http(
        app,
        sock=sock,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = nn_pool_size

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task

    sock.close()


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    parser.add_argument("--model-dir", type=str, help="CosyVoice2 model dir")

    cosyvoice_model_path = os.environ["COSYVOICE_MODEL_DIR"]
    vllm_model_path = os.environ["VLLM_MODEL_PATH"]
    args = parser.parse_args(
        ["--model",
        vllm_model_path,
        "--max-num-seqs",
        "400",
        "--model-dir",
        cosyvoice_model_path,
        "--port",
        "8000",
        "--enforce-eager",
        ]
    )
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args, ws_ping_timeout=3600))
