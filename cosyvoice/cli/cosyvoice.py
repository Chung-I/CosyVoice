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
import logging
import os
import time
from typing import Generator
from tqdm import tqdm
import uuid

from hyperpyyaml import load_hyperpyyaml
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoice2Model
from vllm.inputs import TokensPrompt
from vllm import SamplingParams


class CosyVoice2:

    def __init__(self, llm, model_dir, load_jit=False, load_trt=False, fp16=False):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        self.sos_ids = torch.LongTensor([[151936]])
        self.task_ids = torch.LongTensor([[151937]])
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice-codec.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f) # overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoice2Model(configs['flow'], configs['hift'], fp16)
        self.model.load('{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))

        self.llm = llm
        # self.llm = self.llm.to(self.model.device)

        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                self.fp16)
        del configs

    async def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        this_uuid = str(uuid.uuid4())
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))

            prompt_text = model_input["prompt_text"]
            text = model_input["text"]
            llm_prompt_speech_token = model_input["llm_prompt_speech_token"]
            llm_prompt_speech_token = llm_prompt_speech_token + 151938

            self.sos_ids = self.sos_ids.to(prompt_text.device).to(prompt_text.dtype)
            self.task_ids = self.task_ids.to(prompt_text.device).to(prompt_text.dtype)

            prompt = torch.cat((self.sos_ids, prompt_text, text, self.task_ids, llm_prompt_speech_token), dim=1)

            vllm_prompt = TokensPrompt(
                prompt_token_ids=prompt.cpu().numpy().tolist()[0],
            )

            sampling_params = SamplingParams(
                n=1,
                max_tokens=2048 - len(prompt) - 2,
                temperature=0.8,
                top_p=1.0,
                stop_token_ids=[158499, 158501],
            )

            result_generator = self.llm.generate(
                vllm_prompt,
                sampling_params,
                request_id=this_uuid,
            )
            vllm_result = None
            async for op in result_generator:
                vllm_result = op
            llm_token = torch.LongTensor(vllm_result.outputs[0].token_ids[:-1]).unsqueeze(0).to(self.model.device)
            llm_token -= 151938
            print(llm_token)
            print(llm_token[llm_token >= 6561])
            # outputs = self.llm.generate(
            #     prompt.to(torch.int64),
            #     max_length=2048,  # We trained our model with a max length of 2048
            #     do_sample=True,
            #     eos_token_id=158499,
            #     max_new_tokens=2048,
            #     top_p=1,
            #     temperature=0.8,
            # )
            # outputs -= 151938
            # prompt_length = prompt.size(1)
            # llm_token = outputs[:, prompt_length:-1]

            tts_speech = self.model.token2wav(
                token=llm_token,
                prompt_token=model_input["flow_prompt_speech_token"],
                prompt_feat=model_input["prompt_speech_feat"],
                embedding=model_input["flow_embedding"],
                uuid=this_uuid,
                token_offset=0,
                finalize=True,
                speed=1.0,
            )
            speech_len = tts_speech.shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield {'tts_speech': tts_speech.cpu()}
            start_time = time.time()

    def inference_instruct(self, *args, **kwargs):
        raise NotImplementedError('inference_instruct is not implemented for CosyVoice2!')

    def inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoice2Model), 'inference_instruct2 is only implemented for CosyVoice2!'
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()
