FROM vllm/vllm-openai:v0.7.3

WORKDIR /workspace/CosyVoice
COPY . /workspace/CosyVoice

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEN=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update -y --fix-missing
RUN apt-get install -y git build-essential curl wget ffmpeg unzip git git-lfs sox libsox-dev pipx && \
    apt-get clean && \
    git lfs install

RUN pipx install poetry==2.0.0

ENV PATH="${PATH}:/root/.local/bin"

RUN poetry install --no-interaction --no-ansi && \
    VENV_PATH=$(poetry env info --path) && \
    echo "source ${VENV_PATH}/bin/activate" >> /opt/nvidia/entrypoint.d/100.poetry.sh && \
    echo "source ${VENV_PATH}/bin/activate" >> ~/.bashrc && \
    echo "source ${VENV_PATH}/bin/activate" >> /opt/nvidia/entrypoint.d/110.poetry_default_env.sh && \
    echo "source ${VENV_PATH}/bin/activate" >> $HOME/.bashrc

ENV PYTHONPATH="${PYTHONPATH}:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"
