FROM image-docker.zuoyebang.cc/public/nvidia/tensorrt-llm/release:1.0.0rc4

ENV WORKDIR=/home/homework \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_VISIBLE_DEVICES=all \
    OPENBLAS_NUM_THREADS=1

WORKDIR $WORKDIR

RUN mkdir -p models extra_config

# patch
COPY patch/openai_protocol.py /usr/local/lib/python3.12/dist-packages/tensorrt_llm/serve/openai_protocol.py
COPY patch/llm.py /usr/local/lib/python3.12/dist-packages/tensorrt_llm/llmapi/llm.py
COPY patch/openai_server.py /usr/local/lib/python3.12/dist-packages/tensorrt_llm/serve/openai_server.py
COPY patch/modeling_utils.py /usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/models/modeling_utils.py

COPY models/* /home/homework/models/
COPY extra_config/* /home/homework/extra_config/
COPY run.sh /run.sh

RUN chmod +x /run.sh

# ENTRYPOINT ["/run.sh"]
CMD ["sh", "/run.sh"]
