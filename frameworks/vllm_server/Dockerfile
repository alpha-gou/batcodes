FROM image-docker.zuoyebang.cc/public/nvidia/tensorrt-llm:0.17.0.post1

COPY ./requirements.txt .

RUN pip uninstall -y torch
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

RUN mkdir -p /home/homework/models
WORKDIR /home/homework/
COPY ./models/ models/

COPY api_server.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py
COPY run.sh /run.sh
RUN chmod +x /run.sh

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_VISIBLE_DEVICES all
ENV OPENBLAS_NUM_THREADS 1

# ENTRYPOINT ["/run.sh"]
CMD ["sh", "/run.sh"]
