FROM image-docker.zuoyebang.cc/base/python:3.7-slim-builder as builder

WORKDIR /home/homework/

COPY ./requirements.txt .

COPY 3rdparty 3rdparty

RUN pip install --user -i https://mirrors.bfsu.edu.cn/pypi/web/simple/ -r requirements.txt

#    && cd /home/homework/3rdparty/triton \
#    && pip install --user tritonclientutils-2.2.0.dev0-py3-none-any.whl -i https://mirrors.bfsu.edu.cn/pypi/web/simple/ \
#    && pip install --user tritonhttpclient-2.2.0.dev0-py3-none-any.whl -i https://mirrors.bfsu.edu.cn/pypi/web/simple/

FROM image-docker.zuoyebang.cc/base/python:3.7-slim

ENV PATH=/root/.local/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/homework/winrain/lib
WORKDIR /home/homework/

COPY --from=builder /root/.local /root/.local
COPY . .

CMD ["sh", "start.sh"]
