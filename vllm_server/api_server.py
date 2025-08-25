# coding=utf-8
# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/serve/openai_api_server.py

import argparse
import asyncio
from http import HTTPStatus
import json
import time
import datetime
import logging
from loguru import logger
from typing import AsyncGenerator, Dict, List, Optional

import fastapi
from fastapi import BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
#from fastchat.conversation import Conversation, SeparatorStyle
#from fastchat.model.model_adapter import get_conversation_template
from fastapi import HTTPException
import IPy

import base64

import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    CompletionRequest, CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    LogProbs, ModelCard, ModelList, ModelPermission, UsageInfo)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds
dt = datetime.datetime.now().strftime('%Y-%m-%d')


served_model = None
app = fastapi.FastAPI()


def is_valid_ip(ip: str):
    allowed_ips = ["47.252.40.203",
                   "47.252.68.95",
                   "47.90.204.44",
                   "47.89.185.158",
                   "124.223.93.138",
                   "124.221.195.136",
                   "124.220.205.240",
                   "124.222.77.105",
                   "118.195.223.150",
                   "119.45.118.109",
                   "119.45.124.71",
                   "118.195.153.116",
                   "49.233.8.108",
                   "49.233.8.103",
                   "49.233.8.109",
                   "49.233.8.104",
                   "49.233.8.105",
                   "49.232.138.91",
                   "172.29.213.24",
                   "172.29.213.162",
                   "172.29.212.33"]
    if ip not in allowed_ips and \
       ip not in IPy.IP('10.126.203.0/24') and \
       ip not in IPy.IP('10.104.0.0/16') and \
       ip not in IPy.IP('10.32.0.0/16') and \
       ip not in IPy.IP('10.96.0.0/16') and \
       ip not in IPy.IP('100.127.0.0/16'):
        return False
    return True


country_prompt_dict = {'explanation':{'US':'【Explanation】: ','PH':'【Explanation】: ','ID':'【Penjelasan】: ','VI':'【Giải thích】: '},
                       'answer':{'US':'【Answer】: ','PH':'【Answer】: ','ID':'【Jawaban】: ','VI':'【Câu trả lời】: '}}
order_prompt_dict = {'answer first':{'US':'<answer first>: ','PH':'<answer first>: ','ID':'<jawab dulu>: ','VI':'<trả lời trước>: '},
                     'step by step':{'US':'<step by step>: ','PH':'<step by step>: ','ID':'<selangkah demi selangkah>: ','VI':'<bước bằng bước>: '}}

def get_prompt_by_country(country: str, experiment_id=0):
    prompt_version = args.prompt_version
    if experiment_id == 1:
        prompt = '<answer first>: '
        if prompt_version == 'v1':
            if country in country_prompt_dict['answer']:
                prompt = prompt + country_prompt_dict['answer'][country]
        else:
            if country in order_prompt_dict['answer first']:
                prompt = order_prompt_dict['answer first'][country]
    else:
        prompt = '<step by step>: '
        if prompt_version == 'v1':
            if country in country_prompt_dict['explanation']:
                prompt = prompt + country_prompt_dict['explanation'][country]
        else:
            if country in order_prompt_dict['step by step']:
                prompt = order_prompt_dict['step by step'][country]

    return prompt

def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message,
                                      type="invalid_request_error").dict(),
                        status_code=status_code.value)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    if request.model == served_model:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret


async def get_gen_prompt(request) -> str:
    conv = get_conversation_template(request.model)
    conv = Conversation(
        name=conv.name,
        system=conv.system,
        roles=conv.roles,
        messages=list(conv.messages),  # prevent in-place modification
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        stop_token_ids=conv.stop_token_ids,
    )

    if isinstance(request.messages, str):
        prompt = request.messages
    else:
        for message in request.messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    return prompt


async def check_length(request, prompt, model_config):
    if hasattr(model_config.hf_config, "max_sequence_length"):
        context_len = model_config.hf_config.max_sequence_length
    elif hasattr(model_config.hf_config, "seq_length"):
        context_len = model_config.hf_config.seq_length
    elif hasattr(model_config.hf_config, "max_position_embeddings"):
        context_len = model_config.hf_config.max_position_embeddings
    elif hasattr(model_config.hf_config, "seq_length"):
        context_len = model_config.hf_config.seq_length
    else:
        context_len = 2048

    input_ids = tokenizer(prompt).input_ids
    token_num = len(input_ids)

    if token_num + request.max_tokens > context_len:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {context_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return None

@app.get("/v1/ready")
async def ready():
    return "OK"

@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(id=served_model,
                  root=served_model,
                  permission=[ModelPermission()])
    ]
    return ModelList(data=model_cards)


def create_logprobs(token_ids: List[int],
                    id_logprobs: List[Dict[int, float]],
                    initial_text_offset: int = 0) -> LogProbs:
    """Create OpenAI-style logprobs."""
    logprobs = LogProbs()
    last_token_len = 0
    for token_id, id_logprob in zip(token_ids, id_logprobs):
        token = tokenizer.convert_ids_to_tokens(token_id)
        logprobs.tokens.append(token)
        logprobs.token_logprobs.append(id_logprob[token_id])
        if len(logprobs.text_offset) == 0:
            logprobs.text_offset.append(initial_text_offset)
        else:
            logprobs.text_offset.append(logprobs.text_offset[-1] +
                                        last_token_len)
        last_token_len = len(token)

        logprobs.top_logprobs.append({
            tokenizer.convert_ids_to_tokens(i): p
            for i, p in id_logprob.items()
        })
    return logprobs


@app.post("/v1/chat/completions")
async def create_chat_completion(raw_request: Request):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """
    request = ChatCompletionRequest(**await raw_request.json())
    logger.info(f"Received chat completion request: {request}")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.logit_bias is not None:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    prompt = await get_gen_prompt(request)
    error_check_ret = await check_length(request, prompt, engine_model_config)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.time())
    try:
        sampling_params = SamplingParams(
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            max_tokens=request.max_tokens,
            best_of=request.best_of,
            top_k=request.top_k,
            ignore_eos=request.ignore_eos,
            use_beam_search=request.use_beam_search,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    result_generator = engine.generate(prompt, sampling_params, request_id)

    async def abort_request() -> None:
        await engine.abort(request_id)

    def create_stream_response_json(
        index: int,
        text: str,
        finish_reason: Optional[str] = None,
    ) -> str:
        choice_data = ChatCompletionResponseStreamChoice(
            index=index,
            delta=DeltaMessage(content=text),
            finish_reason=finish_reason,
        )
        response = ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        response_json = response.json(ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # First chunk with role
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(id=request_id,
                                                 choices=[choice_data],
                                                 model=model_name)
            data = chunk.json(exclude_unset=True, ensure_ascii=False)
            yield f"data: {data}\n\n"

        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)
                response_json = create_stream_response_json(
                    index=i,
                    text=delta_text,
                )
                yield f"data: {response_json}\n\n"
                if output.finish_reason is not None:
                    response_json = create_stream_response_json(
                        index=i,
                        text="",
                        finish_reason=output.finish_reason,
                    )
                    yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"

    # Streaming response
    if request.stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream",
                                 background=background_tasks)

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await abort_request()
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        choice_data = ChatCompletionResponseChoice(
            index=output.index,
            message=ChatMessage(role="assistant", content=output.text),
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(
        len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response


@app.post("/v1/completions")
async def create_completion(raw_request: Request):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.

    NOTE: Currently we do not support the following features:
        - echo (since the vLLM engine does not currently support
          getting the logprobs of prompt tokens)
        - suffix (the language models we currently support do not support
          suffix)
        - logit_bias (to be supported by vLLM engine)
    """

    start_time_all = time.time()
    if not is_valid_ip(raw_request.client.host):
        raise HTTPException(status_code=403, detail="Forbidden")
    headers = raw_request.headers
    raw_request_data = await raw_request.json()
    logger.info(f"Received completion raw request: {raw_request_data}, headers: {headers}")
    request = CompletionRequest(**raw_request_data)
    logger.info(f"Received completion request: {request}")


    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.echo:
        # We do not support echo since the vLLM engine does not
        # currently support getting the logprobs of prompt tokens.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "echo is not currently supported")

    if request.suffix is not None:
        # The language models we currently support do not support suffix.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "suffix is not currently supported")

    if request.logit_bias is not None:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    x_bd_logid = headers.get('X_bd_logid', '')
    trace_id = headers.get('Uber-Trace-Id', '')

    if isinstance(request.prompt, list):
        if len(request.prompt) == 0:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "please provide at least one prompt")
        if len(request.prompt) > 1:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                "multiple prompts in a batch is not currently supported")
        prompt = request.prompt[0]
    else:
        prompt = request.prompt

    if 'raw_prompt' not in raw_request_data or raw_request_data['raw_prompt'] == "0":
        country = raw_request_data['country'] if 'country' in raw_request_data else ''
        if args.experiment_id >= 0:
            experiment_id = args.experiment_id
        else:
            experiment_id = 1 if 'answer_first' in raw_request_data and raw_request_data['answer_first']=="1" else 0
        # 增加自定义控制标签，引导模型输出
        prompt = prompt + get_prompt_by_country(country=country, experiment_id=experiment_id)
    logger.info(f"request_id:[{request_id}] x_bd_logid:[{x_bd_logid}] trace_id:[{trace_id}] prompt:[{prompt}]")
    created_time = int(time.time())
    try:
        sampling_params = SamplingParams(
            n=request.n,
            best_of=request.best_of,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=args.temperature if args.temperature > 0 else request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            # stop=[request.stop_words],
            stop=['<|startoftext|>', '<|endoftext|>', '<unk>', '<|pbby_i|>'],
            # stop_token_ids=[0, 1, 2],
            ignore_eos=request.ignore_eos,
            max_tokens=request.max_tokens,
            logprobs=request.logprobs,
            use_beam_search=request.use_beam_search,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    result_generator = engine.generate(prompt, sampling_params, request_id)

    # Similar to the OpenAI API, when n != best_of, we do not stream the
    # results. In addition, we do not stream the results when use beam search.
    stream = (request.stream
              and (request.best_of is None or request.n == request.best_of)
              and not request.use_beam_search)

    async def abort_request() -> None:
        await engine.abort(request_id)

    def create_stream_response_json(
        index: int,
        text: str,
        logprobs: Optional[LogProbs] = None,
        finish_reason: Optional[str] = None,
    ) -> str:
        choice_data = CompletionResponseStreamChoice(
            index=index,
            text=text,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )
        response = CompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        response_json = response.json(ensure_ascii=False)

        return response_json

    def str_xor(text, key):
        return ''.join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(text))

    def encrypt(text, key):
        to_encrypt = base64.b64encode(text.encode()).decode()
        return str_xor(to_encrypt, key)

    def decrypt(text, key):
        to_decrypt = str_xor(text, key)
        return base64.b64decode(to_decrypt.encode()).decode()
    
    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        total_num_tokens = 0
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]):].replace('：',' ')
                if request.logprobs is not None:
                    logprobs = create_logprobs(
                        output.token_ids[previous_num_tokens[i]:],
                        output.logprobs[previous_num_tokens[i]:],
                        len(previous_texts[i]))
                else:
                    logprobs = None
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)
                total_num_tokens = previous_num_tokens[i]
                #response_json = create_stream_response_json(
                #    index=i,
                #    text=delta_text,
                #    logprobs=logprobs,
                #)
                if output.finish_reason is not None:
                    logprobs = (LogProbs()
                                if request.logprobs is not None else None)
                    #response_json = create_stream_response_json(
                    #    index=i,
                    #    text="",
                    #    logprobs=logprobs,
                    #    finish_reason=output.finish_reason,
                    #)
                    stop_time_all = time.time()
                    cost = stop_time_all - start_time_all
                    logger.info(f"request_id:[{request_id}] x_bd_logid:[{x_bd_logid}] trace_id:[{trace_id}] ttk:[{total_num_tokens}] cost:[{cost}]")
                    yield encrypt(f"end: 1\nttk: {total_num_tokens}\ndata: {delta_text}\n\n", "fckhtps")
                    # yield f"tokens:{total_num_tokens}"
                else:
                    yield encrypt(f"end: 0\nttk: {total_num_tokens}\ndata: {delta_text}\n\n", "fckhtps")

    # Streaming response
    if stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream",
                                 background=background_tasks)

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await abort_request()
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        if request.logprobs is not None:
            logprobs = create_logprobs(output.token_ids, output.logprobs)
        else:
            logprobs = None
        choice_data = CompletionResponseChoice(
            index=output.index,
            text=output.text,
            logprobs=logprobs,
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)
    logger.info(f"request_id:[{request_id}] x_bd_logid:[{x_bd_logid}] trace_id:[{trace_id}] choices:[{choices}]")
    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(
        len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host",
                        type=str,
                        default="localhost",
                        help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                        "specified, the model name will be the same as "
                        "the huggingface name.")
    parser.add_argument("--limit_concurrency",
                        type=int,
                        default=60,
                        help="")

    parser.add_argument("--prompt_version",
                        type=str,
                        default='v2',
                        help="")

    parser.add_argument("--experiment_id",
                        type=int,
                        default=-1,
                        help="")

    parser.add_argument("--temperature",
                        type=float,
                        default=0,
                        help="")


    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    logger.add("logs/api_server_{}.log".format(args.port), format="{time}|{level}|{message}", level="INFO", rotation="500 MB")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(engine_args.tokenizer,
                              tokenizer_mode=engine_args.tokenizer_mode,
                              trust_remote_code=engine_args.trust_remote_code)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                limit_concurrency=args.limit_concurrency,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
