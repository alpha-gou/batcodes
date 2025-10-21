#!/usr/bin/env python
# coding=utf-8
import json
import re


S2S_INPUT_PROMOT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{}
<|im_end|>
<|im_start|>assistant
"""

AE_EXPAND_PROMOT_V5 = """你是一个小学数学老师，现在给你一个题目解析，和一个参考答案，请判断题目解析中答案是否正确。
注意:
1. 参考答案有时候可能会有错误，也可能是无效的，因此在判断时不能完全依赖参考答案。
2. 题目解析的开头是题干，题干的内容可能存在问题或者歧义，这种情况不管解析的答案是什么都认为是错误的。
请逐步推理，如果解析答案是正确的，结果返回@正确@，否则返回@错误@。返回的结果只能有一个，当有多个小题时，必须每个小题全部正确，才能返回@正确@，否则返回@错误@
【题目解析】
%s
【参考答案】
%s
"""

INFO_MISS_PROMOT = """请判断下面题干的内容是否完整，根据以下规则：
a) 题干信息完整，语义通顺，根据题干的信息能够进行求解，是一道有效的问题。
b) 若题干出现“如图”“如下表”等涉及图示/表格的描述，可直接判断为题干不完整。
c) 可参考解析中的内容，如果解析中出现了题干中不存在的信息(常识类信息、分情况讨论除外)，则说明题干不完整。
如果题干是完整的，返回@1@，否则返回@0@。
【题干】
%s
【解析】
%s
"""

CHINESE_JX_PROMPT = ""

PROMPT_RM3 = ""

PROMPT_RM4 = """你是一个小学数学老师，现在要判断一个给小学生的讲题视频脚本是“合格”/“不合格”。
讲解过程中引入的一些数据可能是题目或图片有的，读题时没读全所有已知条件也是正常的，其他没有错误，就是合格的。
由于解题讲解过程中，没有专门强调图片信息，默认老师学生能看到，所以图中信息可能在讲解过程中以已知条件的形式提及，所以不要因为“缺图信息”的原因将其判定为不合格，请主要关注解答过程和结果的合格与否。
引入了题干未提供的数值，也很可能是合格的，因为可能有些信息只是读题时没有强调，或者读题时没读完整，但这不影响最终算作合格的讲解。
当你没有完全把握将讲题视频脚本判定为不合格时，请认为讲解合格。

合格问题标准：
    题干问题    无严重题干问题，可能存在优化建议（如“画图更佳”）。
    解析问题    解析稍复杂、跳步或不清晰，但整体逻辑正确。
    答案问题    无答案错误，可能存在优化建议（如“解析不够详细”）。
    教学方法问题  缺少前置分析、总结或互动，但未影响整体理解。
    技术问题    音画不同步或TTS问题，但未影响内容正确性。
    其他问题    无严重问题，可能存在优化建议（如“讲解稍啰嗦”）。

不合格问题标准：
    解析问题    解析错误、超纲、无分析，严重影响理解。
    答案问题    答案错误或答案解析错误，直接影响正确性。
    技术问题    音画不同步或TTS问题，严重影响理解和体验。
    应废弃问题   答案需画图/画线段/列表的形式回答。如果只是题目有图或线段并不一定废弃。

请你判断如下内容属于“合格”/“不合格”哪个分类。最终回答先给出分类结果，如果合格返回“@合格@”，否则返回“@不合格@”，然后给出简短理由概括分类原因：
"""

def get_xgv6_query_data(script):
    prompt = S2S_INPUT_PROMOT_TEMPLATE.format(script)
    query_data = {
        "text_input": prompt,
        "max_tokens": 5120,
        "end_id": 151645,
        "temperature": min(0.1 * data["retry_times"], 0.2),
        "stream": False
    }
    return query_data


def __get_ae_expand_promot(text, answer):
    try:
        res_str_list = []
        data = json.loads(text)
        for jt in data:
            if data[jt]["内容类型"] in {"读题", "解答"}:
                res_str_list.append(re.sub("@.*?@", " ", data[jt]["屏幕内容"]))
        fixed_str = "\n".join(res_str_list)
        return AE_EXPAND_PROMOT_V5 % (fixed_str, answer)
    except Exception as e:
        print(e)
        return ""


def get_ae_expand_request_json(text, answer):
    prompt = __get_ae_expand_promot(text, answer)
    json_data = {
        "model": "qwq32b",
        "messages": [
            {"role": "system", "content": "你是一个AI助手"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "min_p": 0.0,
        "max_tokens": 4096,
        "stream": True
    }
    return json_data


def __get_question_incomplete_prompt(script):
    data = json.loads(script)
    question = re.sub("@.*?@", "", data["镜头1"]["屏幕内容"])
    jt_list = []
    for jt in data:
        if jt != "镜头1":
            jt_list.append(data[jt]["屏幕内容"])
    return INFO_MISS_PROMOT % (question, "\n".join(jt_list))


def get_question_incomplete_json(script):
    prompt = __get_question_incomplete_prompt(script)
    json_data = {
        "model": "qwq32b",
        "messages": [
            {"role": "system", "content": "你是一个AI助手"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "min_p": 0.0,
        "max_tokens": 4096,
        "stream": True
    }
    return json_data


def get_caption_script_euqal_json(caption, script):
    prompt = CAPTION_SCRIPT_EQUAL_PROMPT % (caption, script)
    json_data = {
        "model": "qwen3",
        "messages": [
            {"role": "system", "content": "你是一个AI助手"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "min_p": 0.0,
        "max_tokens": 4096,
        "stream": True
    }
    return json_data


def get_chinese_jx_json(text):
    prompt = CHINESE_JX_PROMPT + text
    json_data = {
        "model": "whatever",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 8192,
        "temperature": 0.0,
    }
    return json_data


def get_rm3_json(script):
    prompt = PROMPT_RM3 + script
    json_data = {
        "model": "qwen3",
        "messages": [
            {"role": "system", "content": "你是一个AI助手"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "min_p": 0.0,
        "max_tokens": 4096,
        "stream": True
    }
    return json_data


def get_rm4_json(script):
    prompt = PROMPT_RM4 + script + " /think"
    json_data = {
        "model": "rm4",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "min_p": 0.0,
        "max_tokens": 4096,
        "stream": True
    }
    return json_data
