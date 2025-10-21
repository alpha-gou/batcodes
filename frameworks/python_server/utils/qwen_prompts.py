#!/usr/bin/env python
# coding=utf-8
import json
import re


def get_qwen_scoring_prompt_round1(text, pic=False):
    try:
        res_str_list = []
        data = json.loads(text)
        for jt in data:
            res_str_list.append(re.sub("@.*?@", " ", data[jt]["屏幕内容"].replace("$", "")))
        fixed_str = "\n".join(res_str_list)
        prompt_template = ROUND1_MM_PROMPTS if pic else ROUND1_PROMPTS
        return prompt_template % (fixed_str)
    except:
        return ""


def get_qwen_scoring_prompt_round2(text, pic=False):
    try:
        data = json.loads(text)
        str1 = ""
        str2 = ""
        for jt in data:
            if data[jt]["内容类型"] == "读题":
                str1 += re.sub("@.*?@", " ", data[jt]["屏幕内容"])
            elif data[jt]["内容类型"] == "解答":
                str2 += re.sub("@.*?@", " ", data[jt]["屏幕内容"])
        fixed_str = MARKDOWN_TEMPLATE % (str1, str2)
        prompt_template = ROUND2_MM_PROMPTS if pic else ROUND2_PROMPTS
        prompt = prompt_template % (fixed_str)
        return prompt
    except:
        return ""

ROUND1_PROMPTS = ""
ROUND2_PROMPTS = ""
MARKDOWN_TEMPLATE = """# 题干：
%s
# 解析：
%s"""
ROUND1_MM_PROMPTS = ""
ROUND2_MM_PROMPTS = ""
