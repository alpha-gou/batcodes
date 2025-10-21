#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json


def construct_middle_data(raw_data, queue_name, callback, vip=False):
    data = {
        "queue_name": queue_name,
        "stage": 0,
        "retry_times": 0,
        "raw": raw_data,
        "base": {"question": "", "answer": "", "subject": "", "depart": ""},
        "res": {"script": "", "err_code": 0, "err_info": ""},
        "history": [],
        "callback": callback,
        "vip": vip,   # 高优数据
    }
    return data


def construct_raw_data(tid, script, teachertid=None):
    raw_data = {
        "tid": tid,
        "teachertid": teachertid if teachertid else tid,
        "queryid": "test%s" % tid,
        "script": script,
        "subject": 2,
        "grade": 1,
        "version": 1,
    }
    return raw_data


def construct_llm_data(query_data, ext=None):
    """
    请求大模型的数据，所有参数均在外部添加，大模型服务只负责推理生产。
    大模型服务的入队列和其他任务共用，因此需要提供返回的队列名，大模型服务生产完毕压入到指定模型中。
    """
    llm_data = {
        "query": query_data,
        "return": {
            "think": "",
            "text": "",
            "succ": 0,
        },
        "ext": ext
    }
    return llm_data

