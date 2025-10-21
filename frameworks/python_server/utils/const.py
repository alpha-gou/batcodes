#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================
#        区分线上、测试的配置
# ===============================
# 配置为offline时，请求dd地址，使用测试队列；否则请求svc地址，使用线上队列
config_status = "offline"
# config_status = "online"
# config_status = "ship"

alpha-gou-svc.mock

if config_status == "online":
    # svc地址
    DATASTACK_URL = "http://alpha-gou-svc.mock:8080"
    QA_URL = "http://alpha-gou-svc.mock:8080/search_preprocess"
    SUBJECT_URL = "http://alpha-gou-svc.mock:8080/get_sub_dep_label"
    AE_URL = "http://alpha-gou-svc.mock:8080/get_text_equivalence"
    AE_URL_v2 = "http://alpha-gou-svc.mock:8080/get_text_equivalence_v2"
    QWQ_SCORING_URL = "http://aigc-aivideo-qwq32b-vllm-svc.mock:8080/v1/chat/completions"
    QWEN3_URL = "http://aigc-aivideo-qwen3-vllm-svc.mock:8080/v1/chat/completions"
    OCR_TRANSFORMATION_URL = "http://alpha-gou-svc.mock:8080/image_transformation.php"
    CHINESE_JX_URL = "http://aivideo-chinese-jx-svc.mock:8080/v1/chat/completions"
    RM4_URL = "http://aigc-aivideo-rewrite-rm4-svc.mock:8080/v1/chat/completions"
    # 线上队列
    REWRITE_MAIN_INPUT_QUEUE = "main_prim_math_rewrite"        # 输入队列
    REWRITE_MAIN_CONTROL_QUEUE = "rewrite_main_ctrl"           # 中控队列
elif config_status == "offline":
    # dd地址
    DATASTACK_URL = "http://alpha-gou.mock.dd"
    QA_URL = "http://alpha-gou.mock.dd/search_preprocess"
    SUBJECT_URL = "http://alpha-gou.mock.dd/get_sub_dep_label"
    AE_URL = "http://alpha-gou.mock.dd/get_text_equivalence"
    AE_URL_v2 = "http://alpha-gou.mock.dd/get_text_equivalence_v2"
    QWQ_SCORING_URL = "http://aigc-aivideo-qwq32b-vllm.mock.dd/v1/chat/completions"
    QWEN3_URL = "http://aigc-aivideo-qwen3-vllm.mock.dd/v1/chat/completions"
    OCR_TRANSFORMATION_URL = "http://alpha-gou.mock.dd/image_transformation.php"
    CHINESE_JX_URL = "http://aivideo-chinese-jx.mock.dd/v1/chat/completions"
    RM4_URL = "http://aigc-aivideo-rewrite-rm4.mock.dd/v1/chat/completions"
    # 测试队列
    REWRITE_MAIN_INPUT_QUEUE = "main_prim_math_rewrite_test"        # 输入队列（测试）
    REWRITE_MAIN_CONTROL_QUEUE = "rewrite_main_ctrl_test"           # 中控队列（测试）
elif config_status == "ship":
    # ship环境  新框架不会用到
    DATASTACK_URL = "https://panshi.zuoyebang.cc"

# ===============================
#       不区分线上、测试的配置
# ===============================
# generate main 配置
SLEEP_SECONDS = 300
SLEEP_SECONDS_WHILE_NO_DATA = 60
THREAD_NUM = 5
MAX_RETRY_TIMES = 15  # 最大重产次数

# 下方URL新框架不会使用到
CALLBACK_URL = "http://alpha-gou-svc.mock:8080/gvideo-hub/callback/scriptrewrite"
SHIP_CALLBACK_URL = "https://alpha-gou-svc.mock/gvideo-hub/callback/scriptrewrite"
NEW_CALLBACK_URL = "http://alpha-gou-svc.mock/aivideo/task/callback"

# DATASTACK 配置
DATASTACK_GROUP = "aigc_aivideo_queue"              # 队列组名
XG_QUEUE_NAME = "math_xgv6_rewrite"                 # 改写队列
RM_QUEUE_NAME = "math_qwen_scoring"                 # 评分（RM1、RM2）队列
DEBUG_RES_QUEUE_NAME = "math_debug_result"          # 默认用于接受debug数据的结果队列
SHIP_RES_QUEUE_NAME = "math_ship_result"            # 用于接受ship数据的结果队列

# 可用版本，0为debug测试，1为非图题，2为图题
VALID_VERSION = {0, 1, 2}

# tid info配置
TID_INFO_URL = "https://alpha-gou-svc.mock/api/getTidInfo?tid={}&type={}&time={}&cooperatorID={}&sign={}"
COOPERATOR_ID = 113
TOKEN = 'alphaGouMockToken1234567'


# ===============================
#         语文解析任务配置
# ===============================
CHINESE_JX_QUEUE_NAME = "chinese_jx_main"       # 语文解析主队列
CHINESE_DEBUG_QUEUE_NAME = "chinese_debug_result"  # 用于接受语文debug数据的结果队列，test_main_ship.py中使用，主要给胡奇老师ship环境调试用

