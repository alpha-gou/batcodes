import json
import time
import requests
import logging
import re
import base64
import hashlib
import random
from urllib.parse import urlencode

from .const import *
from .prompts import *
from .format_check import get_error_code_and_info
from .qwen_prompts import get_qwen_scoring_prompt_round1, get_qwen_scoring_prompt_round2
from .data_structure import construct_llm_data
from .dataqueue import write_dq_data


# 日志记录器
logger = logging.getLogger(__name__)


def request_answer_question(data, use_teachertid=False):
    headers = {"Content-Type": "application/json"}
    tid = data["raw"]["teachertid"] if use_teachertid else data["raw"]["tid"]
    for _ in range(3):  # 重试3次
        try:
            json_data = {
                "tid": int(tid),
                "needTransOcr": 1, "needPicTab": 1,
                "needTidInfo": 1, "needProcessText": 1,
                "needTextFormat": 0, "needOcrFormat": 0
            }
            response = requests.post(QA_URL, headers=headers, json=json_data)
            response_json = response.json()["data"]
            data["base"]["answer"] = response_json["final_answer"]
            data["base"]["question"] = response_json["final_question"]
        except Exception as e:
            logger.error("queryid {} tid {} request_question_answer error with: {}".format(
                    data["raw"]["queryid"], tid, str(e)))
            time.sleep(10)
        else:
            break
    if not data["base"]["answer"]:
        data["base"]["answer"] = data["base"]["aiana_answer"]
    if not data["base"]["answer"]:
        data["base"]["answer"] = data["base"]["small_ocr_answer"]


def request_tid_info(data, use_teachertid=False):
    tid = data["raw"]["teachertid"] if use_teachertid else data["raw"]["tid"]
    for i in ["aiana_answer", "img_url", "small_ocr_answer"]:
        data["base"][i] = ""
    log_prefix = "queryid {} tid {}".format(data["raw"]["queryid"], tid)
    for _ in range(3):  # 重试3次
        try:
            nowtime = int(time.time())
            get_url = TID_INFO_URL.format(
                    tid, 2, nowtime, COOPERATOR_ID, __generate_sign(tid, TOKEN, nowtime))
            tid_info = requests.get(url=get_url).json()["data"]
        except Exception as e:
            logger.error("{} request tid_info - with error: {}".format(log_prefix, str(e)))
            time.sleep(10)
        else:
            if "answer" in tid_info:
                regex = r'<img[^>]*src=["\']?(https?://[^"\'>]+)["\']?'  # 匹配src中的URL
                matches = re.findall(regex, tid_info["answer"])
                if len(matches) > 0:
                    data["base"]["img_url"] = matches[0]
            if "analysis" in tid_info:
                try:
                    analysis_data = json.loads(tid_info["analysis"])
                    data["base"]["aiana_answer"] = analysis_data["aiAnalysis"]["answer"]
                except Exception as e:
                    logger.info("{} request aiAnalysis failed with: {}".format(
                            log_prefix, str(e)))
            break
    if not data["base"]["aiana_answer"]:
        data["base"]["small_ocr_answer"] = __rq_small_ocr(data["base"]["img_url"], log_prefix)


def __generate_sign(tid, token, timestamp):
    s1 = 'params1={}&params2={}&token={}'.format(tid, timestamp, token)
    s2 = base64.b64encode(s1.encode('utf8')).decode('utf8')
    s3 = hashlib.md5(token.encode('utf8')).hexdigest()
    s4 = s3 + s2
    return hashlib.md5(s4.encode('utf8')).hexdigest()


def __rq_small_ocr(img_url, log_prefix):
    if not img_url: return ""
    try:
        image = requests.get(img_url, timeout=3)
        img_data = image.content
        img_b64_string = base64.b64encode(img_data)
        post_data = {
            "imageData": img_b64_string,
            "sid": random.randint(112123, 142353464654575),
            "source": "cnprint_reco",
            "picFormat": "jpg"
        }
        headers = {
            'Content-type': "application/x-www-form-urlencoded",
            'User-Agent': "PostmanRuntime/7.17.1",
            'Accept': "*/*",
            'Cache-Control': "no-cache",
            'Accept-Encoding': "gzip, deflate",
            'Content-Length': "102248",
            'Connection': "keep-alive",
            'cache-control': "no-cache"
        }
        data = urlencode(post_data)

        response = requests.post(OCR_TRANSFORMATION_URL, data=data, headers=headers, timeout=5)
        # 控制下置信度 prob  < 0.92  可以不用
        text = json.loads(response.text)['ret']
        text_list = text.split("\t")
        if float(text_list[1]) < 0.92:
            return ""
        else:
            return text_list[0]
    except Exception as e:
        logger.error("{} request small ocr failed: {}".format(log_prefix, e))
        return ""


def req_sub_dep(data):
    for _ in range(3):  # 重试3次
        try:
            tid = data["raw"]["tid"]
            text = data["base"]["question"]
            json_data = {
                'input_list': [
                    {'input_key_id': tid, 'query_text': text}
                ]
            }
            res = requests.post(SUBJECT_URL, data=json.dumps(json_data))
            res_data = json.loads(res.text)
            if res_data['errNo'] == 0:
                sub_dep = res_data['answer_list'][0]['q_label_name']
                data["base"]["subject"] = sub_dep[-2:]
                data["base"]["depart"] = sub_dep[:-2]
            else:
                raise ValueError(res.text)
        except Exception as e:
            logger.error("queryid {} tid {} req_sub_dep error with: {}".format(
                        data["raw"]["queryid"], data["raw"]["tid"], str(e)))
            time.sleep(10)
        else:
            break


def remove_pinyin(script):
    data = json.loads(script)
    for jt in data:
        data[jt]["屏幕内容"] = re.sub("#[a-zA-Z]+[1-5]#", "", data[jt]["屏幕内容"])  # 屏幕内容去掉所有拼音
        data[jt]["旁白内容"] = re.sub("#[a-zA-Z]+5#", "", data[jt]["旁白内容"])  # 旁白内容去掉轻音5
    return json.dumps(data, ensure_ascii=False)


def format_check(data):
    script = data["res"]["script"]
    try:
        err_code, err_info = get_error_code_and_info(script, data["base"]["answer"])
        if err_code != 0:
            err_code += 20000   # 格式类问题以2开头
            data["res"]["err_code"], data["res"]["err_info"] = err_code, err_info
        else:
            data["res"]["script"] = remove_pinyin(script)
    except Exception as e:
        logger.error("format_check error with: {}".format(str(e)))


def remove_latex(text):
    try:
        text = text.replace("\\boxed", " ").replace("boxed", "")
        text = text.replace("$", " ")  # 去掉所有的$符号
        return text
    except:
        return ""


def request_answer_equal_main(data):
    headers = {"Content-Type": "application/json"}
    try:
        # 答案一致性判断
        query_id = data["raw"]["queryid"]
        question = data["base"]["question"]
        answer = data["base"]["answer"]
        text = data["res"]["script"]
        ae_res = request_answer_equal_v1(question, answer, text)

        # 新版答案一致性
        # 如果失败，修复后用v2版进行判断
        if ae_res != 1:
            logger.info("queryid {} failed at answer_equal round 1".format(query_id))
            tid = data["raw"]["tid"]
            answer_fix = remove_latex(answer)
            ae_res = request_answer_equal_v2(tid, question, answer_fix, text)

        # 一致性再召回：增加对ai解析答案的判断
        if ae_res != 1:
            if data["base"]["aiana_answer"]:
                answer = data["base"]["answer"]
            elif data["base"]["small_ocr_answer"]:
                answer = data["base"]["small_ocr_answer"]
            else:
                answer = ""
            if answer:
                ae_res = request_answer_equal_v1(question, answer, text)
                if ae_res != 1:
                    tid = data["raw"]["tid"]
                    ae_res = request_answer_equal_v2(tid, question, answer_fix, text)

        # 暂时下线qwq再召回
        # # 如果ai解析答案仍然不一致，使用qwq模型进行判断。图题不过qwq再召回
        # if (ae_res != 1) and (data["version"] != 2):
        #     ae_res = request_answer_equal_qwq(text, data["base"]["answer"])

        # 原地写入结果
        if ae_res != 1:
            data["res"]["err_code"] = 40001
            data["res"]["err_info"] = "答案一致性校验未通过"
        elif ae_res == -1:
            data["res"]["err_code"] = 40002
            data["res"]["err_info"] = "答案一致性请求失败"
    except Exception as e:
        logger.error("request_answer_equal error with: {}".format(str(e)))


def request_answer_equal_v1(question, answer, text):
    headers = {"Content-Type": "application/json"}
    try:
        json_data = {
            "stem1": question, "analysis1": answer,
            "stem2": question, "analysis2": text,
            "courseid": 2, "departid": 0, "categoryid": 0
        }
        response = requests.post(AE_URL, headers=headers, json=json_data)
        ae_res = json.loads(response.text)
        return 1 if ae_res["is_equal"] else 0
    except Exception as e:
        logger.error("request_answer_equal_v1 error with: {}".format(str(e)))
        return -1


def request_answer_equal_v2(tid, question, answer, text):
    json_data = {
        "logid": str(tid),
        "stem1": question,
        "analysis1": answer,
        "analysis2": text,
        "product": "k12_aianalysis"
    }
    headers = {'Content-Type': "application/json"}
    try:
        ret = requests.post(url=AE_URL_v2, json=json_data, headers=headers)
        ret = ret.json()
        return ret["judge"]
    except Exception as e:
        logger.error("request_answer_equal_v2 error with: {}".format(str(e)))
        return -1


def request_answer_equal_qwq(text, answer):
    json_data = get_ae_expand_request_json(text, answer)
    thinking, result, final_answer = "", "", -1
    response = ""
    for attempt in range(3):
        try:
            thinking, result = __post_vllm_stream_thinking(QWQ_SCORING_URL, json_data)
            if result.strip().startswith("@正确@"):
                final_answer = 1
            elif result.strip().startswith("@错误@"):
                final_answer = 0
            else:
                continue
        except Exception as e:
            logger.error("request_answer_equal_qwq error with: {}".format(str(e)))
        else:
            break
    return final_answer


def request_question_incomplete(data):
    script = data["res"]["script"]
    json_data = get_question_incomplete_json(script)
    final_answer = -1
    for attempt in range(3):
        try:
            thinking, result = __post_vllm_stream_thinking(QWQ_SCORING_URL, json_data)
            if result.strip().startswith("@1@"):
                final_answer = 1
            elif result.strip().startswith("@0@"):
                final_answer = 0
            else:
                continue
        except Exception as e:
            logger.info("request_question_incomplete error with: {}".format(str(e)))
        else:
            break
    # 原地写入结果
    if final_answer != 1:
        data["res"]["err_code"] = 40003
        data["res"]["err_info"] = "题干残缺"
    elif final_answer == -1:
        # 避免vllm异常退出，如果判残逻辑失败，不会对其他流程造成影响
        # data["res"]["err_code"] = 40004
        # data["res"]["err_info"] = "判残请求失败"
        logger.error("request_question_incomplete failed, but does not affect other processes")


def __post_vllm_stream_thinking(api_url, json_data):
    text, thinking, result = "", "", ""
    headers = {"Content-Type": "application/json"}
    try:
        with requests.post(api_url, headers=headers, json=json_data, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                try:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith('data: '):
                        chunk = json.loads(decoded_line[6:])  # 去除"data:"前缀
                        text += chunk['choices'][0]['delta']['content']
                except:
                    pass
        contents = text.strip("<think>").strip("\n").split("</think>")
        thinking = contents[0].strip("\n")
        result = contents[-1].strip("\n")
    except Exception as e:
        logger.error("__post_vllm_stream_thinking error with: {}".format(str(e)))
    return thinking, result


def script_rewrite(data, log_prefix):
    """
    请求改写模型，状态更新为1
    """
    data["stage"] = 1
    prompt = S2S_INPUT_PROMOT_TEMPLATE.format(data["raw"]["script"])
    query_data = {
        "text_input": prompt,
        "max_tokens": 5120,
        "end_id": 151645,
        "temperature": min(0.1 * data["retry_times"], 0.2),
        "stream": False
    }
    data["llm"] = construct_llm_data(query_data, "xg")
    __push_result_to_data_queue(data, XG_QUEUE_NAME, log_prefix)


def request_qwen_scoring_round1(data, log_prefix, pic):
    prompt = get_qwen_scoring_prompt_round1(data["res"]["script"], pic)
    query_data = {
        "text_input": prompt,
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.0,
    }
    data["llm"] = construct_llm_data(query_data, "check_gs1")
    __push_result_to_data_queue(data, RM_QUEUE_NAME, log_prefix)


def request_qwen_scoring_round2(data, log_prefix, pic):
    prompt = get_qwen_scoring_prompt_round2(data["res"]["script"], pic)
    query_data = {
        "text_input": prompt,
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.0,
    }
    data["llm"] = construct_llm_data(query_data, "check_gs2")
    __push_result_to_data_queue(data, RM_QUEUE_NAME, log_prefix)


def __push_result_to_data_queue(data, queue_name, log_prefix):
    data_str = json.dumps(data, ensure_ascii=False)
    write_to_head = data.get("vip", False)  # 非vip数据写入到队尾，否则写入队首
    for _ in range(10):
        try:
            response = write_dq_data(
                group_name = DATASTACK_GROUP,
                queue_name = queue_name,
                data = [data_str],
                write_to_head = write_to_head,
            )
            if response["errNo"] != 0:
                raise ValueError("write_dq_data to {} returns False !!".format(queue_name))
        except Exception as e:
            logger.info("{} : push dataqueue [{}] error: {}".format(
                    log_prefix, queue_name, str(e)))
            time.sleep(60)  # 当队列出现问题时，每分钟重试一次，10分钟重试10次
        else:
            return
    # 重试10次仍然出错，则丢弃数据，上报错误日志
    logger.error("{} : push dataqueue [{}] error !!!".format(log_prefix, queue_name))


def __result_router(data, log_prefix):
    """
    结果路由，判断是否需要重产，压入改写队列或者中控队列callback
    """
    try:
        retry_times = data["retry_times"]

        # 生产失败
        if data["res"]["err_code"] != 0:
            # 生成过程日志
            logger.info("{} - RECORD data - {}".format(
                    log_prefix, json.dumps(data["res"], ensure_ascii=False)))
            data["retry_times"] += 1  # 重试次数增加
            # 未超出重试次数，则退回重产
            if retry_times < MAX_RETRY_TIMES:
                data["history"].append(data["res"])
                logger.info("{} TRANS stage {} >> 1 - into xg queue.".format(log_prefix, data["stage"]))
                script_rewrite(data, log_prefix)
            # 先不上
            # # 最后一次失败，使用原脚本
            # elif retry_times == MAX_RETRY_TIMES:
            #     data["history"].append(data["res"])
            #     logger.info("{} ORIGIN_STRATEGY stage {} >> 2 - into gs queue.".format(log_prefix, data["stage"]))
            #     data["llm"]["return"]["text"] = data["raw"]["script"]
            else:
                process_stage_99(data, log_prefix)  # 失败返回
        else:
            process_stage_99(data, log_prefix)  # 生产成功
    except Exception as e:
        logger.error("{} result_router failed with error {}".format(log_prefix, str(e)))


def callback(data, log_prefix):
    try:
        callback_url = data["callback"]
        script_data = data["res"]
        callback_data = {
            "queryid": data["raw"]["queryid"],
            "tid": data["raw"]["tid"],
            "teachertid": data["raw"]["teachertid"],
            "script": script_data["script"],
            "status_code": script_data["err_code"],
            "status_info": script_data["err_info"],
            "source": data["raw"]["version"],
        }
        logger.info("{} RET_CALLBACK - data: {}".format(
                log_prefix, json.dumps(script_data, ensure_ascii=False)))
        resp = requests.post(callback_url,  data=json.dumps(callback_data), timeout=300)
        logger.info("{} CALLBACK resp: {}".format(log_prefix, resp.text))
    except Exception as e:
        logger.info("{} CALLBACK error: {}".format(log_prefix, str(e)))
        time.sleep(10)
    return


def callback_new(data, log_prefix):
    try:
        callback_url = data["callback"]
        script_data = data["res"]
        rewriteScript = {}
        if script_data["err_code"] == 2:
            rewriteScript = json.loads(script_data["script"])
        callback_data = {
            "taskId": data["raw"]["taskId"],
            "tid": data["raw"]["tid"],
            "expId": data["raw"].get("expId", ""),
            "status": script_data["err_code"],  # 成功为2 
            "msg": script_data["err_info"],
            "solveModel": "改写",
            "ext": {
                "rewriteScript": rewriteScript,
                "pureCalcScript": {},
                "univCalcScript": {}
            },
            "param": data["raw"]["param"]
        }
        logger.info("{} RET_CALLBACK_NEW - data: {}".format(
                log_prefix, json.dumps(callback_data, ensure_ascii=False)))
        headers = {'Content-Type': "application/json"}
        resp = requests.request("POST", callback_url, data=json.dumps(callback_data), headers=headers).json()
        logger.info("{} CALLBACK_NEW resp: {}".format(log_prefix, json.dumps(resp, ensure_ascii=False)))
    except Exception as e:
        logger.info("{} CALLBACK_NEW error: {}".format(log_prefix, str(e)))
        time.sleep(10)
    return


def complete_base_info(data, log_prefix):
    # 请求tid info
    request_tid_info(data)
    request_answer_question(data)
    if not (data["base"]["answer"] and data["base"]["question"]):
        # 更换teachertid重试
        logger.info("{} request answer switch to teachertid, data_base: {}".format(
                log_prefix, json.dumps(data["base"], ensure_ascii=False)))
        if data["raw"]["tid"] != data["raw"]["teachertid"]:
            request_tid_info(data, True)
            request_answer_question(data, True)
    # 请求题干答案异常时直接返回
    if not (data["base"]["answer"] and data["base"]["question"]):
        logger.error("{} answer_question_error, data_base: {}".format(
                log_prefix, json.dumps(data["base"], ensure_ascii=False)))
        data["res"] = {"err_code": 10004, "err_info": "未查询到题干答案", "script": ""}
        data["retry_times"] = MAX_RETRY_TIMES + 1
        __result_router(data, log_prefix)
        return 0

    # 请求学科判断接口
    req_sub_dep(data)
    # 学科判断异常时直接返回
    if not data["base"]["subject"] or data["base"]["subject"] != "数学":
        logger.error("{} subject_error, data: {}".format(
                log_prefix, json.dumps(data["base"], ensure_ascii=False)))
        data["res"] = {"err_code": 10005, "err_info": "学科异常", "script": ""}
        data["retry_times"] = MAX_RETRY_TIMES + 1
        __result_router(data, log_prefix)
        return 0
    return 1


def remove_repeated_text(text):
    """
    如果 pattern 在字符串 text 中出现超过两次，则删除所有 pattern。
    """
    # 仅对带导图的题生效
    if "导图" not in text:
        return text, False
    pattern = r"下面(?:[\s,，]*)来看具体过程。"
    # 计算 pattern 在 text 中出现的次数
    count = len(re.findall(pattern, text))

    # 如果出现次数大于2，则删除所有pattern
    if count > 2:
        return re.sub(pattern, '', text), True
    else:
        return text, False


def request_rm3(data, log_prefix):
    json_data = get_rm3_json(data["res"]["script"])
    response = ""
    for attempt in range(3):
        try:
            _, result = __post_vllm_stream_thinking(QWEN3_URL, json_data)
            final_result = __extract_rm3_result(result)
            if final_result == "不合格":
                # 未通过RM3，标记状态
                data["res"]["err_code"] = 50003
                data["res"]["err_info"] = "未通过RM3筛选"
            elif final_result != "合格":
                raise ValueError("bad result")
        except Exception as e:
            logger.info("{} request_rm3 error with: {}".format(log_prefix, str(e)))
        else:
            break
    else:
        # RM3为高准确低召回模型，因此RM3请求失败时不算错误，继续后续流程。
        logger.error("After retry 3 times, {} request_rm3 still returns ERROR.".format(log_prefix))


def __extract_rm3_result(text):
    """"
    提取答案
    """
    # 定义可能出现的结果关键词
    keywords = ["合格", "不合格"]
    # 初始化最早出现的位置为无穷大
    earliest_index = float('inf')
    result = ""
    # 遍历关键词，查找最早出现的位置
    for keyword in keywords:
        index = text.find(keyword)
        if index != -1 and index < earliest_index:  # 如果找到且位置更早
            earliest_index = index
            result = keyword
    # 返回最早出现的关键词
    return result


def request_rm4(data, log_prefix):
    json_data = get_rm4_json(data["res"]["script"])
    response = ""
    for attempt in range(3):
        try:
            _, result = __post_vllm_stream_thinking(RM4_URL, json_data)
            if "@合格@" in result and "@不合格@" not in result:
                break
            elif "@合格@" not in result and "@不合格@" in result:
                # 未通过RM4，标记状态
                data["res"]["err_code"] = 50004
                data["res"]["err_info"] = "未通过RM4筛选"
            else:
                raise ValueError("bad result")
        except Exception as e:
            logger.info("{} request_rm4 error with: {}".format(log_prefix, str(e)))
        else:
            break
    else:
        # RM4为高准确低召回模型，因此RM3请求失败时不算错误，继续后续流程。
        logger.error("After retry 3 times, {} request_rm4 still returns ERROR.".format(log_prefix))


# ===============================
#         各阶段主逻辑
# ===============================
def process_stage_0(data, stage_0_log_prefix):
    """
    阶段0
    输入：原始中间数据
    处理：添加题干答案等基础信息
    输出：数据存入改写队列
    """
    # 补全base_info
    succ = complete_base_info(data, stage_0_log_prefix)
    # 更新状态，配置改写模型参数，压入改写队列
    if succ == 1:
        logger.info("{} TRANS stage 0 >> 1 - into xg queue.".format(stage_0_log_prefix))
        script_rewrite(data, stage_0_log_prefix)


def process_stage_1(data, stage_1_log_prefix):
    """
    阶段1
    输入：改写后的中间数据
    处理：过规则策略，答案一致性策略，题干残缺策略
    输出：如果通过以上判断，数据存入判别队列；如果不通过，进入结果路由
    """
    # 取改写结果
    script = data["llm"]["return"]["text"]
    # 修复“下面，来看具体过程”多次出现
    script, repeat_flag = remove_repeated_text(script)
    if repeat_flag:
        logger.info("{} HIT remove_repeat: {}".format(stage_1_log_prefix, data["llm"]["return"]["text"]))

    # 创建结果数据
    data["res"] = {"err_code": 0, "err_info": "", "script": script}
    log_data = {k: v for k, v in data["res"].items() if k not in {'script'}}

    # 规则判断
    if data["res"]["err_code"] == 0:
        format_check(data)
        logger.info("{} format_check finished. data: {}:".format(
                stage_1_log_prefix, json.dumps(log_data, ensure_ascii=False)))
    # 答案一致性
    if data["res"]["err_code"] == 0:
        request_answer_equal_main(data)
        logger.info("{} request_answer_equal finished. data: {}:".format(
                stage_1_log_prefix, json.dumps(log_data, ensure_ascii=False)))
    # 题干残缺（图题不过判残）
    if (data["version"] != 2) and (data["res"]["err_code"] == 0):
        request_question_incomplete(data)
        logger.info("{} request_question_incomplete finished. data: {}:".format(
                stage_1_log_prefix, json.dumps(log_data, ensure_ascii=False)))
    # 更新状态
    data["stage"] = 2
    # 如果前两步都通过，压入评分队列做第一轮判断
    if data["res"]["err_code"] == 0:
        request_qwen_scoring_round1(data, stage_1_log_prefix, data["version"] == 2)  # 区分图题
        logger.info("{} TRANS stage 1 >> 2 - into gs queue.".format(stage_1_log_prefix))
    # 前两步没有通过，进入结果路由
    else:
        __result_router(data, stage_1_log_prefix)


def process_stage_2(data, stage_2_log_prefix):
    """
    阶段2
    输入：RM1判断后的中间数据
    处理：获取RM1判断结果
    输出：如果RM1通过，数据存入判别队列；如果不通过，进入结果路由
    """
    # 取第一轮评分模型结果
    try:
        text_output = data["llm"]["return"]["text"]
        res = int(text_output[1])
        if res > 0:
            # 第一轮评分模型通过，压入评分队列做第二轮判断
            data["stage"] = 3  # 更新状态
            request_qwen_scoring_round2(data, stage_2_log_prefix, data["version"] == 2)  # 区分图题
            logger.info("{} TRANS stage 2 >> 3 - into gs queue.".format(stage_2_log_prefix))
        else:
            # 第一轮未通过，进入结果路由
            data["res"]["err_code"] = 50001
            data["res"]["err_info"] = "未通过第一轮评分模型筛选"
            __result_router(data, stage_2_log_prefix)
    except Exception as e:
        logger.error("第一轮评分模型请求失败 {} ".format(str(e)))
        data["res"]["err_code"] = 50011
        data["res"]["err_info"] = "第一轮评分模型请求失败"
        __result_router(data, stage_2_log_prefix)


def process_stage_3(data, stage_3_log_prefix):
    """
    阶段3
    输入：RM2判断后的中间数据
    处理：获取RM2判断结果，RM3判断，RM4判断
    输出：进入结果路由
    """
    # 取第二轮评分模型结果
    try:
        text_output = data["llm"]["return"]["text"]
        res = int(text_output[1])
        if res <= 0:
            data["res"]["err_code"] = 50002
            data["res"]["err_info"] = "未通过第二轮评分模型筛选"
    except Exception as e:
        logger.error("第二轮评分模型请求失败 {} ".format(str(e)))
        data["res"]["err_code"] = 50012
        data["res"]["err_info"] = "第二轮评分模型请求失败"

    # RM3判别
    if data["res"]["err_code"] == 0:
        logger.info("{} request RM3 ...".format(stage_3_log_prefix))
        request_rm3(data, stage_3_log_prefix)

    # RM4判别
    if data["res"]["err_code"] == 0:
        logger.info("{} request RM4 ...".format(stage_3_log_prefix))
        request_rm4(data, stage_3_log_prefix)
    __result_router(data, stage_3_log_prefix)  # 无论何种情况均入结果路由


def process_stage_98(data, log_prefix):
    """
    仅测试，不改写
    """
    succ = complete_base_info(data, log_prefix)  # 补全base_info
    if succ == 1:
        logger.info("{} TRANS stage 98 >> 1 - without xg.".format(log_prefix))
        data["llm"] = construct_llm_data({}, "no_xg")
        data["llm"]["return"]["text"] = data["raw"]["script"]
        data["stage"] = 1
        data["retry_times"] = MAX_RETRY_TIMES + 1  # 仅挑选，不重产
        __push_result_to_data_queue(data, data["queue_name"], log_prefix)


def process_stage_99(data, log_prefix):
    """
    生产完毕，数据callback或者进入debug结果队列
    """
    # 留存final日志
    label = "[success]" if data["res"]["err_code"] == 0 else "[failed]"
    log_data = {k: v for k, v in data.items() if k not in {'history', 'llm'}}
    logger.info("{} FINAL {} - code {} info {} - data: {}".format(
            log_prefix, label, data["res"]["err_code"], data["res"]["err_info"],
            json.dumps(log_data, ensure_ascii=False)))
    if data["callback"].startswith("http"):
        if "taskId" in data["raw"]:
            if data["res"]["err_code"] == 0:
                data["res"]["err_code"] = 2
            callback_new(data, log_prefix)
        else:
            callback(data, log_prefix)
    else:
        logger.info("{} RET_TO_QUEUE - data: {}".format(
                log_prefix, json.dumps(data["res"], ensure_ascii=False)))
        __push_result_to_data_queue(data, data["callback"], log_prefix)

