# encoding:UTF-8
import json
import logging
import yaml
import sys
import numpy as np
from winrain import *

import utils.dataqueue as dataqueue
import utils.data_structure as datastruc
from utils.const import *


# log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MyApp(winrain.WinRain):
    def __init__(self):
        winrain.WinRain.__init__(self)
        # 此处可以添加自己的需要预先load的数据


my_app = MyApp()


@my_app.add_uri('/ready')
def test_inter():
    data = dict()
    data['code'] = '0'
    data['msg'] = 'ok'
    return json.dumps(data)


@my_app.add_uri('/modelVersion')
def get_model_version():
    data = {
      "errNo": 0,
      "errMsg": "",
      "version": "1.0",
      "cost": ""
    }
    return json.dumps(data)


@my_app.add_uri('/generate_main')
def generate_main():
    """
    脚本生产主接口，将数据压入队列
    请求格式：
    {
        "queryid": "", "tid":1, "teachertid":1, "script":"",
        "grade":1, "subject":2, "version":3, "ext":"{\"caption\": \"图像描述\"}",
    }
    debug测试时，"ext"增加callback字段：
    "ext":"{\"callback\": \"math_debug_result\"}"
    """
    try:
        raw_str: str = my_app.get_body().decode('utf-8')
    except:
        logging.error('request error: failed to get raw_data')
        return json.dumps({"errNo": 1, "errMsg": "请求失败"})

    # 转化数据，生成流转于各个队列的数据结构
    res = __server_main(raw_str, CALLBACK_URL, no_xg=False)
    return res


@my_app.add_uri('/debug_main')
def debug_main():
    """
    仅内部测试用
    全流程，结果入debug队列，等价于使用/generate_main并在ext字段中添加callback
    """
    try:
        raw_str: str = my_app.get_body().decode('utf-8')
    except:
        logging.error('request error: failed to get raw_data')
        return json.dumps({"errNo": 1, "errMsg": "请求失败"})

    # 转化数据，生成流转于各个队列的数据结构
    res = __server_main(raw_str, DEBUG_RES_QUEUE_NAME, no_xg=False, vip=True)
    return res


@my_app.add_uri('/generate_main_ship')
def generate_main_ship():
    """
    ship测试环境使用
    全流程，结果入ship队列，等价于使用/generate_main并在ext字段中添加callback
    """
    try:
        raw_str: str = my_app.get_body().decode('utf-8')
    except:
        logging.error('request error: failed to get raw_data')
        return json.dumps({"errNo": 1, "errMsg": "请求失败"})

    # 转化数据，生成流转于各个队列的数据结构
    res = __server_main(raw_str, SHIP_RES_QUEUE_NAME, no_xg=False)
    return res


@my_app.add_uri('/script_filter')
def script_filter():
    """
    仅内部测试用
    不改写，只走挑选流程，输入参数和其他接口一致
    """
    try:
        raw_str: str = my_app.get_body().decode('utf-8')
    except:
        logging.error('request error: failed to get raw_data')
        return json.dumps({"errNo": 1, "errMsg": "请求失败"})

    # 转化数据，生成流转于各个队列的数据结构
    res = __server_main(raw_str, DEBUG_RES_QUEUE_NAME, no_xg=True, vip=True)
    return res


@my_app.add_uri('/chinese_jx')
def chinese_jx():
    """
    脚本生产主接口，将数据压入队列
    请求格式：
    {
        "taskId": "",
        "text": "",
        "callbackUrl": "",
        "version": 1,  # 0代表debug模式
        "ext": "",
    }
    """
    try:
        raw_str: str = my_app.get_body().decode('utf-8')
        raw_data = json.loads(raw_str)
        if "callbackUrl" not in raw_data:
            return json.dumps({"errNo": 4, "errMsg": "未指定callbackUrl"})
        if ("taskId" in raw_data) and ("text" in raw_data) and (len(raw_data["text"]) > 0):
            res = __write_into_data_queue(
                    CHINESE_JX_QUEUE_NAME, json.dumps(raw_data, ensure_ascii=False), "chinese_jx")
        else:
            res = {"errNo": 2, "errMsg": "数据格式非法"}
    except:
        logging.error('request error: failed to get raw_data')
        res = {"errNo": 1, "errMsg": "请求失败"}
    return res


def __server_main(raw_str, callback, no_xg=False, vip=False):
    # 转化数据，生成流转于各个队列的数据结构
    try:
        raw_data = json.loads(raw_str)
        query_id = raw_data["queryid"]
        version = raw_data["version"]
        if version not in VALID_VERSION:
            return json.dumps({"errNo": 5, "errMsg": "version字段非法"})
        if "ext" in raw_data and raw_data["ext"]:
            ext_data = json.loads(raw_data["ext"])
            raw_data["ext"] = ext_data
            if "callback" in ext_data:
                callback = ext_data["callback"]
        data = datastruc.construct_middle_data(raw_data, REWRITE_MAIN_CONTROL_QUEUE, callback, vip)
        if no_xg:
            data["stage"] = 98  # 跳至stage98
        data_str = json.dumps(data, ensure_ascii=False)
    except Exception as e:
        logging.error("data_preprocess failed, Error Message: {}".format(e))
        return json.dumps({"errNo": 2, "errMsg": "数据格式非法"})

    logging.info('APP_IN queryid {} version {} - data: {}'.format(query_id, version, data_str))
    return __write_into_data_queue(REWRITE_MAIN_CONTROL_QUEUE, data_str, query_id, data.get("vip", False))


def __write_into_data_queue(queue_name, data_str, query_id, write_to_head=False):
    # 写入数据到路由队列
    try:
        response = dataqueue.write_dq_data(
            group_name=DATASTACK_GROUP,
            queue_name=queue_name,
            data=[data_str],
            write_to_head=write_to_head,
        )
        if response["errNo"] != 0:
            logging.error(
                'dataqueue write error: {}'.format(json.dumps(response, ensure_ascii=False)))
            raise ValueError()
    except Exception as e:
        logging.error('write queue error: {}'.format(e))
        return json.dumps({"errNo": 3, "errMsg": "队列写入失败"})
    else:
        logging.info('successfully write queryid - {} - data into queue: {}'.format(
                query_id, queue_name))
        return json.dumps({"errNo": 0, "errMsg": "it works"})

