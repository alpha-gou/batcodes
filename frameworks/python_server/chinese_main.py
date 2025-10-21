import json
import time
import requests
import logging
import threading
import sys

from utils import dataqueue
from utils.const import *
from utils.prompts import get_chinese_jx_json
from utils.processor import __push_result_to_data_queue, __post_vllm_stream_thinking


# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class myThread (threading.Thread):
    def __init__(self, index):
        threading.Thread.__init__(self)
        self.index = index
    def run(self):
        print("开始线程：" + str(self.index))
        infer_main(self.index)
        print("退出线程：" + str(self.index))


def __callback(data, log_prefix):
    try:
        headers = {"Content-Type": "application/json"}
        resp = requests.post(data["callbackUrl"], data=json.dumps(data), headers=headers)
        logging.info("{} CALLBACK resp: {}".format(log_prefix, resp.text))
    except Exception as e:
        logging.info("{} CALLBACK error: {}".format(log_prefix, str(e)))
        time.sleep(10)


def request_chinese_jx(json_data, log_prefix):
    result = ""
    for _ in range(3):
        _, result = __post_vllm_stream_thinking(CHINESE_JX_URL, json_data)
        if result:
            break
        else:
            logging.info("{} post_vllm_stream failed".format(log_prefix))
    return result


def infer_main(index):
    """
    主逻辑入口, 死循环监听队列
    """
    logging.info("CHINESE thread {} init: sleep {}s ...".format(index, SLEEP_SECONDS))
    time.sleep(SLEEP_SECONDS)
    while True:
        try:
            datas = dataqueue.get_dq_data(
                group_name = DATASTACK_GROUP,
                queue_name = CHINESE_JX_QUEUE_NAME,
                max_num = 1
            )
        except:
            logging.error("CHINESE thread {} :".format(index) + "requests dataqueue service err")
            time.sleep(600)  # 队列出现问题时，等待10分钟
            continue

        # 如果没有数据, 等待SLEEP_SECONDS_WHILE_NO_DATA秒后继续轮询
        if len(datas) == 0:
            logging.info("CHINESE thread {} : queue is empty! sleep {}s and continue".format(
                    index, SLEEP_SECONDS_WHILE_NO_DATA))
            time.sleep(SLEEP_SECONDS_WHILE_NO_DATA)
            continue

        # ===============================
        #     转化为json & 合法性判断
        # ===============================
        try:
            data = json.loads(datas[0])
            taskId = data["taskId"]
            text = data["text"]
            data["result"] = ""
            basic_log_prefix = "CHINESE taskId {} -".format(taskId)
            logging.info("{} ori queue {}".format(basic_log_prefix, json.dumps(data, ensure_ascii=False)))
            data["errNo"] = 1
            data["errMsg"] = "生产失败"
        except Exception as e:
            logging.error("thread {} data json parse failed with: {}".format(index, str(e)))
            continue

        # 请求模型
        try:
            json_data = get_chinese_jx_json(text)
            data["result"] = request_chinese_jx(json_data, basic_log_prefix)
        except Exception as e:
            logging.error("{} request llm failed with: {}".format(basic_log_prefix, str(e)))

        if len(data["result"]) > 10:
            data["errNo"] = 0
            data["errMsg"] = ""
            logging.info("{} generate SUCCESS !!".format(basic_log_prefix))
        else:
            logging.info("{} generate FAILED !!".format(basic_log_prefix))

        logging.info("{} Result - data: {}".format(
                basic_log_prefix, json.dumps(data, ensure_ascii=False)))

        # 判断是否需要回调
        if data["version"] == 0:
            __push_result_to_data_queue(data, CHINESE_DEBUG_QUEUE_NAME, basic_log_prefix)
        else:
            __callback(data, basic_log_prefix)


if __name__ == "__main__":
    data_all = [[] for xx in range(10)]   # 并发20以内没有问题
    thread_list = []
    for i, datas in enumerate(data_all):
        thread = myThread(i)
        thread.start()
        thread_list.append(thread)
    for thread in thread_list:
        thread.join()

