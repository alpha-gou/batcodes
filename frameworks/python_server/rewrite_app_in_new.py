import json
import time
import requests
import logging
import threading
import sys
import yaml

from utils import dataqueue
from utils import processor
from utils.data_structure import construct_middle_data
from utils.const import *


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


def infer_main(index):
    """
    主逻辑入口, 死循环监听队列
    """
    logging.info("thread {} init: sleep {}s ...".format(index, SLEEP_SECONDS))
    time.sleep(SLEEP_SECONDS)
    while True:
        try:
            datas = dataqueue.get_dq_data(
                group_name = DATASTACK_GROUP,
                queue_name = REWRITE_MAIN_INPUT_QUEUE,
                max_num = 1
            )
        except:
            logging.error("thread {} :".format(index) + "requests dataqueue service err")
            time.sleep(600)  # 队列出现问题时，等待10分钟
            continue

        # 如果没有数据, 等待SLEEP_SECONDS_WHILE_NO_DATA秒后继续轮询
        if len(datas) == 0:
            logging.info("thread {} : queue is empty! sleep {}s and continue".format(
                    index, SLEEP_SECONDS_WHILE_NO_DATA))
            time.sleep(SLEEP_SECONDS_WHILE_NO_DATA)
            continue

        # ===============================
        #     转化为json & 合法性判断
        # ===============================
        try:
            raw_data = json.loads(datas[0])
            is_vip = raw_data.get("vip", False)
            version = raw_data.get("version", 1)
            callback_url = raw_data.get("callbackUrl", NEW_CALLBACK_URL)
            data = construct_middle_data(raw_data, REWRITE_MAIN_CONTROL_QUEUE, callback_url, is_vip)
            script = data["raw"]["script"]
            data["raw"]["teachertid"] = data["raw"]["tid"]
            data["raw"]["queryid"] = data["raw"]["taskId"]
            data["raw"]["version"] = version
            data["base"]["answer"] = data["raw"]["questionInfo"]
            data["base"]["question"] = data["raw"]["answerInfo"]
            basic_log_prefix = "Version_{} taskId {} tid {} retry {}".format(
                1, data["raw"]["taskId"], data["raw"]["tid"], data["retry_times"])
        except Exception as e:
            logging.error("thread {} data json parse failed with: {}".format(index, str(e)))
            continue

        logging.info('APP_IN_NEW taskId {} version {} - data: {}'.format(
                data["raw"]["taskId"], 1, json.dumps(data, ensure_ascii=False)))

        # 学科判断
        processor.req_sub_dep(data)
        if not data["base"]["subject"] or data["base"]["subject"] != "数学":
            logging.info("{} subject_error, data: {}".format(
                    basic_log_prefix, json.dumps(data["base"], ensure_ascii=False)))
            data["res"] = {"err_code": 10005, "err_info": "学科异常", "script": ""}
            data["retry_times"] = MAX_RETRY_TIMES + 1
            processor.__result_router(data, basic_log_prefix)
            continue

        # 图题关键字过滤
        if version == 2:
            if ("阴影部分" in script) and ("周长" in script or "面积" in script):
                logging.info("{} image_type_error, data: {}".format(
                        basic_log_prefix, json.dumps(data["raw"], ensure_ascii=False)))
                data["res"] = {"err_code": 60001, "err_info": "阴影面积题不生产", "script": ""}
                data["retry_times"] = MAX_RETRY_TIMES + 1
                processor.__result_router(data, basic_log_prefix)
                continue

        # 新框架没有stage0，直接stage1起步
        logging.info("{} TRANS stage 0 >> 1 - into xg queue.".format(basic_log_prefix))
        processor.script_rewrite(data, basic_log_prefix)


if __name__ == "__main__":
    data_all = [[] for xx in range(THREAD_NUM)]
    thread_list = []
    for i, datas in enumerate(data_all):
        thread = myThread(i)
        thread.start()
        thread_list.append(thread)
    for thread in thread_list:
        thread.join()