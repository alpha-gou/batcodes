import json
import time
import requests
import logging
import threading
import sys
import yaml

from utils import dataqueue
from utils import processor
from utils.const import *
from chinese_main import __callback


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
    logging.info("thread {} init: sleep {}s ...".format(index, 60))
    time.sleep(6)
    while True:
        try:
            datas = dataqueue.get_dq_data(
                group_name = DATASTACK_GROUP,
                # queue_name = SHIP_RES_QUEUE_NAME,   # 只取ship队列内容
                queue_name = CHINESE_DEBUG_QUEUE_NAME,   # 只取ship队列内容
                max_num = 1
            )
        except:
            logging.error("thread {} :".format(index) + "requests dataqueue service err")
            time.sleep(600)
            continue

        # 如果没有数据, 等待SLEEP_SECONDS_WHILE_NO_DATA秒后继续轮询
        if len(datas) == 0:
            logging.info("thread {} : queue is empty! sleep {}s and continue".format(
                    index, SLEEP_SECONDS_WHILE_NO_DATA))
            time.sleep(SLEEP_SECONDS_WHILE_NO_DATA)
            continue

        # try:
        #     logging.info("thread {} :".format(index) + "ori queue data, Msg: {}".format(datas[0]))
        #     data = json.loads(datas[0])
        #     query_id = data["raw"]["queryid"]
        #     tid = data["raw"]["tid"]
        # except Exception as e:
        #     logging.error("thread {} data json parse failed with: {}".format(index, str(e)))
        #     continue

        # ===============================
        #         路由分发主逻辑
        # processor函数均原地修改且带异常检测
        # ===============================
        # 测试时，直接callback
        # log_prefix = "SHIP queryid {} tid {} retry {}".format(query_id, tid, data["retry_times"])
        # callback(data, log_prefix)

        logging.info("thread {} :".format(index) + "ori queue data, Msg: {}".format(datas[0]))
        data = json.loads(datas[0])
        __callback(data, "ship recall")



def callback(data, log_prefix):
    try:
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
        logging.info("{} RET_CALLBACK - data: {}".format(
                log_prefix, json.dumps(script_data, ensure_ascii=False)))
        resp = requests.post(SHIP_CALLBACK_URL,  data=json.dumps(callback_data), timeout=300)
        logging.info("{} CALLBACK resp: {}".format(log_prefix, resp.text))
    except Exception as e:
        logging.info("{} CALLBACK error: {}".format(log_prefix, str(e)))
        time.sleep(10)


if __name__ == "__main__":
    data_all = [[] for xx in range(THREAD_NUM)]

    thread_list = []
    for i, datas in enumerate(data_all):
        thread = myThread(i)
        thread.start()
        thread_list.append(thread)
    for thread in thread_list:
        thread.join()

