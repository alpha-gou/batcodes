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
                queue_name = REWRITE_MAIN_CONTROL_QUEUE,
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
            data = json.loads(datas[0])
            query_id = data["raw"]["queryid"]
            tid = data["raw"]["tid"]
            version = data["raw"]["version"]
            basic_log_prefix = "Version_{} queryid {} tid {} retry {}".format(
                version, query_id, tid, data["retry_times"])
        except Exception as e:
            logging.error("thread {} data json parse failed with: {}".format(index, str(e)))
            continue

        # ===============================
        #         路由分发主逻辑
        # processor函数均原地修改且带异常检测
        # ===============================

        # 阶段0，未改写，完善base信息后压入改写队列
        if data["stage"] == 0:
            stage_0_log_prefix = "{} STAGE_0 -".format(basic_log_prefix)
            processor.process_stage_0(data, stage_0_log_prefix)

        # 阶段1，改写完成，策略判断后压入评分队列或分发队列
        elif data["stage"] == 1:
            stage_1_log_prefix = "{} STAGE_1 -".format(basic_log_prefix)
            processor.process_stage_1(data, stage_1_log_prefix)

        # 阶段2，第一轮评分完成
        elif data["stage"] == 2:
            stage_2_log_prefix = "{} STAGE_2 -".format(basic_log_prefix)
            processor.process_stage_2(data, stage_2_log_prefix)

        # 阶段3，第二轮评分完成，进入结果路由
        elif data["stage"] == 3:
            stage_3_log_prefix = "{} STAGE_3 -".format(basic_log_prefix)
            processor.process_stage_3(data, stage_3_log_prefix)

        # 阶段98，仅评分不改写
        elif data["stage"] == 98:
            stage_98_log_prefix = "{} STAGE_98 -".format(basic_log_prefix)
            processor.process_stage_98(data, stage_98_log_prefix)

        # 错误阶段报警
        else:
            logging.error(
                "thread {} - ERROR: stage error with stage: {}".format(index, data["stage"])
            )


if __name__ == "__main__":
    data_all = [[] for xx in range(THREAD_NUM)]
    thread_list = []
    for i, datas in enumerate(data_all):
        thread = myThread(i)
        thread.start()
        thread_list.append(thread)
    for thread in thread_list:
        thread.join()

