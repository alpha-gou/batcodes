import json
import re
import time
import requests
from multi_thread import MultiThreadRequester
from qwen_prompts import PROMPT_RM3_V2


# QWEN3_URL = "http://alpha-gou-mock:8991/v1/chat/completions"


RES_DICT = {"合格": 0, "不合格": 1}


class AERequester(MultiThreadRequester):
    def request_main(self, data):
        tid = data["tid"]
        # prompt = PROMPT_RM3 + data["script"]
        prompt = PROMPT_RM3_V2 + data["script"]
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
        _, result = self.post_vllm_stream_thinking(json_data)
        final_result = self.extract_result(result)
        data["RM3"] = RES_DICT.get(final_result, -1)
        if data["RM3"] == 1:
            print("pick!!")
        self._safe_write_csv(data)

    def extract_result(self, text):
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


if __name__ == "__main__":
    requester = AERequester(api_url=QWEN3_URL, max_workers=10, rate_limit=15)
    requester.load_csv("case_debug_1017_for_rm3.csv")
    requester.output_to_csv("case_debug_1017_rm3_res5.csv", add_cols=["RM3"])
    # requester.conitnue_run("data/human_0824_p1_100w.csv", "data/human_0824_p1_100w_rm3_res.csv", "tid")
    requester.run()