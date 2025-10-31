import requests
import json
import re
from multi_thread import MultiThreadRequester
from qwen_promots import get_qwen_scorint_promot_round1, get_qwen_scorint_promot_round2


INFO_MISS_PROMOT = """请判断下面题干的内容是否完整，根据以下规则：
a) 题干信息完整，语义通顺，根据题干的信息能够进行求解，是一道有效的问题。
b) 若题干出现“如图”“如下表”等涉及图示/表格的描述，可直接判断为题干不完整。
c) 可参考解析中的内容，如果解析中出现了题干中不存在的信息(常识类信息、分情况讨论除外)，则说明题干不完整。
如果题干是完整的，返回@1@，否则返回@0@。
【题干】
%s
【解析】
%s
"""

def __get_question_incomplete_prompt(script):
    try:
        data = json.loads(script)
        question = re.sub("@.*?@", "", data["镜头1"]["屏幕内容"])
        jt_list = []
        for jt in data:
            if jt != "镜头1":
                jt_list.append(data[jt]["屏幕内容"])
        return INFO_MISS_PROMOT % (question, "\n".join(jt_list))
    except:
        return ""


def get_question_incomplete_json(data):
    # prompt = __get_question_incomplete_prompt(script)
    prompt = INFO_MISS_PROMOT % (data["ques"], data["ans"])
    if prompt:
        json_data = {
            "model": "qwq32b",
            "messages": [
                {"role": "system", "content": "你是一个AI助手"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "min_p": 0.0,
            "max_tokens": 4096,
            "stream": True
        }
        return json_data
    return {}


class QwenTritonRequester(MultiThreadRequester):
    def request_main(self, data):
        """请求的主逻辑，可按需重写此函数"""
        final_answer = -1
        # if not (data["r2_prob"] == 1 and data["err_code"] == 0):
        #     data["incomplete"] = final_answer
        #     self._safe_write_csv(data)
        #     return 

        json_data = get_question_incomplete_json(data)
        if json_data:
            for attempt in range(3):
                try:
                    thinking, result = self.post_triton(json_data)
                    if result.strip().startswith("@1@"):
                        final_answer = 1
                    elif result.strip().startswith("@0@"):
                        final_answer = 0
                    else:
                        continue
                except Exception as e:
                    print("request_question_incomplete error with: {}".format(str(e)))
                else:
                    break
        data["incomplete"] = final_answer
        self._safe_write_csv(data)

    def post_triton(self, json_data):
        text, thinking, result = "", "", ""
        headers = {"Content-Type": "application/json"}
        try:
            with requests.post(self.api_url, headers=headers, json=json_data, stream=True) as response:
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
            print("__post_vllm_stream_thinking error with: {}".format(str(e)))
        return thinking, result


if __name__ == "__main__":
    API_URL = "http://alpha-gou-mock/v1/chat/completions"
    requester = QwenTritonRequester(api_url=API_URL, max_workers=15, rate_limit=10)
    requester.load_csv("info_miss_mid.csv")
    requester.output_to_csv("info_miss_res.csv", add_cols=["incomplete"])
    # requester.output_to_csv("fjs_gs_res2.csv", add_cols=[])
    requester.run()
