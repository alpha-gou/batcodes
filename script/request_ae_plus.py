import json
import re
import time
import requests
from multi_thread import MultiThreadRequester


QWQ_SCORING_URL = "http://aigc-aivideo-qwq32b-vllm.alpha-gou-mock.dd/v1/chat/completions"
QWEN3_SCORING_URL = "http://aigc-aivideo-qwen3-vllm.alpha-gou-mock.dd/v1/chat/completions"
AE_EXPAND_PROMOT_V5 = """你是一个小学数学老师，现在给你一个题目解析，和一个参考答案，请判断题目解析中答案是否正确。
注意:
1. 参考答案有时候可能会有错误，也可能是无效的，因此在判断时不能完全依赖参考答案。
2. 题目解析的开头是题干，题干的内容可能存在问题或者歧义，这种情况不管解析的答案是什么都认为是错误的。
请逐步推理，如果解析答案是正确的，结果返回@正确@，否则返回@错误@。返回的结果只能有一个，当有多个小题时，必须每个小题全部正确，才能返回@正确@，否则返回@错误@
【题目解析】
%s
【参考答案】
%s
"""

def __get_ae_expand_promot(text, answer):
    try:
        res_str_list = []
        data = json.loads(text)
        for jt in data:
            if data[jt]["内容类型"] in {"读题", "解答"}:
                res_str_list.append(re.sub("@.*?@", " ", data[jt]["屏幕内容"]))
        fixed_str = "\n".join(res_str_list)
        return AE_EXPAND_PROMOT_V5 % (fixed_str, answer)
    except Exception as e:
        print(e)
        return ""


def get_ae_expand_request_json(text, answer):
    prompt = __get_ae_expand_promot(text, answer)
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
        print("__post_vllm_stream_thinking error with: {}".format(str(e)))
    return thinking, result


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
            print("request_answer_equal_qwq error with: {}".format(str(e)))
        else:
            break
    return final_answer


def request_answer_equal_qwq_v2(text, answer):
    json_data = get_ae_expand_request_json(text, answer)
    thinking, result, final_answer = "", "", -1
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
            print("request_answer_equal_qwq error with: {}".format(str(e)))
        else:
            break
    return thinking, result, final_answer


class AERequester(MultiThreadRequester):
    def request_main(self, data):
        """请求的主逻辑，可按需重写此函数"""
        data["think"], data["result"], data["qwq_res"] = request_answer_equal_qwq_v2(data["xgv6_res"], data["ans1"])
        self._safe_write_csv(data)


if __name__ == "__main__":
    requester = AERequester(api_url=QWQ_SCORING_URL, max_workers=20, rate_limit=5)
    requester.load_csv("mm/ae_test_529_ae_res2.csv")
    requester.output_to_csv("mm/ae_test_529_ae_res3.csv", add_cols=["think", "result", "qwq_res"])
    requester.run()
