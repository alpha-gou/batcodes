import requests
import json
import re
from format_check import get_error_code
from multi_thread import MultiThreadRequester


INPUT_PROMOT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{}
<|im_end|>
<|im_start|>assistant
"""


def is_bad_json(x):
    try:
        x = json.loads(x)
        return False
    except:
        return True


class Xgv6Requester(MultiThreadRequester):
    def request_main(self, data):
        """请求的主逻辑，可按需重写此函数"""
        if type(data["script"]) == "str" and data["script"]:
            self._safe_write_csv(data)
            return
        if type(data["original_script"]) == "str" or not data["original_script"]:
            self._safe_write_csv(data)
            return
        prompt = INPUT_PROMOT_TEMPLATE.format(data["original_script"])
        json_data = {
            "text_input": prompt,
            "max_tokens":4096,
            "end_id":151645,
            # "pad_id":151643,
            "temperature": 0.2,
            "stream":True
        }

        for i in range(10):
            try:
                xgv6_res = self.post_triton_stream(json_data)
                error_code = get_error_code(xgv6_res)
                if error_code == 0:
                    data["script"] = xgv6_res
                    break
                else:
                    raise ValueError("bad json !!")
            except Exception as e:
                print(data["tid"], "retry: ",  i, " error: ", e)                
        self._safe_write_csv(data)

    def post_triton(self, json_data):
        for attempt in range(self.retry_attempts + 1):
            try:
                headers = {"Content-Type": "application/json"}
                response = requests.post(self.api_url, headers=headers, json=json_data)
                response.raise_for_status()
                response_text = response.text[6:]
                if is_bad_json(response_text):
                    raise ValueError("bad json !!")
                return response_text
            except Exception as e:
                print(e)
                json_data["temperature"] += 0.1
        return ""

    def post_triton_stream(self, json_data):
        text = ""
        headers = {"Content-Type": "application/json"}
        with requests.post(API_URL, headers=headers, json=json_data, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith('data: '):
                        chunk = json.loads(decoded_line[6:])  # 去除"data:"前缀
                        if chunk['text_output']:
                            text += chunk['text_output']
        return text


if __name__ == "__main__":
    API_URL = "http://aigc-aivdeo-math-rewrite.rec-strategy.zuoyebang.dd/v2/models/tensorrt_llm_bls/generate_stream"    
    requester = Xgv6Requester(api_url=API_URL, max_workers=20, rate_limit=5)
    requester.load_csv("rundata/0908_mm_data_4.csv")
    requester.output_to_csv("rundata/0908_mm_data_5.csv")
    requester.run()
