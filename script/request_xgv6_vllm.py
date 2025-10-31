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


class Xgv6Requester(MultiThreadRequester):
    def request_main(self, data):
        """请求的主逻辑，可按需重写此函数"""
        prompt = INPUT_PROMOT_TEMPLATE.format(data["raw_script"])

        json_data = {
            "model": "xgv6",
            "prompt": prompt,
            "stream": False,
            "max_tokens": 8192,
            "temperature": 0.0,
        }

        data["xgv6_res"] = ""
        for i in range(10):
            try:
                xgv6_res = self.post_vllm(json_data)
                error_code = get_error_code(xgv6_res)
                if error_code == 0:
                    data["xgv6_res"] = xgv6_res
                    break
                else:
                    raise ValueError("bad json !!")
            except Exception as e:
                print(data["tid"], "retry: ",  i, " error: ", e)
            json_data["temperature"] = 0.2
        self._safe_write_csv(data)

    def post_vllm(self, json_data):
        for attempt in range(self.retry_attempts + 1):
            try:
                headers = {"Content-Type": "application/json"}
                response = requests.post(self.api_url, headers=headers, json=json_data)
                response_json = response.json()
                response_text = response_json["choices"][0]["text"]
                return response_text
            except Exception as e:
                print(e)
                json_data["temperature"] += 0.1
        return ""


if __name__ == "__main__":
    API_URL = "http://alpha-gou-mock:8986/v1/completions"
    requester = Xgv6Requester(api_url=API_URL, max_workers=200, rate_limit=10)
    requester.load_csv("/mm/.csv")
    requester.output_to_csv(
            "mm/img_caption/image_caption_old_xgv6_res.csv", add_cols=["xgv6_res"])
    requester.run()
