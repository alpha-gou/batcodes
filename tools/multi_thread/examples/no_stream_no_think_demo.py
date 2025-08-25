import requests
import json
from multi_thread import MultiThreadRequester


class QwenVllmRequester(MultiThreadRequester):
    def request_main(self, data):
        """请求的主逻辑，可按需重写此函数"""
        json_data = {
            "model": "qwen",
            "prompt": data["prompt"],
            "stream": False,
            "max_tokens":500,
            "temperature": 0,
            # "logprobs": True
        }

        prob, text = self.post_vllm(json_data)
        data |= {"prob": prob, "text": text}
        self._safe_write_csv(data)

    def post_vllm(self, json_data):
        for attempt in range(self.retry_attempts + 1):
            try:
                headers = {"Content-Type": "application/json"}
                response = requests.post(self.api_url, headers=headers, json=json_data)
                response_json = response.json()
                text = response_json["choices"][0]["text"]
                res = int(text[1])
                probs = -1
                if res > 0:
                    probs = 1
                elif res == 0:
                    probs = 0
                else:
                    raise ValueError("bad result number !!")
                return probs, text
            except Exception as e:
                print(e)
                json_data["temperature"] += 0.1
        return -1, ""


if __name__ == "__main__":
    API_URL = "http://demo-url/v1/completions"
    requester = QwenVllmRequester(api_url=API_URL, max_workers=200, rate_limit=10)
    requester.load_csv("demo_input.csv")
    requester.output_to_csv("demo_output.csv", add_cols=["prob", "text"])
    requester.run()
