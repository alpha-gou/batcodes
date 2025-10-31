import requests
import json
from multi_thread import MultiThreadRequester
from qwen_prompts import get_qwen_scorint_promot_round1, get_qwen_scorint_promot_round2


class QwenVllmRequester(MultiThreadRequester):
    def request_main(self, data):
        """请求的主逻辑，可按需重写此函数"""
        target_str = data["script"]
        if data["ae_res_1"] != 1:
            data |= {"r1_prob": -1, "r1_text": "", "r2_prob": -1, "r2_text": ""}
            self._safe_write_csv(data)
            return
        prompt = get_qwen_scorint_promot_round1(target_str)
        json_data = {
            "model": "qwen",
            "prompt": prompt,
            "stream": False,
            "max_tokens":500,
            "temperature": 0,
            # "logprobs": True
        }

        r1_prob, r1_text, r2_prob, r2_text = -1, "", -1, ""
        r1_prob, r1_text = self.post_vllm(json_data)
        prompt = get_qwen_scorint_promot_round2(target_str)
        if r1_prob == 1:
            json_data["prompt"] = prompt
            r2_prob, r2_text = self.post_vllm(json_data)
        data |= {"r1_prob": r1_prob, "r1_text": r1_text, "r2_prob": r2_prob, "r2_text": r2_text}
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
    # API_URL = "http://alpha-gou-mock/v2/models/tensorrt_llm_bls/generate_stream"
    API_URL = "http://alpha-gou-mock/v1/completions"
    requester = QwenVllmRequester(api_url=API_URL, max_workers=200, rate_limit=10)
    requester.load_csv("fjs_ae_res.csv")
    requester.output_to_csv("fjs_gs_res.csv", add_cols=["r1_prob", "r1_text", "r2_prob", "r2_text"])
    requester.run()
