import requests
import json
from multi_thread import MultiThreadRequester
from qwen_prompts import *


class QwenTritonRequester(MultiThreadRequester):
    def request_main(self, data):
        """请求的主逻辑，可按需重写此函数"""
        target_str = data["script"]
        # if data["EQ"] != 1:
        #     data |= {"r1_prob": -1, "r1_text": -1, "r2_prob": -1, "r2_text": ""}
        #     self._safe_write_csv(data)
        #     return
        # if data["model"] == "泛计算":
        #     question_str=data["ques"]
        #     prompt = get_qwen_scorint_promot_round1_fjs(question_str, target_str)
        # else:
        #     prompt = get_qwen_scorint_promot_round1(target_str)
        if data["ae_res_2"] != 1:
            data |= {"rm1": -1, "rm2": -1}
            self._safe_write_csv(data)
            return
        prompt = get_qwen_scorint_promot_round1(target_str)
        json_data = {
            "text_input": prompt,
            "stream": True,
            "max_tokens":500,
            "temperature": 0,
        }
        r1_prob, r1_text = self.post_triton(json_data)

        # 剪枝
        # if r1_prob != 1:
        #     data |= {"r1_prob": r1_prob, "r1_text": r1_text, "r2_prob": -1, "r2_text": ""}
        #     self._safe_write_csv(data)
        #     return

        # if data["model"] == "泛计算":
        #     prompt = get_qwen_scorint_promot_round2_fjs(question_str, target_str)
        # else:
        #     prompt = get_qwen_scorint_promot_round2(target_str)
        prompt = get_qwen_scorint_promot_round2(target_str)
        # json_data["text_input"] = prompt
        r2_prob, r2_text = self.post_triton(json_data)
        # data |= {"r1_prob": r1_prob, "r1_text": r1_text, "r2_prob": r2_prob, "r2_text": r2_text}
        data |= {"rm1": r1_prob, "rm2": r2_prob}
        self._safe_write_csv(data)

    def post_triton(self, json_data):
        for attempt in range(self.retry_attempts + 1):
            try:
                headers = {"Content-Type": "application/json"}
                # response = requests.post(self.api_url, headers=headers, json=json_data)
                # response.raise_for_status()
                # text = json.loads(response_text[6:].strip()).get('text_output')

                text = self.post_triton_stream(json_data)
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
                print("post_triton:", e)
                json_data["temperature"] += 0.1
        return -1, ""
    
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
    API_URL = "http://aigc-k12-aivideo-rm.alpha-gou-mock.dd/v2/models/tensorrt_llm_bls/generate_stream"
    requester = QwenTritonRequester(api_url=API_URL, max_workers=32, rate_limit=5)
    requester.load_csv("rundata/0908_mm_data_6.csv")
    requester.output_to_csv("rundata/0908_mm_data_7.csv", add_cols=["rm1", "rm2"])
    # requester.output_to_csv("fjs_gs_res2.csv", add_cols=[])
    requester.run()
