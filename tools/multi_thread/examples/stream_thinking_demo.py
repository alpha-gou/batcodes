from multi_thread import MultiThreadRequester


QWEN3_URL = "http://qwen3-demo-server-url/v1/chat/completions"
PROMPT_RM3 = """this is demo prompt"""


class DemoRequester(MultiThreadRequester):
    def request_main(self, data):
        tid = data["tid"]  # 主key
        prompt = PROMPT_RM3 + data["target_script"]
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
        thinking, result = self.post_vllm_stream_thinking(json_data)
        data["thinking"], data["result"] = thinking, result
        self._safe_write_csv(data)


if __name__ == "__main__":
    requester = DemoRequester(api_url="", max_workers=200, rate_limit=20)
    requester.load_csv("demo_input.csv")
    requester.output_to_csv("demo_output.csv", add_cols=["thinking", "result"])
    # 断点续跑
    # requester.conitnue_run("demo_input.csv", "demo_output.csv", "tid")
    requester.run()
