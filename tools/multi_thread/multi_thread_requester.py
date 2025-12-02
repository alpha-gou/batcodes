import csv
import json
import pandas as pd
import queue
import requests
import threading
import time
from tqdm import tqdm


class TokenBucket:
    """简化的令牌桶算法实现"""
    def __init__(self, capacity, fill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.fill_rate = fill_rate
        self.last_update = time.time()

    def consume(self, tokens=1):
        now = time.time()
        delta = now - self.last_update
        self.tokens = min(self.tokens + delta * self.fill_rate, self.capacity)
        self.last_update = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class MultiThreadRequester:
    def __init__(self, api_url, max_workers, rate_limit, retry_attempts=2):
        """
        参数说明：
        - api_url: vLLM服务地址
        - max_workers: 最大线程数，黄金配比公式为：max_workers = rate_limit × 1.2
        - rate_limit: 每秒请求数限制，即目标QPS值
        - retry_attempts: 失败重试次数
        """
        # 核心配置
        self.api_url = api_url
        self.retry_attempts = retry_attempts

        # 线程控制
        self.task_queue = queue.Queue()
        self.worker_threads = []
        self._shutdown_flag = False

        # 速率控制
        self.token_bucket = TokenBucket(capacity=rate_limit, fill_rate=rate_limit)
        self.rate_lock = threading.Lock()

        # 输出配置
        self.csv_lock = None
        self.output_csv = None

        # 进度条
        self.total_data = 0
        self.pbar = None

        # 初始化工作线程
        for _ in range(max_workers):
            t = threading.Thread(target=self._worker)
            t.daemon = True
            t.start()
            self.worker_threads.append(t)

    def _init_csv(self):
        """初始化CSV文件头"""
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.col_list)
            writer.writeheader()

    def _safe_write_csv(self, row_dict):
        """线程安全写入CSV"""
        with self.csv_lock:
            with open(self.output_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.col_list)
                writer.writerow(row_dict)

    def _worker(self):
        """工作线程（消费者）核心逻辑"""
        while not self._shutdown_flag:
            try:
                # 获取任务
                data = self.task_queue.get(block=False)
            except queue.Empty:
                time.sleep(0.1)
                continue

            # 速率控制
            with self.rate_lock:
                while not self.token_bucket.consume():
                    time.sleep(0.01)

            # 请求执行
            self.request_main(data)
            self.task_queue.task_done()
            self.pbar.update(1)

    def load_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        self.load_dataframe(df)

    def load_dataframe(self, df):
        self.col_list = list(df.columns)
        rows_dicts = df.to_dict(orient="records")
        for data in rows_dicts:
            self.add_data(data)


    def output_to_csv(self, output_path, col_list:list=None, add_cols:list=None):
        """
        参数说明：
        - output_csv: 输出CSV文件路径
        - col_list: 输出CSV文件列名，全量替换；为空时使用原列名
        - add_cols: 输出CSV文件列名，原列增加
        """
        self.csv_lock = threading.Lock()  # CSV写入锁
        self.output_csv = output_path
        if col_list is not None:
            self.col_list = col_list
        elif add_cols is not None:
            self.col_list = self.col_list + add_cols
        self._init_csv()

    def conitnue_run(self, input_csv, output_csv, key):
        """
        断点续跑，根据key过滤output_csv中已经跑过的
        """
        print("parsing data ...")
        df1 = pd.read_csv(input_csv)
        df2 = pd.read_csv(output_csv)
        df = df1[~df1[key].isin(df2[key])]
        self.col_list = df2.columns.to_list()
        print("conitnue_run: total %d, complited %d, remain %d" % (
                df1.shape[0], df2.shape[0], df.shape[0]))
        rows_dicts = df.to_dict(orient="records")
        self.data_cols = list(rows_dicts[0].keys())
        for data in rows_dicts:
            self.add_data(data)
        self.output_to_csv(output_csv, append=True)

    def add_data(self, data):
        """添加待处理数据到处理队列"""
        self.task_queue.put(data)
        self.total_data += 1

    def shutdown(self):
        """优雅关闭"""
        self._shutdown_flag = True
        for t in self.worker_threads:
            t.join()

    def run(self):
        """启动入口"""
        self.pbar = tqdm(total=self.total_data)
        self.task_queue.join()  # 等待队列处理完成
        self.pbar.close()
        self.shutdown()  # 关闭服务

    def post_vllm_thinking(self, json_data):
        """
        post请求带thinking的vllm服务
        """
        thinking, result = "", ""
        for attempt in range(self.retry_attempts + 1):
            try:
                headers = {"Content-Type": "application/json"}
                response = requests.post(self.api_url, headers=headers, json=json_data).json()
                content = response['choices'][0]['message']['content']
                content = content.strip("<think>").strip("\n")
                contents = content.split("</think>")
                thinking = contents[0].strip("\n")
                result = contents[1].strip("\n")
            except Exception as e:
                print(e)
            else:
                break
        return thinking, result

    def post_vllm_stream_thinking(self, json_data):
        text, thinking, result = "", "", ""
        headers = {"Content-Type": "application/json"}
        try:
            with requests.post(
                self.api_url, headers=headers, json=json_data, stream=True
            ) as response:
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
            print("self.post_vllm_stream_thinking error with: {}".format(str(e)))
        return thinking, result

    def request_main(self, data):
        """请求的主逻辑，须重写此函数"""
        headers = {"Content-Type": "application/json"}
        enable_thinking = True
        prompt = ""
        if enable_thinking:
            user_content = f"/think {prompt}"
        else:
            user_content = f"/no_think {prompt}"

        json_data = {
            "model": "gscore",
            "messages": [
                {"role": "system", "content": "你是一个AI助手"},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 4096,
            "temperature": 0.6,
        }
        thinking, result = self.post_vllm_thinking(json_data)
        data["answer"] = result
        data["thinking"] = thinking
        self._safe_write_csv(data)


if __name__ == "__main__":
    # 初始化请求器
    requester = MultiThreadRequester(
        api_url="http://localhost:8000/generate",
        max_workers=12,
        rate_limit=10
    )
    requester.load_csv("inputs.csv")
    requester.output_to_csv("responses.csv", add_cols=["thinking", "answer"])
    requester.run()
