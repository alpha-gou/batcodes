import os
import asyncio
import aiohttp
import aiofiles
import csv
from io import StringIO
from functools import wraps
from tqdm import tqdm


# ========== 装饰器 ==========
def retry(max_retries=3, delay=1):
    """通用异步重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(delay)
        return wrapper
    return decorator


# ========== 基础类 ==========
class AsyncRequestClient:
    """异步HTTP客户端，需要继承使用"""
    def __init__(self, concurrency=10):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc):
        await self.session.close()

    async def process_row(self, row):
        """需子类实现的具体处理逻辑"""
        raise NotImplementedError

    async def raise_error(self, e, row):
        """需子类实现的报错逻辑"""
        raise NotImplementedError

    @retry(max_retries=3)
    async def __process_row_case(self, row):
        """示例process_row方法，禁止调用"""
        url = "http://1.2.3.4:8405/test_api"
        data = {**row}
        async with self.session.post(url, json=data) as resp:
            text = await resp.text()
            return json.loads(text)


class AsyncCSVWriter:
    """
    异步安全CSV写入器
    通过如下方式定义写入器：
        async with AsyncCSVWriter(output_filename, col_list) as writer:
    使用写入器：
        await writer.safe_write_row({**data_row, **result})
    其中output_filename为输出文件名，col_list为各列的列名，写入时的dict键值需要和col_list相同
    """
    def __init__(self, filename, col_list):
        self.filename = filename
        self.header = col_list
        self.file = None

    async def __aenter__(self):
        self.file = await aiofiles.open(self.filename, 'a', encoding='utf-8-sig')
        if os.path.getsize(self.filename) == 0:
            await self._write_header()
        return self

    async def __aexit__(self, *exc):
        await self.file.close()

    async def _write_header(self):
        """动态生成CSV表头"""
        buffer = StringIO()
        csv.writer(buffer).writerow(self.header)
        await self.file.write(buffer.getvalue())

    async def safe_write_row(self, data):
        """安全写入单行数据"""
        buffer = StringIO()
        row = [data[key] for key in self.header]
        csv.writer(buffer, quoting=csv.QUOTE_MINIMAL).writerow(row)
        await self.file.write(buffer.getvalue())


# ========== 处理框架 ==========
class AsyncDataProcessor:
    """
    异步数据处理框架
    需传入输入输出文件地址，以及api访问client_class，client_class继承自AsyncRequestClient
    1. 实现api访问类：
    class CustomRequestClient(AsyncRequestClient):
        async def request_data(self, tid: int):
            pass
    2. 实现每行具体逻辑：
    class CustomDataProcessor(AsyncDataProcessor):
        client_class = CustomRequestClient
        
        async def process_row(self, client, writer, row, progress):
            result = await client.enhanced_api_call(row.url)
            await writer.safe_write([row.id, result])
    3. 使用异步数据处理框架
    processor = AsyncDataProcessor(input_csv, output_csv, extra_columns, CustomRequestClient, 200)
    processor.run()
    """
    def __init__(
        self,
        input_csv: str,
        output_csv: str,
        extra_columns: list,
        client_class: type,
        concurrency: int = 10,
        filter_func = lambda x : x
    ):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.concurrency = concurrency
        self.input_df = filter_func(pd.read_csv(input_csv))
        self.headers = list(self.input_df.columns) + extra_columns
        self.client_class = client_class

    async def process_row(self, client, writer, row, progress):
        """处理单行数据工作流，具体请求逻辑在继承的client类中实现"""
        try:
            result = await client.process_row(row)
            await writer.safe_write_row({**row, **result})
        except Exception as e:
            client.raise_error(e, row)
        finally:
            progress.update(1)

    async def _main_flow(self):
        async with self.client_class(self.concurrency) as client, \
                   AsyncCSVWriter(self.output_csv, self.headers) as writer:
            with tqdm(total=len(self.input_df), desc='Processing') as progress:
                tasks = [
                    self.process_row(client, writer, row, progress)
                        for row in self.input_df.itertuples()
                ]
                await asyncio.gather(*tasks)

    def run(self):
        """启动入口"""
        asyncio.run(self._main_flow())


# ========== 辅助工具 ==========
def no_filter(df):
    return df



