from async_data_processor.core import AsyncDataProcessor, AsyncRequestClient, retry
import json
from async_data_processor.utils import validate_dataframe
from custom_processor import CustomProcessor
import pandas as pd


class CustomProcessor(AsyncDataProcessor):
    """自定义处理器（实现业务逻辑）"""
    def _get_headers(self):
        return ["tid", "result", "analysis"]

    async def process_row(self, client, writer, row, progress):
        try:
            # 实现具体业务逻辑
            async with client.semaphore:
                # 示例：调用某个API
                async with client.session.post('https://api.example.com', json={'tid': row.tid}) as resp:
                    data = await resp.json()

            # 写入结果
            await writer.safe_write([
                row.tid,
                data.get('result', ''),
                data.get('analysis', '')
            ])
        finally:
            progress.update(1)


if __name__ == '__main__':
    # 配置参数
    input_file = "data/input.csv"
    output_file = "data/output.csv"
    
    # 数据校验
    df = pd.read_csv(input_file)
    validate_dataframe(df, required_columns=['tid', 'question'])

    # 运行处理器
    processor = CustomProcessor(input_csv=input_file, output_csv=output_file, concurrency=5)
    processor.run()
    print("数据处理完成！")
