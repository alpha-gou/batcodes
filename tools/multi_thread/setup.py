# setup.py 使用 pip install -e . 命令实现实时更新
from setuptools import setup, find_packages

setup(
    name="multi_thread",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'asyncio',
        'aiohttp',
        'aiofiles',
        'tqdm',
        'pandas',
    ],
    python_requires=">=3.8",
)