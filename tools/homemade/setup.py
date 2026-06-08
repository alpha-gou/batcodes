"""
安装命令： pip install -e .
卸载历史版本： pip uninstall homemade-tools
"""
from setuptools import setup

setup(
    name='homemade-tools',  # 你工具的名称
    version='0.1.0',
    py_modules=['mkfiles', 'batch_rename', 'nameplus', 'replace_words'],  # 如果你的代码是 func.py，就在这里写明
    install_requires=[],     # 可以在这里填入你的脚本依赖的其他库，例如 ['requests', 'numpy']
    entry_points={
        'console_scripts': [
            'mkfs=mkfiles:main',        # 关键配置！'func'是命令名，'func:main'指向你的函数
            'bfrename=batch_rename:main',   # batch file rename
            'nameplus=nameplus:main',
            'replacewd=replace_words:main',
        ],
    },
)

