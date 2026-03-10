#!/usr/bin/env python3
"""
重命名当前目录下指定范围内的数字.md文件，将文件名增加指定增量。
用法: name_plus <增量> <起始数字> <结束数字>
示例: name_plus 1 4 10   # 将4.md到10.md重命名为5.md到11.md
"""

import os
import sys
import argparse
import re


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="重命名当前目录下指定范围内的数字.md文件，文件名增加指定增量。"
    )
    parser.add_argument(
        "increment", type=int,
        help="要增加的数字（可为负数）"
    )
    parser.add_argument(
        "start", type=int,
        help="起始数字（包含）"
    )
    parser.add_argument(
        "end", type=int,
        help="结束数字（包含）"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    inc = args.increment
    start = args.start
    end = args.end

    if start > end:
        print("错误：起始数字不能大于结束数字", file=sys.stderr)
        sys.exit(1)

    # 生成源数字列表（包含start和end）
    src_nums = list(range(start, end + 1))

    # 检查源文件是否存在
    missing = []
    for num in src_nums:
        filename = f"{num}.md"
        if not os.path.isfile(filename):
            missing.append(num)
    if missing:
        print("警告：以下源文件不存在，将被跳过：", missing, file=sys.stderr)

    # 计算目标数字，并构建映射
    mapping = {}  # src_num -> dst_num
    dst_nums = set()
    for num in src_nums:
        dst = num + inc
        # 检查目标文件名是否与现有文件冲突（且该文件不在本次源文件列表中）
        dst_file = f"{dst}.md"
        if os.path.exists(dst_file) and dst not in src_nums:
            print(f"错误：目标文件 {dst_file} 已存在，且不是本次待移动的源文件。操作中止。", file=sys.stderr)
            sys.exit(1)
        mapping[num] = dst
        dst_nums.add(dst)

    # 确定处理顺序：增量>0时从大到小，增量<0时从小到大，避免覆盖未处理的源文件
    sorted_src = sorted(src_nums, reverse=(inc > 0))

    # 预览操作
    print("将执行以下重命名操作：")
    for num in sorted_src:
        if num in missing:
            continue  # 跳过缺失文件
        dst = mapping[num]
        print(f"  {num}.md -> {dst}.md")

    # 询问确认
    if missing:
        print(f"注意：上述列表中已自动跳过不存在的文件：{missing}")
    response = input("\n是否执行以上操作？[y/N] ").strip().lower()
    if response not in ('y', 'yes'):
        print("操作已取消。")
        return

    # 执行重命名
    for num in sorted_src:
        if num in missing:
            continue
        src_file = f"{num}.md"
        dst_file = f"{mapping[num]}.md"
        try:
            os.rename(src_file, dst_file)
            print(f"已重命名: {src_file} -> {dst_file}")
        except OSError as e:
            print(f"重命名 {src_file} 失败: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()