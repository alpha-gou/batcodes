#!/usr/bin/env python3
import os
import re
import sys
import argparse


def batch_rename_files(directory, pattern, replacement='', dry_run=False):
    """
    批量重命名目录中的文件，删除文件名中匹配正则表达式的部分
    
    Parameters:
    directory (str): 要处理的目录路径
    pattern (str): 正则表达式模式
    replacement (str): 替换匹配内容的字符串，默认为空字符串（即删除）
    dry_run (bool): 是否为试运行模式（只显示将要进行的更改，不实际执行）
    """
    try:
        # 检查目录是否存在
        if not os.path.isdir(directory):
            print(f"错误：目录 '{directory}' 不存在")
            return False
        
        # 获取目录中的所有文件（不包括子目录）
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        if not files:
            print(f"目录 '{directory}' 中没有文件")
            return True
        
        print(f"在目录 '{directory}' 中找到 {len(files)} 个文件")
        print(f"正则表达式模式: {pattern}")
        print(f"替换为: '{replacement}'")
        print("-" * 50)
        
        # 编译正则表达式
        try:
            regex = re.compile(pattern)
        except re.error as e:
            print(f"错误：无效的正则表达式 '{pattern}': {e}")
            return False
        
        renamed_count = 0
        skip_count = 0
        
        for filename in files:
            # 使用正则表达式替换文件名中的匹配部分
            new_filename = regex.sub(replacement, filename)
            
            # 如果文件名没有变化，跳过
            if filename == new_filename:
                skip_count += 1
                continue
            
            # 构建完整的文件路径
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            # 检查新文件名是否已存在
            if os.path.exists(new_path):
                print(f"警告：跳过 '{filename}' -> '{new_filename}'（目标文件已存在）")
                skip_count += 1
                continue
            
            if dry_run:
                print(f"[试运行] '{filename}' -> '{new_filename}'")
            else:
                try:
                    os.rename(old_path, new_path)
                    print(f"重命名: '{filename}' -> '{new_filename}'")
                    renamed_count += 1
                except OSError as e:
                    print(f"错误：无法重命名 '{filename}': {e}")
                    skip_count += 1
        
        print("-" * 50)
        if dry_run:
            print(f"试运行完成：将重命名 {renamed_count} 个文件，跳过 {skip_count} 个文件")
        else:
            print(f"完成：成功重命名 {renamed_count} 个文件，跳过 {skip_count} 个文件")
        
        return True
        
    except Exception as e:
        print(f"发生错误: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='批量重命名文件，删除文件名中匹配正则表达式的部分',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 删除文件名中的所有数字
  python batch_rename.py /path/to/directory "\\\\d+"
  
  # 删除文件名中的方括号及其内容
  python batch_rename.py /path/to/directory "\\\\[.*?\\\\]"
  
  # 删除文件名开头的日期格式（如2023-01-01_）
  python batch_rename.py /path/to/directory "^\\\\d{4}-\\\\d{2}-\\\\d{2}_"
  
  # 试运行模式（只显示将要进行的更改）
  python batch_rename.py /path/to/directory "pattern" --dry-run
  
  # 替换匹配内容为指定字符串（而不是删除）
  python batch_rename.py /path/to/directory "old" "new"
        '''
    )

    parser.add_argument('directory', help='要处理的目录路径')
    parser.add_argument('pattern', help='要匹配的正则表达式模式')
    parser.add_argument('replacement', nargs='?', default='', 
                       help='替换匹配内容的字符串（默认：空字符串，即删除）')
    parser.add_argument('--dry-run', action='store_true', 
                       help='试运行模式，只显示将要进行的更改，不实际执行')
    args = parser.parse_args()
    
    # 执行批量重命名
    success = batch_rename_files(
        directory=args.directory,
        pattern=args.pattern,
        replacement=args.replacement,
        dry_run=args.dry_run
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()