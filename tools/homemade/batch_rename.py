#!/usr/bin/env python3
import os
import re
import sys
import argparse


def batch_rename_files(directory, pattern, replacement=''):
    """
    批量重命名目录中的文件，删除文件名中匹配正则表达式的部分
    
    Parameters:
    directory (str): 要处理的目录路径
    pattern (str): 正则表达式模式
    replacement (str): 替换匹配内容的字符串，默认为空字符串（即删除）
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
        
        # 第一步：试运行，收集所有重命名操作
        rename_operations = []
        no_change_count = 0
        conflict_count = 0
        
        for filename in files:
            # 使用正则表达式替换文件名中的匹配部分
            new_filename = regex.sub(replacement, filename)
            
            # 如果文件名没有变化
            if filename == new_filename:
                no_change_count += 1
                continue
            
            # 构建完整的文件路径
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            # 检查新文件名是否已存在
            if os.path.exists(new_path):
                print(f"警告：跳过 '{filename}' -> '{new_filename}'（目标文件已存在）")
                conflict_count += 1
                continue
            
            # 记录重命名操作
            rename_operations.append((old_path, new_path, filename, new_filename))
            print(f"重命名: '{filename}' -> '{new_filename}'")
        
        print("-" * 50)
        
        # 如果没有需要重命名的文件
        if not rename_operations:
            print(f"没有需要重命名的文件。")
            print(f"统计: {no_change_count} 个文件无需修改, {conflict_count} 个文件有命名冲突")
            return True
        
        print(f"试运行完成：将重命名 {len(rename_operations)} 个文件")
        print(f"统计: {len(rename_operations)} 个文件将被重命名, {no_change_count} 个文件无需修改, {conflict_count} 个文件有命名冲突")
        
        # 第二步：询问用户是否确认执行
        while True:
            response = input(f"\n是否确认执行以上 {len(rename_operations)} 个重命名操作？(y/n): ").strip().lower()
            
            if response in ['y', 'yes']:
                # 执行重命名操作
                executed_count = 0
                failed_count = 0
                
                for old_path, new_path, old_name, new_name in rename_operations:
                    try:
                        os.rename(old_path, new_path)
                        executed_count += 1
                    except OSError as e:
                        print(f"错误：无法重命名 '{old_name}' -> '{new_name}': {e}")
                        failed_count += 1
                
                print(f"\n重命名完成：成功 {executed_count} 个，失败 {failed_count} 个")
                return True
                
            elif response in ['n', 'no']:
                print("操作已取消。")
                return True
            else:
                print("请输入 y 或 n。")
        
    except KeyboardInterrupt:
        print("\n\n操作被用户中断。")
        return False
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
  
  # 替换匹配内容为指定字符串（而不是删除）
  python batch_rename.py /path/to/directory "old" "new"
        '''
    )

    parser.add_argument('directory', help='要处理的目录路径')
    parser.add_argument('pattern', help='要匹配的正则表达式模式')
    parser.add_argument('replacement', nargs='?', default='', 
                       help='替换匹配内容的字符串（默认：空字符串，即删除）')
    
    args = parser.parse_args()
    
    # 执行批量重命名
    success = batch_rename_files(
        directory=args.directory,
        pattern=args.pattern,
        replacement=args.replacement
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()