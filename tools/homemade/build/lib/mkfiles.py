import os
import sys


def create_integer_files(a, b, format_str="md"):
    """
    在当前目录下创建从a到b之间每个整数命名的文件
    
    参数:
    a -- 起始整数
    b -- 结束整数
    """
    # 确保a和b是整数且a <= b
    try:
        a = int(a)
        b = int(b)
    except ValueError:
        print("错误：请输入有效的整数")
        return False
    
    if a > b:
        a, b = b, a  # 交换a和b，确保a <= b
    
    # 创建文件
    created_count = 0
    for i in range(a, b + 1):
        filename = f"{i}.{format_str}"
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(f"第{i} \n")
            print(f"已创建文件: {filename}")
            created_count += 1
        except Exception as e:
            print(f"创建文件 {filename} 时出错: {e}")
    
    print(f"\n成功创建了 {created_count} 个文件")
    return True


def main():
    """主函数，处理用户输入并执行文件创建"""
    if len(sys.argv) == 3:
        # 从命令行参数获取a和b
        a, b = sys.argv[1], sys.argv[2]
        format_str = "md"
    elif len(sys.argv) == 4:
        # 从命令行参数获取a和b
        a, b, format_str = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        # 交互式输入
        try:
            a = input("请输入起始整数 a: ")
            b = input("请输入结束整数 b: ")
        except KeyboardInterrupt:
            print("\n用户取消操作")
            return
    
    # 执行文件创建
    create_integer_files(a, b, format_str)


if __name__ == "__main__":
    main()