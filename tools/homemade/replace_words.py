#!/usr/bin/env python3
"""
批量替换 .md 文件中的文本内容
用法: python replace_words.py <目录> <原文本> <新文本> [--regex]
"""
import os, re, sys
from pathlib import Path


# ANSI 颜色代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'


def highlight_differences(old_line, new_line, old_text, new_text, use_regex=False):
    """高亮显示新旧文本的差异"""
    if not old_line or not new_line:
        return old_line, new_line
    
    # 普通文本替换的高亮
    if not use_regex:
        # 高亮旧文本
        highlighted_old = old_line
        if old_text in old_line:
            highlighted_old = old_line.replace(old_text, f"{RED}{old_text}{RESET}")
        
        # 高亮新文本
        highlighted_new = new_line
        if new_text in new_line:
            highlighted_new = new_line.replace(new_text, f"{GREEN}{new_text}{RESET}")
        
        return highlighted_old, highlighted_new
    else:
        # 正则表达式替换的高亮
        try:
            pattern = re.compile(old_text)
            
            # 高亮旧行中的匹配部分
            matches = list(pattern.finditer(old_line))
            if matches:
                parts = []
                last_end = 0
                for match in matches:
                    # 添加非匹配部分
                    parts.append(old_line[last_end:match.start()])
                    # 添加高亮的匹配部分
                    parts.append(f"{RED}{match.group()}{RESET}")
                    last_end = match.end()
                # 添加剩余部分
                parts.append(old_line[last_end:])
                highlighted_old = ''.join(parts)
            else:
                highlighted_old = old_line
            
            # 对于新行，高亮替换的部分
            highlighted_new = new_line
            
            # 尝试高亮新行中新添加的文本
            # 这里简化处理：高亮整个新行中被替换的部分
            # 但更准确的方法是使用正则替换的逆向匹配
            if new_text != old_text:
                # 检查新行中是否包含新文本
                if new_text in new_line:
                    highlighted_new = new_line.replace(new_text, f"{GREEN}{new_text}{RESET}")
            
            return highlighted_old, highlighted_new
        except:
            pass
    
    return old_line, new_line


def find_md_files_with_matches(directory, old_text, new_text, use_regex=False):
    """查找所有 .md 文件中匹配的内容"""
    dir_path = Path(directory)
    matches = {}
    
    for md in dir_path.rglob("*.md"):
        try:
            with open(md, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            changes = []
            for i, line in enumerate(lines, 1):
                if (use_regex and re.search(old_text, line)) or (not use_regex and old_text in line):
                    new_line = re.sub(old_text, new_text, line) if use_regex else line.replace(old_text, new_text)
                    changes.append((i, line.rstrip(), new_line.rstrip()))
            
            if changes:
                matches[str(md)] = changes
        except:
            continue
    
    return matches

def preview_changes(matches, base_dir, old_text, new_text, use_regex=False):
    """预览更改内容，用不同颜色高亮差异"""
    if not matches:
        print("没有找到匹配的内容")
        return False
    
    print(f"\n{BOLD}{YELLOW}🔍 找到 {len(matches)} 个文件需要修改{RESET}")
    print("=" * 80)
    
    for fpath, changes in matches.items():
        rel_path = Path(fpath).relative_to(base_dir)
        print(f"\n{BOLD}{CYAN}📄 文件: {rel_path}{RESET}")
        print(f"{BLUE}{'─' * 40}{RESET}")
        
        for i, (ln, old_line, new_line) in enumerate(changes[:3]):
            # 高亮差异
            highlighted_old, highlighted_new = highlight_differences(
                old_line, new_line, old_text, new_text, use_regex
            )
            
            print(f"{BOLD}   行 {ln}:{RESET}")
            print(f"     - 原: {highlighted_old}")
            print(f"     + 新: {highlighted_new}")
            if i < len(changes[:3]) - 1:
                print()  # 在多个更改之间添加空行
        
        if len(changes) > 3:
            print(f"\n    {YELLOW}... 还有 {len(changes)-3} 处更改{RESET}")
    
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    return True

def apply_changes(matches, old_text, new_text, use_regex=False):
    """应用更改"""
    count = 0
    for fpath, changes in matches.items():
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            new_content = re.sub(old_text, new_text, content) if use_regex else content.replace(old_text, new_text)
            
            if new_content != content:
                with open(fpath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                count += 1
                print(f"{GREEN}✓ 已修改: {Path(fpath).name}{RESET}")
        except Exception as e:
            print(f"{RED}✗ 处理失败 {Path(fpath).name}: {e}{RESET}")
    
    return count


def main():
    if len(sys.argv) < 4:
        print(f"{BOLD}{CYAN}批量替换 .md 文件中的文本内容{RESET}")
        print(f"\n{BOLD}用法:{RESET} python replace_words.py <目录> <原文本> <新文本> [--regex]")
        print(f"{BOLD}示例:{RESET}")
        print(f"  python replace_words.py ./docs '旧内容' '新内容'")
        print(f"  python replace_words.py ./docs 'foo\\d+' 'bar' -r")
        print(f"\n{BOLD}颜色说明:{RESET}")
        print(f"  {RED}红色{RESET}: 将被替换的文本")
        print(f"  {GREEN}绿色{RESET}: 替换后的文本")
        return
    
    dir_path = Path(sys.argv[1])
    old_text = sys.argv[2]
    new_text = sys.argv[3]
    use_regex = '-r' in sys.argv

    if not dir_path.exists():
        print(f"{RED}错误: 目录不存在: {dir_path}{RESET}")
        return
    
    print(f"{BOLD}{CYAN}🔍 正在扫描目录: {dir_path} 中的 .md 文件...{RESET}")
    
    # 查找匹配
    matches = find_md_files_with_matches(dir_path, old_text, new_text, use_regex)
    
    if not matches:
        print(f"{YELLOW}没有找到匹配的内容{RESET}")
        return
    
    # 预览
    if not preview_changes(matches, dir_path, old_text, new_text, use_regex):
        return
    
    # 确认
    print(f"\n{BOLD}{YELLOW}⚠️  是否确认执行替换？(y/n){RESET}")
    choice = input(f"{BOLD}请选择: {RESET}").strip().lower()
    
    if choice == 'y' or choice == 'yes':
        print(f"\n{BOLD}{CYAN}🔄 正在执行替换...{RESET}")
        count = apply_changes(matches, old_text, new_text, use_regex)
        print(f"\n{BOLD}{GREEN}✅ 完成！成功修改了 {count} 个文件{RESET}")
    else:
        print(f"{RED}❌ 操作已取消{RESET}")


if __name__ == "__main__":
    main()