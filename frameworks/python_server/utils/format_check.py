import json
import re
import sys


LEGAL_MARK_SET = {"@START@","@END@","@BR@","@sc@","@ec@"}
MARK_MAP = {"@START@": "@END@", "@sc@": "@ec@"}
END_MARK = {"@END@", "@ec@"}
CONTENT_MAX_LEN = 300     # 屏幕或旁白最大的字符数
FORMULA_CH_MAX_LEN = 45   # 屏幕公式最大汉字数
ERROR_CODE_DICT = {
    0: "",
    10: "脚本json格式非法", 11: "出现非法latex字符", 12: "屏幕内容标记不符合括号原则", 
    # latex 公式格式问题
    21: "公式边界问题", 22: "公式中存在非法字符", 23: "余数单位问题", 24: "出现多个约等号", 25: "公式外含有非法字符",
    # 内容问题
    31: "屏幕内容和旁白内容为空", 32: "屏幕内容或旁白内容重复", 33: "无效导图",
    # 字数问题
    41: "屏幕或旁白内容字数超出限制", 42: "导图内容过多", 43: "公式字数超出限制",
    # 超纲问题
    51: "分数超纲", 52: "负数超纲", 53: "超纲词汇",
}


def is_num(text):
    try:
        text = float(text)
        return True
    except:
        return False


def bad_format(text):
    # 公式判断
    if text.count('$') % 2 != 0:
        return 21
    pattern = r'\$(.*?)\$'    
    matches = re.findall(pattern, text) 
    for match in matches:
        # 检查每个匹配的内容中是否包含bad字符
        if '@' in match:
            return 22

        match_content = match.replace(" ", "")
        # 余数如果带单位，前面必须也带
        if "\\cdots\\cdots" in match_content:
            substr1 = match_content.split("\\cdots\\cdots")
            if len(substr1) == 2:
                head = substr1[0].split("=")[-1]
                tail = substr1[1]
                if is_num(head) and not is_num(tail):
                    return 23
            else:
                return 23

        # 一个式子里存在两个以上的约等号
        approx_list = re.findall(r'\\approx', text)
        if len(approx_list) > 1:
            return 24
        
        # 公式内容字数超出限制
        formula_ch = extract_chinese(match_content)
        if len(formula_ch) > FORMULA_CH_MAX_LEN:
            return 43

    # 公式外判断
    text = re.sub("\$.*?\$", "", text)
    no_format_patterns = [  
        r'<[^>]+(?=>)',  # 匹配未闭合的标签，例如 <div  
        r'<[^>]+>',      # 匹配完整的标签，但可能属性不完整  
        r'&[^;]+;',      # 匹配HTML实体，例如 &nbsp;
        # r'\\',
    ] 
    for pattern in no_format_patterns:  
        if re.search(pattern, text):  
            return 25
    return 0


def brackets_not_balanced(text):
    """
    正则表达式匹配 @BR@ 和 @START@ @END@ 是否遵循括号匹配原则
    """
    pattern = r'@.*?@'
    matches = re.findall(pattern, text)
    pre = ""
    for match in matches:
        if match not in LEGAL_MARK_SET:
            return True
        if not pre:
            if match in END_MARK:
                return True
            if match != "@BR@":
                pre = MARK_MAP[match]
        else:
            if pre == match:
                pre = ""
            else:
                return True
    return pre


def above_level(text, data, answer):
    text = text.replace(" ", "")
    # 超纲判断：存在负数
    if re.search(r"(=|\(|（)-\d", text):
        return 52

    # 超纲判断：存在分数或小数
    if ("frac{" in text) or has_decimal(text):
        jt1_text = data["镜头1"]["屏幕内容"].replace(" ", "") \
                + data["镜头1"]["旁白内容"].replace(" ", "") \
                + answer.replace(" ", "")
        # 题干和答案中不含分数、比值、百分数、小数
        if not (
            re.search(r"(%|\d:\d|frac\{|分之|小数点|比例|体积|面积|真分数|假分数)", jt1_text)
                or has_decimal(jt1_text)
        ):
            return 51

    # 超纲词汇
    badwords_pattern = '|'.join([r"负\d+", "真命题", "假命题", "独立事件", "矩阵", "全等形"])
    if re.search(badwords_pattern, text):
        return 53
    return 0


def has_decimal(s):
    pattern = r'\d+\.\d+'  # 仅匹配"数字+小数点+数字"的格式
    return bool(re.search(pattern, s))


def extract_chinese(text):
    """提取字符串中所有中文字符"""
    return ''.join(re.findall(r'[\u4e00-\u9fff]+', text))


def main_check(text, answer=""):
    err_code = 0
    data = json.loads(text)
    dt_str = ""
    main_content = ""  # 屏幕和旁白内容，用于判断超纲

    if "@LATEX" in text:
        return 11

    content1_pre = ""
    content2_pre = ""
    dt_cnt = 0
    for jt in data:
        if data[jt]["内容类型"] == "导图":
            dt_str += data[jt]["屏幕内容"]
            dt_cnt += 1
        content1 = data[jt]["屏幕内容"].replace(" ", "")
        content2 = data[jt]["旁白内容"].replace(" ", "")
        content1 = re.sub("@.*?@", "", content1)
        content2 = re.sub("#.*?#", "", content2)
        main_content = main_content + "#" + content1 + "#" + content2

        # 屏幕内容标记符合括号原则
        if brackets_not_balanced(data[jt]["屏幕内容"]):
            return 12

        # 屏幕中公式内容规则判断
        err_code =  bad_format(data[jt]["屏幕内容"])
        if err_code != 0:
            return err_code

        # 屏幕内容和旁白内容为空
        if not data[jt]["屏幕内容"]:
            return 31
        if not data[jt]["旁白内容"]:
            if data[jt]["内容类型"] != "导图":
                return 31

        # 屏幕内容或旁白内容重复
        if (content1 == content1_pre) or (content2 == content2_pre):
            return 32
        content1_pre, content2_pre = content1, content2

        # 屏幕和旁白内容字数超出限制
        if len(content1) > CONTENT_MAX_LEN or len(content2) > CONTENT_MAX_LEN:
            return 41

    # 导图内容过多
    if len(dt_str) > 100:
        return 42

    # 无效导图
    if dt_cnt == 1:
        return 33

    err_code = above_level(main_content, data, answer)
    return err_code


def get_error_code(text, answer):
    try:
        err_code = main_check(text, answer)
        return err_code
    except:
        return 10


def get_error_info(err_code):
    return ERROR_CODE_DICT.get(err_code, "未知错误类型")


def get_error_code_and_info(text, answer):
    try:
        err_code = main_check(text, answer)
        return err_code, ERROR_CODE_DICT[err_code]
    except:
        return 10, "脚本json格式非法"

