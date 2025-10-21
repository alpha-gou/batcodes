import requests
import json
import re
import datetime
import time

from .const import DATASTACK_URL


def get_dq_data(group_name, queue_name, max_num):
    """
    获取数据队列中相关数据, 从队列头部出数据
    """
    # 参数校验
    if type(group_name) != str or type(queue_name) != str:
        return -1, "参数非法"
    group_name = group_name.strip()
    queue_name = queue_name.strip()
    if len(group_name) == 0 or len(queue_name) == 0:
        return -1, "参数非法"

    url = DATASTACK_URL + "/dcflow-dashboard/aigcZiyanData/pop"
    data = {
        "sys": group_name,
        "name": queue_name,
        "n": max_num,
        "parseData": True,
    }
    response = requests.get(url, params=data)
    response = json.loads(response.text)
    if "data" not in response or response["data"] is None:
        return []
    return response["data"] 


def write_dq_data(group_name, queue_name, data, write_to_head=True):
    """
    写入数据到数据队列中, data一定要是字符串数组["", "", ""]
    write_to_head True时写入到队列头部 False时写入到队列尾部
    返回示例:{"errNo":0,"errMsg":"succ","data":{"ids":[1,30001,60001,60002],"len":4}}
    """
    # 参数校验
    if type(group_name) != str or type(queue_name) != str:
        return {"errNo": -1, "errMsg": "参数非法"}
    group_name = group_name.strip()
    queue_name = queue_name.strip()
    if len(group_name) == 0 or len(queue_name) == 0:
        return {"errNo": -1, "errMsg": "参数非法"}

    url = DATASTACK_URL + "/dcflow-dashboard/aigcZiyanData/pushBatch"
    data = {
        "sys": group_name,
        "name": queue_name,
        "dataArr": data,
        "toHead": write_to_head,
    }
    response = requests.post(url, json=data)
    response = json.loads(response.text)
    return response
    # return True if response["errNo"] == 0 else False


def create_dq(group_name, queue_name, queue_comment="default", creator="username"):
    """
    创建数据队列
    @params group_name 队列所属的组
            queue_name 队列名称
            queueu_comment 队列说明
            creator 队列创建者
    @return 是否成功
    """
    # 参数校验
    if type(group_name) != str or type(queue_name) != str:
        return -1, "参数非法"
    group_name = group_name.strip()
    queue_name = queue_name.strip()
    if len(group_name) == 0 or len(queue_name) == 0:
        return -1, "参数非法"

    # 封装请求内容
    url = DATASTACK_URL + "/dcflow-dashboard/dcQue/createQue"
    try:
        data = {
            "sys": group_name,
            "name": queue_name,
            "creator": creator,
            "comment": queue_comment,
        }
        response = requests.get(url, params=data)
        response = json.loads(response.text)

        return response["data"]["queId"], response["errMsg"]
    except:
        return -1, "请求创建队列接口失败"


def delete_dq(group_name, queue_name):
    """
    删除数据队列
    @params group_name 队列所属的组
            queue_name 队列名称
    @return 是否成功
    """
    if get_dq_len(group_name, queue_name) > 0:
        print("队列不为空, 不能删除")
        return False

    # 参数校验
    if type(group_name) != str or type(queue_name) != str:
        return False, "参数非法"
    group_name = group_name.strip()
    queue_name = queue_name.strip()
    if len(group_name) == 0 or len(queue_name) == 0:
        return False, "参数非法"

    url = DATASTACK_URL + "/dcflow-dashboard/dcQue/delQue"
    try:
        data = {
            "sys": group_name,
            "name": queue_name,
        }
        response = requests.get(url, params=data)
        response = json.loads(response.text)

        return True if response["errNo"] == 0 else False, response["errMsg"]
    except:
        return False, "请求删除队列接口出错"


def trim_dq(group_name, queue_name, trim_num):
    """
    删除数据队列尾部数据
    @params group_name 队列所属的组
            queue_name 队列名称
            trim_num 保留最新的n条数据
    @return 是否成功
    """
    # 参数校验
    if type(group_name) != str or type(queue_name) != str:
        return False, "参数非法"
    group_name = group_name.strip()
    queue_name = queue_name.strip()
    if len(group_name) == 0 or len(queue_name) == 0:
        return False, "参数非法"

    url = DATASTACK_URL + "/dcflow-dashboard/dcQue/trimTail"
    try:
        data = {
            "sys": group_name,
            "name": queue_name,
            "trimReserveNum": trim_num,
        }
        response = requests.get(url, params=data)
        response = json.loads(response.text)

        return True if response["errNo"] == 0 else False, response["errMsg"]
    except:
        return False, "请求trim队列接口出错"


def get_all_dq(group, creator="username"):
    """
    查看所有数据队列信息, 非实时
    """
    url = DATASTACK_URL + "/dcflow-dashboard/dcQue/feGetQues"
    data = {
        "sys": group,
        "deleted": 0,
        "creator": creator,
    }
    response = requests.get(url, params=data)
    response = json.loads(response.text)
    return response["data"]


def get_dq_len(group_name, queue_name):
    """
    查看数据队列当前积压数据量
    """
    url = DATASTACK_URL + "/dcflow-dashboard/dcQue/len"
    data = {
        "sys": group_name,
        "name": queue_name,
    }
    response = requests.get(url, params=data)
    response = json.loads(response.text)
    return response["data"]["len"] if response["errNo"] == 0 else -1


def get_all_dq_len(group_name):
    """
    查看所有数据队列数量, 实时
    """
    data = get_all_dq(group_name)
    result = []
    for item in data:
        queue_name = item["name"]
        queue_len = get_dq_len(group_name, queue_name)
        result.append([group_name, queue_name, queue_len])
    result = sorted(result, key=lambda x:x[1])
    return result


def clean_dq(group_name, queue_name):
    """
    清空队列数据
    """
    url = DATASTACK_URL + "/dcflow-dashboard/dcQue/clean"
    data = {
        "sys": group_name,
        "name": queue_name,
    }
    response = requests.get(url, params=data)
    response = json.loads(response.text)
    return True if response["errNo"] == 0 else False


def get_lock(key):
    """
    获取key锁
    """
    key = str(key)
    url = DATASTACK_URL + "/dcflow-dashboard/aigcZiyanData/lock"
    data = {
        "sid": key,
        "expire": 3,
    }
    response = requests.post(url, json=data)
    response = json.loads(response.text)
    if "data" in response and response["data"] == True:
        return True
    return False


def release_lock(key):
    """
    释放key锁
    """
    key = str(key)
    url = DATASTACK_URL + "/dcflow-dashboard/aigcZiyanData/lock"
    data = {
        "sid": key,
        "unlock": True,
    }
    response = requests.post(url, json=data)
    response = json.loads(response.text)
    if "data" in response and response["data"] == True:
        return True
    return False


def monitor():
    monitor_list = [
        [["aigc_k12", "english_other@out1024"], ["aigc_k12", "english_other@out1536"]],
    ]

    index = 0
    while True:
        group_len_list = []
        for group_list in monitor_list:
            group_len = 0
            for queue in group_list:
                group_len += get_dq_len(*queue)
            group_len_list.append(group_len)
        if 0 in group_len_list:
            print("[{}] {}".format(datetime.datetime.now(), "空了"))
        if index % 12 == 0:
            print(get_all_dq_len("aigc_k12"))
            print(get_all_dq_len("aigc_k12_zyp"))
            print()
        time.sleep(5)
        index += 1

if __name__ == "__main__":
    # 新框架中控队列
    create_dq("aigc_aivideo_queue", "main_prim_math_compute", "AI视频-计算题主队列", "alpha_gou")
    create_dq("aigc_aivideo_queue", "main_prim_math_algebra", "AI视频-泛计算主队列", "alpha_gou")
    create_dq("aigc_aivideo_queue", "main_prim_math_rewrite", "AI视频-改写主队列", "alpha_gou")
    create_dq("aigc_aivideo_queue", "main_prim_math_rewrite_test", "AI视频-改写主队列测试队列", "alpha_gou")
    create_dq("aigc_aivideo_queue", "main_prim_math_control", "AI视频-小数中控主队列", "alpha_gou")
    create_dq("aigc_aivideo_queue", "main_math_type", "AI视频-小数题型分类队列", "alpha_gou")
    create_dq("aigc_aivideo_queue", "main_junior_math_compute", "AI视频-初数计算题主队列", "alpha_gou")
    create_dq("aigc_aivideo_queue", "main_junior_math_algebra", "AI视频-初数泛计算主队列", "alpha_gou")

    # 改写相关队列
    create_dq("aigc_aivideo_queue", "rewrite_main_ctrl", "AI视频-小数改写中控队列", "alpha_gou")
    create_dq("aigc_aivideo_queue", "rewrite_main_ctrl_test", "AI视频-小数改写中控测试队列", "alpha_gou")
    create_dq("aigc_aivideo_queue", "math_xgv6_rewrite", "小数视频免人工生产改写模型队列", "alpha_gou")
    create_dq("aigc_aivideo_queue", "math_qwen_scoring", "小数视频免人工生产评分模型队列", "alpha_gou")
    create_dq("aigc_aivideo_queue", "math_ship_result", "小数视频ship环境测试结果队列", "alpha_gou")
    create_dq("aigc_aivideo_queue", "math_debug_result", "小数测试结果回收队列", "alpha_gou")

    # 语文队列
    create_dq("aigc_aivideo_queue", "chinese_jx_main", "语文视频解析生产队列", "alpha_gou")
    create_dq("aigc_aivideo_queue", "chinese_debug_result", "语文视频解析测试队列", "alpha_gou")

