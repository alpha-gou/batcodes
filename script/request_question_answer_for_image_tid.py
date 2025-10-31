import json
import re
import time
import requests
import logging
import base64
import hashlib
import random
from urllib.parse import urlencode

from multi_thread import MultiThreadRequester


TID_INFO_URL = "https://alpha-gou-mock/tkapi/api/getTidInfo?tid={}&type={}&time={}&cooperatorID={}&sign={}"
COOPERATOR_ID = 113
TOKEN = 'alpha-gou-mock'
# URL_TEMPLATE = "https://img.zuoyebang.cc/{}.jpg"
# OEDER_LIST = ["question_ori_1080", "question_ori_ty", "question_2.99", "question_n1", "question_n2"]
OCR_TRANSFORMATION_URL = "http://alpha-gou-mock/image_transformation.php"
QA_URL = "http://alpha-gou-mock/search_preprocess"


def request_answer_question(tid):
    headers = {"Content-Type": "application/json"}
    answer, question = "", ""
    answer_plus, question_plus = 0, 0
    for _ in range(3):  # 重试3次
        try:
            json_data = {
                "tid": int(tid),
                "needTransOcr": 1, "needPicTab": 1,
                "needTidInfo": 1, "needProcessText": 1,
                "needTextFormat": 0, "needOcrFormat": 0
            }
            response = requests.post(QA_URL, headers=headers, json=json_data)
            response_json = response.json()["data"]
            answer = response_json["final_answer"]
            question = response_json["final_question"]
        except Exception as e:
            print("tid {} request_question_answer error with: {}".format(tid, str(e)))
            time.sleep(1)
        else:
            break
    if not answer:
        answer = request_tid_info(tid, "answer")
        answer_plus = 1
        print(tid, "answer_plus", "failed" if answer else "succ")
    if not question:
        question = request_tid_info(tid, "question")
        question_plus = 1
        print(tid, "question_plus", "failed" if question else "succ")
    return answer, question, answer_plus, question_plus


def request_tid_info(tid, target):
    aiana_res, img_url = "", ""
    for _ in range(3):  # 重试3次
        try:
            nowtime = int(time.time())
            get_url = TID_INFO_URL.format(
                    tid, 2, nowtime, COOPERATOR_ID, __generate_sign(tid, TOKEN, nowtime))
            tid_info = requests.get(url=get_url).json()["data"]
        except Exception as e:
            print(e)
            time.sleep(1)
        else:
            if target in tid_info:
                regex = r'<img[^>]*src=["\']?(https?://[^"\'>]+)["\']?'  # 匹配src中的URL
                matches = re.findall(regex, tid_info[target])
                if len(matches) > 0:
                    img_url = matches[0]
            if "analysis" in tid_info:
                try:
                    analysis_data = json.loads(tid_info["analysis"])
                    aiana_res = analysis_data["aiAnalysis"][target]
                except Exception as e:
                    pass
            break
    if not aiana_res:
        aiana_res = __rq_small_ocr(img_url)
    return aiana_res


def __generate_sign(tid, token, timestamp):
    s1 = 'params1={}&params2={}&token={}'.format(tid, timestamp, token)
    s2 = base64.b64encode(s1.encode('utf8')).decode('utf8')
    s3 = hashlib.md5(token.encode('utf8')).hexdigest()
    s4 = s3 + s2
    return hashlib.md5(s4.encode('utf8')).hexdigest()


def __rq_small_ocr(img_url):
    if not img_url: return ""
    try:
        image = requests.get(img_url, timeout=3)
        img_data = image.content
        img_b64_string = base64.b64encode(img_data)
        post_data = {
            "imageData": img_b64_string,
            "sid": random.randint(112123, 142353464654575),
            "source": "cnprint_reco",
            "picFormat": "jpg"
        }
        headers = {
            'Content-type': "application/x-www-form-urlencoded",
            'User-Agent': "PostmanRuntime/7.17.1",
            'Accept': "*/*",
            'Cache-Control': "no-cache",
            'Accept-Encoding': "gzip, deflate",
            'Content-Length': "102248",
            'Connection': "keep-alive",
            'cache-control': "no-cache"
        }
        data = urlencode(post_data)

        response = requests.post(OCR_TRANSFORMATION_URL, data=data, headers=headers, timeout=5)
        # 控制下置信度 prob  < 0.92  可以不用
        text = json.loads(response.text)['ret']
        text_list = text.split("\t")
        if float(text_list[1]) < 0.92:
            return ""
        else:
            return text_list[0]
    except Exception as e:
        print(e)
        return ""


class QARequester(MultiThreadRequester):
    def request_main(self, data):
        """请求的主逻辑，可按需重写此函数"""
        tid = data["tid"]
        data["answer"], data["question"], data["answer_plus"], data["question_plus"] = request_answer_question(tid)
        self._safe_write_csv(data)


if __name__ == "__main__":
    requester = QARequester(api_url="", max_workers=20, rate_limit=5)
    requester.load_csv("mm/modi_total_8633.csv")
    requester.output_to_csv(
            "mm/modi_total_8633_qa_res.csv", add_cols=["answer", "question", "answer_plus", "question_plus"])
    requester.run()
