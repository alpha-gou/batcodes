import json
import re
import time
import requests
from multi_thread import MultiThreadRequester
from request_ae_plus import request_answer_equal_qwq_v2, request_answer_equal_qwq


QA_URL = "http://alpha-gou-mock/search_preprocess"
AE_URL = "http://alpha-gou-mock/get_text_equivalence"
AE_URL_v2 = "http://alpha-gou-mock/get_text_equivalence_v2"


def request_answer_question(tid):
    headers = {"Content-Type": "application/json"}
    answer, question = "", ""
    for _ in range(3):  # 重试3次
        try:
            json_data = {
                # "tid": int(tid),
                "tid": tid,
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
    return answer, question


def request_answer_equal_v1(question, answer, text):
    headers = {"Content-Type": "application/json"}
    try:
        json_data = {
            "stem1": question, "analysis1": answer,
            "stem2": question, "analysis2": text,
            "courseid": 2, "departid": 0, "categoryid": 0
        }
        response = requests.post(AE_URL, headers=headers, json=json_data)
        ae_res = json.loads(response.text)
        return 1 if ae_res["is_equal"] else 0
    except Exception as e:
        print("request_answer_equal_v1 error with: {}".format(str(e)))
        return -1


def request_answer_equal_v2(tid, question, answer, text):
    json_data = {
        "logid": str(tid),
        "stem1": question,
        "analysis1": answer,
        "analysis2": text,
        "product": "k12_aianalysis"
    }
    headers = {'Content-Type': "application/json"}
    try:
        ret = requests.post(url=AE_URL_v2, json=json_data, headers=headers)
        ret = ret.json()
        return ret["judge"]
    except Exception as e:
        print("request_answer_equal_v2 error with: {}".format(str(e)))
        return -1


def fjs_mid_script_to_script(mid_script, question):
    res = {
        "镜头1": {
            "内容类型": "读题",
            "屏幕内容": question,
            "旁白内容": question,
        }
    }
    data = json.loads(mid_script)
    sc_cnt = 1
    for screen_list in data["解答"]:
         for screen in screen_list:
            if screen["类型"] in {"解答", "总结"}:
                sc_cnt += 1
                res["镜头%s" % sc_cnt] = {
                    "内容类型": screen["类型"],
                    "屏幕内容": screen["屏幕"],
                    "旁白内容": screen["旁白"],
                }
    return json.dumps(res, ensure_ascii=False)


class AERequester(MultiThreadRequester):
    def request_main(self, data):
        """请求的主逻辑，可按需重写此函数"""
        tid = int(data["tid"])
        data["tid"] = tid
        script = data["script"]
        data["ans"], data["ques"] = "", ""
        # data["ae_res_1"] = -1
        # data["ae_res_2"] = -1
        # data["ae_qwq"] = -1
        # data["thinking"], data["result"] = "", ""
        # if data["err_code"] == -1:
        if True:
            answer, question = request_answer_question(tid)
            data["ans"], data["ques"] = answer, question

            # answer, question = data["ans"], data["ques"]

            # if data["model"] == "泛计算":
            #     script = fjs_mid_script_to_script(script, question)
            data["ae_res_1"] = request_answer_equal_v1(question, answer, script)

            # 剪枝
            # if data["ae_res_1"] != 1:
            #     data["ae_res_2"] = request_answer_equal_v2(tid, question, answer, script)
            #     if data["ae_res_2"] != 1:
            #         data["ae_qwq"] = request_answer_equal_qwq(script, answer)

            # 不剪枝
            data["ae_res_2"] = request_answer_equal_v2(tid, question, answer, script)
            # data["thinking"], data["result"], data["ae_qwq"] = request_answer_equal_qwq_v2(script, answer)

        # data["EQ"] = 1 if (data["ae_res_1"] == 1) or (data["ae_res_2"] == 1) or (data["ae_qwq"] == 1) else 0
        data["EQ"] = 1 if (data["ae_res_1"] == 1) or (data["ae_res_2"] == 1) else 0
        self._safe_write_csv(data)


if __name__ == "__main__":
    requester = AERequester(api_url="", max_workers=20, rate_limit=20)
    requester.load_csv("debug_badcase_info.csv")
    requester.output_to_csv("debug_badcase_info_ae_res.csv",
                            # add_cols=["ans", "ques", "ae_res_1", "ae_res_2", "ae_qwq", "EQ"])
                            # add_cols=["ans", "ques", "ae_res_1", "ae_res_2", "ae_qwq", "EQ", "thinking", "result"])
                            add_cols=["ans", "ques", "ae_res_1", "ae_res_2", "EQ"])
                            # add_cols=["ans", "ques"])
                            # add_cols=["ae_res_1", "ae_res_2"])
    requester.run()
