from async_data_processor import AsyncDataProcessor, AsyncRequestClient, retry
# from async_data_processor.utils import filter_processed_data
import json
import pandas as pd


class AnswerEquivalenceProcessor(AsyncDataProcessor):
    """答案一致性处理专用类"""
    
    def __init__(self, input_csv, output_csv, concurrency=5):
        super().__init__(input_csv, output_csv, concurrency)
        self.required_columns = ['tid', 'original_script', 'final_script']

    def _get_headers(self):
        return self.required_columns + ["ae_res", "ans_1", "ans_2"]

    async def process_row(self, client, writer, row, progress):
        try:
            # 获取题干信息
            res_data = await self._get_question_info(client, row.tid)
            final_question = res_data.get('data', {}).get('final_question', '')
            final_answer = res_data.get('data', {}).get('final_answer', '')

            # 答案一致性判断
            ae_res = await self._get_text_equivalence(
                client, final_question, final_answer, row.final_script
            )

            new_res = [
                ae_res.get("is_equal", 0),
                ae_res.get("mining_answer_1", ""),
                ae_res.get("mining_answer_2", "")
            ]
            # 安全写入
            await writer.safe_write(list(row.values()), new_res)
        except Exception as e:
            print(f"Error processing tid {row.tid}: {str(e)}")
        finally:
            progress.update(1)

    @retry(max_retries=3)
    async def _get_question_info(self, client, tid):
        """题干信息获取"""
        async with client.session.post(
            'http://10.126.203.6:8405/search_preprocess',
            json={
                "tid": int(tid),
                "needTransOcr": 1, "needPicTab": 1,
                "needTidInfo": 1, "needProcessText": 1,
                "needTextFormat": 0, "needOcrFormat": 0
            }
        ) as resp:
            text = await resp.text()
            return json.loads(text)

    @retry(max_retries=3)
    async def _get_text_equivalence(self, client, stem1, analysis1, script):
        """答案一致性API"""
        async with client.session.post(
            "http://knowledge-aigc-controller.rec-strategy.zuoyebang.dd/get_text_equivalence",
            json={
                "stem1": stem1, "analysis1": analysis1,
                "stem2": stem1, "analysis2": script,
                "courseid": 2, "departid": 0, "categoryid": 0
            }
        ) as resp:
            text = await resp.text()
            return json.loads(text)


if __name__ == '__main__':
    processor = AnswerEquivalenceProcessor(
        input_csv="raw_data/xg_noae_pos_data_v4_18w.csv",
        output_csv="raw_data/xg_ae_res_data_v4_18w.csv",
        concurrency=5
    )
    processor.run()








import requests

KEY_NAME = "xg_res"

# 答案一致性判断
def ask_equal(stem1, analysis1, stem2, analysis2):
    """
    题干 预测结果  题干 预测结果
    """
#     url = "http://172.29.149.161:8406/get_text_equivalence"
    url = "http://knowledge-aigc-controller.rec-strategy.zuoyebang.dd/get_text_equivalence"
    data = {
        "stem1": stem1,
        "analysis1": analysis1,
        "stem2": stem2,
        "analysis2": analysis2,
        "courseid": 2,
        "departid": 0,
        "categoryid": 0,
    }
    response = requests.post(url, json=data)
    return response.text


# 取题干和答案
def request_url(tid):
    url = 'http://10.126.203.6:8405/search_preprocess'
    data = {
        "tid" : int(tid),
        "needTransOcr": 1, # 题目信息中的图片url是否过OCR获取文本信息，1-过,0-不过
        "needPicTab": 1,   # 是否需要保留图表题，1-不判断图表题，0-判断图表题
        "needTidInfo": 1, # 是否请求题库接口，0-不请求，1-请求
        "needProcessText": 1, # 是否过文本预处理，0-否，1-是
        "needTextFormat": 0, # 是否保留文本格式，0-不保留，1-保留
        "needOcrFormat": 0 # 是否保留ocr特型，0-不保留，1-保留
    }
    res = requests.post(url, data = json.dumps(data))
    return res


def get_ask_equal_res(row):
#     res = request_url(row["tid"].split("_")[0])
    res = request_url(row["tid"])
    res = json.loads(res.text)
    final_question = res.get('data', {}).get('final_question', '')
    final_answer = res.get('data', {}).get('final_answer', '')

    cnt = 3
    while cnt > 0:
        text = row[KEY_NAME]
        ae_res = ask_equal(final_question, final_answer, final_question, text)
        ae_res_json = json.loads(ae_res)
        if ae_res_json["mining_answer_1"] and ae_res_json["mining_answer_2"]:
            if ae_res_json["is_equal"]:
                return ae_res
            else:
                break
        else:
            cnt -= 1
    return ae_res


def get_is_equal(text):
    data = json.loads(text)
    return 1 if data["is_equal"] else 0


def get_ans1(text):
    data = json.loads(text)
    return data["mining_answer_1"]


def get_ans2(text):
    data = json.loads(text)
    return data["mining_answer_2"]


def run_ae(df):
    df.loc[:, "ae_res"] = df.apply(get_ask_equal_res, axis=1)
    df.loc[:, "is_equal"] = df["ae_res"].apply(get_is_equal)
    df.loc[:, "ans1"] = df["ae_res"].apply(get_ans1)
    df.loc[:, "ans2"] = df["ae_res"].apply(get_ans2)
    return df



