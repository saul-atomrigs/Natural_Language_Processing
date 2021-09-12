# 한국어-영어 기계번역 교육을 위한 데이터를 생성한다.
# 기계번역 연구를 위해서는 AI Hub (https://aihub.or.kr/aidata/87)에서 제공하는 160만개의 번역 데이터를
# 번역한 후 기계번역 프로그램을 실습하는 것으로 한다.

from time import sleep
from tqdm import tqdm
import re
import pickle
import pandas as pd
import googletrans
# !pip install googletrans == 4.0.0-rc1


# 데이터 파일을 읽어온다.
data_df = pd.read_csv('data/ChatBotData.csv', header=0)
question, answer = list(data_df['Q']), list(data_df['A'])

# 특수 문자를 제거한다.
FILTERS = "([~.,!?\"':;)(])"
question = [re.sub(FILTERS, "", s) for s in question]
answer = [re.sub(FILTERS, "", s) for s in answer]

data_df.head()

# document를 영어로 번역한다.
# currently google translate API allows 5 calls/second, caps it at 200k a day


def translation(source, translator):
    target = []
    for s in tqdm(source):
        try:
            tg = translator.translate(s, dest='en').text
        except:
            print(s)
            tg = ''
        target.append(tg)
        sleep(0.5)
    return target


translator = googletrans.Translator()

# 10개 문장만 번역해 본다
a = translation(question[:10], translator)
a

## 시간 오래 걸림 : 4 ~ 5시간 ##
# question 문장을 번역한다.
qus_target = translation(question, translator)

# answer 문장을 번역한다.
ans_target = translation(answer, translator)

target = qus_target + ans_target

# question과 answer를 합친 후 저장한다.
source = question + answer
target = qus_target + ans_target

if len(source) == len(target):
    print('length ok')

mt_data = pd.DataFrame({'source': source, 'target': target})
mt_data.to_csv('data/machine_trans.csv', index=False)

# 저장 결과를 확인한다.
mt_df = pd.read_csv('data/machine_trans.csv')

mt_df.head(20)
