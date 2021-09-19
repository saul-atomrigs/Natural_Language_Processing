
# TODO: FastText GENSIM

from gensim.models.fasttext import FastText
import numpy as np

texts = [['human', 'interface', 'computer'],
         ['survey', 'user', 'computer', 'system', 'response', 'time'],
         ['eps', 'user', 'interface', 'system'],
         ['system', 'human', 'system', 'eps'],
         ['user', 'response', 'time'],
         ['trees'],
         ['graph', 'trees'],
         ['graph', 'minors', 'trees'],
         ['graph', 'minors', 'survey']]

model = FastText(vector_size=4, window=3, min_count=1, sentences=texts, 
                 epochs=100, bucket=10, min_n=3, max_n=3, sg=1)

# 워드 벡터 확인
model.wv['computer']

# oov라도 다른 벡터를 갖는다
model.wv['comoklksjd']     # 'com' 성분이 포함돼 있다.
model.wv['omplkasjdflkd']  # 'omp' 성분이 포함돼 있다.

# 유사도 확인
model.wv.most_similar('computer', topn = 5)

# 어휘 사전 확인
model.wv.vocab['eps']

# hash table (bucket) 확인. subword들의 워드 벡터가 저장된 공간.
model.wv.vectors_ngrams


# TODO: 자소 단위 fasttext
from hangul_utils import split_syllables, join_jamos
from gensim.models.fasttext import FastText
import numpy as np
import pickle

# 한글 자모 분리/합침 연습
jamo = split_syllables('안녕하세요')
word = join_jamos(jamo)
print(jamo)
print(word)

# Commented out IPython magic to ensure Python compatibility.
# 전처리가 완료된 한글 코퍼스를 읽어온다.
# %cd '/content/drive/MyDrive/Colab Notebooks'
with open('data/konovel_preprocessed.pkl', 'rb') as f:
    sentence_list = pickle.load(f)

sentence_list[0]

# sentence_list를 한글 자모로 분리한다.
sentence_jamo = []
for sentence in sentence_list:
    jamo = []
    for word in sentence:
        jamo.append(split_syllables(word))
    sentence_jamo.append(jamo)

sentence_jamo[0]

model = FastText(size=100, window=5, min_count=10, sentences=sentence_jamo, 
                 max_vocab_size=10000, sample=1e-5, iter=100, bucket=2000000, sg=1, negative=2)

def get_word_vector(model, word):
    jamo = split_syllables(word)
    return model.wv[jamo]

def get_similar_words(model, word, top_n=10):
    jamo = split_syllables(word)
    sim = []
    for (jamo, score) in model.wv.most_similar(jamo, topn = top_n):
        sim.append((join_jamos(jamo), score))
    return sim

# 어휘 사전 확인
model.wv.vocab

get_word_vector(model, '학교')

get_similar_words(model, '바다')
