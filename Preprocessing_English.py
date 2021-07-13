
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import PorterStemmer
import collections
import pickle

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('your_data.tsv', header=0, sep='\t', quoting=3)

# Pre-processing
stemmer = PorterStemmer()
stopwords = nltk.corpus.stopwords.words('english')

clean_text = []
for review in df['column_name']:
    # 1. 영문자와 숫자만 사용한다. 그 이외의 문자는 공백 문자로 대체한다.
    review = review.replace('<br />', ' ')       # <br> --> space
    review = review.replace('\'', '')            # dont't --> dont
    review = re.sub("[^a-zA-Z]", ' ', review)    # 영문자만 사용

    tmp = []
    for word in nltk.word_tokenize(review):
        # 2. 불용어 처리
        if len(word.lower()) > 1 and word.lower() not in stopwords:
            # 3. Stemming
            tmp.append(stemmer.stem(word.lower()))
    clean_text.append(' '.join(tmp))

clean_text[1]

# 어휘 사전을 생성한다.
vocab = collections.Counter()
for review in clean_text:
    for word in nltk.word_tokenize(review):
        vocab[word] += 1

# 빈도가 높은 순서로 max_vocab개로 어휘 사전을 생성한다.
max_vocab = 20000
word2idx = {w:(i+2) for i,(w,_) in enumerate(vocab.most_common(max_vocab))}
word2idx["<PAD>"] = 0   
word2idx["<OOV>"] = 1

# review 문장을 word2idx의 인덱스로 표시한다.
x_idx = []
for review in clean_text:
    tmp = []
    for word in nltk.word_tokenize(review):
        if word in word2idx:
            tmp.append(word2idx[word])
        # else:
        #     tmp.append(word2idx['<OOV>'])
    x_idx.append(tmp)

# 학습 데이터를 저장해 둔다.
with open('data/popcorn.pkl', 'wb') as f:
    pickle.dump([clean_text, x_idx, list(df['sentiment']), word2idx], f, pickle.DEFAULT_PROTOCOL)

print(x_idx[0])

print(clean_text[0])
