import numpy as np
import re
import pickle
from sklearn.datasets import fetch_20newsgroups
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# news data를 읽어온다.
newsData = fetch_20newsgroups(
    shuffle=True, random_state=1, remove=('footers', 'quotes'))

news = newsData['data']
topic = newsData['target']
n_topic = len(set(topic))

# Subject만 추출한다.
subjects = []
y_topic = []
for text, top in zip(news, topic):
    for sent in text.split('\n'):
        idx = sent.find('Subject:')
        if idx >= 0:       # found
            subject = sent[(idx + 9):].replace('Re: ', '').lower()
            subject = re.sub("[^a-zA-Z]", " ", subject)
            if len(subject.split()) > 3:  # subject가 3단어 이상인 것만 허용한다.
                subjects.append(subject)
                y_topic.append(top)
            break

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', cache_dir='bert_eng_ckpt', do_lower_case=False)

enc = tokenizer.encode('i love you'.split())
dec = tokenizer.decode(enc)
print(enc)
print(dec)

word2idx = tokenizer.vocab
idx2word = {v: k for k, v in word2idx.items()}
[idx2word[x] for x in enc]

# word index로 변환한다.
subject_idx = [tokenizer.encode(s.split()) for s in subjects]
print(subject_idx[:10])

subject_len = [len(x) for x in subject_idx]
print('max = ', np.max(subject_len))
sns.displot(subject_len)

MAX_LEN = 15

# Bert Tokenizer
# 참조: https://huggingface.co/transformers/main_classes/tokenizer.html?highlight=encode_plus#transformers.PreTrainedTokenizer.encode_plus


def bert_tokenizer(sent):

    encoded_dict = tokenizer.encode_plus(
        text=sent,
        add_special_tokens=True,      # Add '[CLS]' and '[SEP]'
        max_length=MAX_LEN,           # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True    # Construct attn. masks.
    )

    input_id = encoded_dict['input_ids']
    # And its attention mask (simply differentiates padding from non-padding).
    attention_mask = encoded_dict['attention_mask']
    # differentiate two sentences
    token_type_id = encoded_dict['token_type_ids']

    return input_id, attention_mask, token_type_id


id, mask, typ = bert_tokenizer(subjects[0])
print(id)
print(mask)
print(typ)


def build_data(doc):
    x_ids = []
    x_msk = []
    x_typ = []

    for sent in tqdm(doc):
        input_id, attention_mask, token_type_id = bert_tokenizer(sent)
        x_ids.append(input_id)
        x_msk.append(attention_mask)
        x_typ.append(token_type_id)

    x_ids = np.array(x_ids, dtype=int)
    x_msk = np.array(x_msk, dtype=int)
    x_typ = np.array(x_typ, dtype=int)

    return x_ids, x_msk, x_typ


x_train_ids, x_train_msk, x_train_typ = build_data(x_train)
x_test_ids, x_test_msk, x_test_typ = build_data(x_test)

y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

x_train_ids.shape, y_train.shape, x_test_ids.shape, y_test.shape

# Load BERT model
bert_model = TFBertModel.from_pretrained(
    'bert-base-uncased', cache_dir='bert_eng_ckpt')
bert_model.summary()  # bert_model을 확인한다. trainable params = 109,482,240

# TFBertMainLayer는 fine-tuning을 하지 않는다. (시간이 오래 걸림)
bert_model.trainable = False
bert_model.summary()  # bert_model을 다시 확인한다. trainable params = 0

# BERT 입력
# ---------
x_input_ids = Input(batch_shape=(None, MAX_LEN), dtype=tf.int32)
x_input_msk = Input(batch_shape=(None, MAX_LEN), dtype=tf.int32)
x_input_typ = Input(batch_shape=(None, MAX_LEN), dtype=tf.int32)

# BERT 출력
# ---------
output_bert = bert_model([x_input_ids, x_input_msk, x_input_typ])

# Downstream task : 주제 분류 (multi class classification)
# -------------------------------------------------------
y_output = Dense(n_topic, activation='softmax')(output_bert[1])
model = Model([x_input_ids, x_input_msk, x_input_typ], y_output)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.01))
model.summary()

output_bert

# 모델을 학습한다.
x_train = [x_train_ids, x_train_msk, x_train_typ]
x_test = [x_test_ids, x_test_msk, x_test_typ]
hist = model.fit(x_train, y_train, validation_data=(
    x_test, y_test), epochs=10, batch_size=1024)

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label='Test loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 시험 데이터로 학습 성능을 평가한다
pred = model.predict(x_test)
y_pred = np.argmax(pred, axis=1).reshape(-1, 1)
accuracy = (y_pred == y_test).mean()
print("\nAccuracy = %.2f %s" % (accuracy * 100, '%'))
