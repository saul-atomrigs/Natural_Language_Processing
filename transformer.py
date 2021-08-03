from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import pickle

# %cd '/content/drive/MyDrive/Colab Notebooks/삼성멀캠/자연어처리/7.챗봇_번역'
from transformer import Transformer

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/MyDrive/Colab Notebooks'
MODEL_PATH = 'data/transformer_model.h5'
LOAD_MODEL = False

# 단어 목록 dict를 읽어온다.
with open('data/chatbot_voc.pkl', 'rb') as f:
    word2idx,  idx2word = pickle.load(f)
    
# 학습 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 읽어온다.
with open('data/chatbot_train.pkl', 'rb') as f:
    trainXE, trainXD, trainYD, _, _ = pickle.load(f)

# Model
# -----
K.clear_session()
src = Input((None, ), dtype="int32", name="src")
tar = Input((None, ), dtype="int32", name="tar")

logits, enc_enc_attention, dec_dec_attention, enc_dec_attention = Transformer(
    num_layers=4,
    d_model=128,
    num_heads=8,
    d_ff=512,
    input_vocab_size=len(word2idx) + 2,
    target_vocab_size=len(word2idx) + 2,
    dropout_rate=0.1)(src, tar)

model = Model(inputs=[src, tar], outputs=logits)
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy')

if LOAD_MODEL:
    model.load_weights(MODEL_PATH)

model.summary()

# 학습 (teacher forcing)
# ----------------------
hist = model.fit([trainXE, trainXD], trainYD, batch_size = 512, epochs=100, shuffle=True)

# 학습 결과를 저장한다
model.save(MODEL_PATH)

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
