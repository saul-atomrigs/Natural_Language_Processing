
import numpy as np
import pickle
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %cd '/content/drive/MyDrive/Colab Notebooks'

with open('data/pv_dm_D.pkl', 'rb') as f:
    _, D_array, y_data = pickle.load(f)

n_topic = len(set(y_data[:, 0]))
x_train, x_test, y_train, y_test = train_test_split(D_array, y_data, test_size=0.2)

# FNN 모델을 생성한다.
x_input = Input(batch_shape = (None, 400))
h_layer = Dense(50, activation='relu', kernel_regularizer=l2(0.005))(x_input)
y_output = Dense(n_topic, activation='softmax', kernel_regularizer=l2(0.005))(h_layer)
model = Model(x_input, y_output)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0005))
model.summary()

# 모델을 학습한다.
hist = model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 128, epochs = 50)

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label = 'Test loss')
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

