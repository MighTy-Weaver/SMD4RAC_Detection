import keras
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Bidirectional

tf.config.run_functions_eagerly(True)

data = pd.read_csv('./summer_data_compiled.csv', index_col=0)
print(list(data))
AC = data['AC'].to_numpy().astype('float32')
print(AC, len(AC))
df_X = data.drop(['Date', 'Hour', 'Location', 'AC', 'Time'], axis=1)
print(list(df_X))
X_num = df_X.to_numpy().astype('float32')
X_train, X_test, y_train, y_test = train_test_split(X_num, AC, test_size=0.1, shuffle=True)


def build_MLP_model():
    model = keras.Sequential()
    model.add(Dense(256, input_dim=10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='softmax'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='softmax'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def build_LSTM_model():
    model = keras.Sequential()
    model.add(LSTM(64, input_dim=10, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='softmax'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def build_BiLSTM_model():
    model = keras.Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_dim=10))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='softmax'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


model = build_MLP_model()
print(model.summary())

model.fit(x=X_train.reshape(-1, 10, 1), y=y_train, batch_size=256, epochs=30, verbose=1,
          validation_data=(X_test.reshape(-1, 10, 1), y_test))
