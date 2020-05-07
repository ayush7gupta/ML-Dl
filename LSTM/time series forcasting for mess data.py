# univariate data preparation
from numpy import array
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense

#the lines before wget got the code to work after struggling for 2 days due to dependencies
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
mess_data = [13.6, 31.8, 26, 29.9, 23.1, 11.4, 22.3, 13.8, 30.8, 25.6, 25.6, 17.2, 7.1, 22.75, 15.16, 29.9, 25, 26.7, 27.8, 7.8, 23, 12.8, 30.1, 22.2, 26.6, 25.5, 10.15, 19.7, 14, 29.9, 24, 26.3, 23.6, 8.9, 19.4, 16.1, 28.4, 26.7, 32.3, 24, 11.3, 20.3, 23.8, 32, 28.2, 30.9, 30.9, 29.43, 17.34, 13.57,
             32.86, 18.25, 20.33, 25.79, 29.15, 37.94, 29.39, 36.75, 23.58, 24.82, 17.26, 25.92, 20.72, 8.3, 4.45, 21.23, 22.4, 32.75, 24.1, 10.35, 30.64, 23.58, 25.3, 12.27, 29.5]

# choose a number of time steps
# chosen as 7 dues to periodicity of the data
n_steps = 7
n_features = 1
# split into samples
X, y = split_sequence(mess_data, n_steps)
for i in range(len(X)):
    print(X[i], y[i])

X = X.reshape((X.shape[0], X.shape[1], n_features))
# summarize the data


# define model using vanilla LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([24.1, 10.35, 30.64, 23.58, 25.3, 12.27, 29.5])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)