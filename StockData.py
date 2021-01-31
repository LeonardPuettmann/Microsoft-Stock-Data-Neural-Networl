import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yf 
import matplotlib.pyplot as plt

msft = yf.Ticker('MSFT')

history = msft.history(period='max', interval='1d')
print(history)

def generate_series(data, value_num):
    close = data['Close']
    dividends = data['Dividends']
    tsg = tf.keras.preprocessing.sequence.TimeseriesGenerator(close, close,
                              length=value_num,
                              batch_size=len(close))
    global_index = value_num
    i, t = tsg[0]
    has_dividends = np.zeros(len(i))
    for b_row in range(len(t)):
        assert(abs(t[b_row] - close[global_index]) <= 0.001)
        has_dividends[b_row] = dividends[global_index] > 0            
        global_index += 1
    return np.concatenate((i, np.transpose([has_dividends])),
                           axis=1), t

# Normalizing the data
inputs, targets = generate_series(history, 4)
print(inputs[3818])

h_min = history.min()
normalized_h = (history - h_min) / (history.max() - h_min)

inputs, targets = generate_series(normalized_h, 4)
print(inputs[3818])

# Creating a Sequential Model with three layers
def create_model(n):
    model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.tanh, input_shape=(n+1,)),
    keras.layers.Dense( 128, activation =tf.nn.tanh),
    keras.layers.Dense(1)
    ])
    return model


def select_inputs(data, start, end, epochs):
    models = {}
    for inputs in range(start, end+1):
        print('Using {} inputs'.format(inputs))
        model_inputs, targets = generate_series(data, inputs)
        
        train_inputs = model_inputs[:-1000]
        val_inputs = model_inputs[-1000:]
        train_targets = targets[:-1000]
        val_targets = targets[-1000:]
        
        model = create_model(inputs)
        print('Training')
        model.compile(optimizer='adam', loss='mse') 
        h = model.fit(train_inputs, train_targets,
                  epochs=epochs,
                  batch_size=32,
                  validation_data=(val_inputs, val_targets))
        model_info = {'model': model, 'history': h.history}
        models[inputs] = model_info
    return models

trained_models = select_inputs(normalized_h, 2, 10, 20)


model_stats = {}
for k, v in trained_models.items():
    train_history = v['history']
    loss = train_history['loss'][-1]
    val_loss = train_history['val_loss'][-1]
    model_stats[k] = {'inputs': k, 'loss': loss, 'val_loss': val_loss}

# Plotting out the test error
val_loss = []
indices = []
for k, v in model_stats.items():
    indices.append(k)
    val_loss.append(v['val_loss'])
plt.plot(indices, val_loss)

close_min = history['Close'].min()
close_max = history['Close'].max()
for k in model_stats:
    e = ((close_max - close_min) * model_stats[k]['val_loss'] + close_min)
    print(k, e)
