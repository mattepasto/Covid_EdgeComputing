import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

# Set random seed to get the same result after each time running the code
tf.random.set_random_seed(1)
df = pd.read_csv(r'.\datasets\dpc-covid19-ita-andamento-nazionale.csv', index_col = 'data', parse_dates = ['data'], usecols=['nuovi_positivi','data'])
dataset = df.values

# Scale data
# The input to scaler.fit -> array-like, sparse matrix, dataframe of shape (n_samples, n_features)
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

lookback = 15   # 5; optimal value for vanilla is 15
# A “lookback period” defines how many previous timesteps are used in order to predict the subsequent timestep. 
# In this regard, we are using a one-step prediction model. The lookback period is set to 5 in this instance. 
# This means that we are using the time steps at t-4, t-3, t-2, t-1, and t to predict the value at time t+1.

#n_neurons = 32  # 8 # 16
n_batch = 1

# Create input dataset
# The input shape should be [samples, time steps, features]
def create_dataset (X, look_back = 1):
    Xs, ys = [], []
    
    for i in range(len(X)-look_back-1):
        v = X[i:(i+look_back)]
        Xs.append(v)
        ys.append(X[i+look_back])
        
    return np.array(Xs), np.array(ys)

#lookback = 15   # 5
n_neurons = 8  # 8 # 16
n_features = 1
X_train, y_train = create_dataset(train, lookback)
X_test, y_test = create_dataset(test, lookback)
# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))  # trying to set time steps to 3 but got an error -> check dedicated file
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

'''# split a univariate sequence into samples
def split_sequence(sequence, look_back, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + look_back
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
 
# choose a number of time steps
n_steps_out = 3   # https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/ (Multi-Step LSTM Models -> Vector Output Model)
# split into samples
X_train, y_train = split_sequence(train, lookback, n_steps_out)
X_test, y_test = split_sequence(train, lookback, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], n_features,X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], n_features,X_test.shape[1]))
y_train = y_train.reshape((y_train.shape[0],y_train.shape[1]))
y_test = y_train.reshape((y_test.shape[0],y_test.shape[1]))'''

print('X_train.shape: ', X_train.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape) 
print('y_test.shape: ', y_test.shape)

'''X_train.shape:  (355, 1, 15)
y_train.shape:  (355, 1)
X_test.shape:  (167, 1, 15)
y_test.shape:  (167, 1)'''

# Create vanilla GRU model
def create_gru(units):
    model = Sequential()
    # Input layer 
    model.add(GRU (units = units, return_sequences = False, 
                 input_shape = [X_train.shape[1], X_train.shape[2]]))
    model.add(Dropout(0.01)) 
    # Output layer
    model.add(Dense(1)) 
    #Compile model
    model.compile(optimizer='adam',loss='mse')
   
    return model

model_gru = create_gru(n_neurons)

def fit_model(model):
    # This callback will stop the training when there is no improvement in the loss for 'patience' consecutive epochs.
    early_stop = EarlyStopping(monitor = 'val_loss', patience = 20, verbose=1)  
    history = model.fit(X_train, y_train, epochs = 200, validation_split = 0, #0.2
                    batch_size = 1, validation_data=(X_test, y_test), shuffle = False, callbacks = [early_stop]) # batch_size = 16
    return history

# Plot train loss and validation loss
def plot_loss (history, model_name):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Train vs Validation Loss for ' + model_name)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')

history_gru = fit_model(model_gru)

plot_loss (history_gru, 'Vanilla GRU')
plt.show()

# Transform data back to original data space
y_test = scaler.inverse_transform(y_test)
y_train = scaler.inverse_transform(y_train)

# Make prediction
def prediction(model):
    prediction_test = model.predict(X_test, batch_size = 1)     # In a stateful network, you should only pass inputs with a number of samples that can be divided by the batch size. Found: 167 samples. Batch size: 32.
    prediction_test = scaler.inverse_transform(prediction_test)
    prediction_train = model.predict(X_train, batch_size = 1)
    prediction_train = scaler.inverse_transform(prediction_train)
    return prediction_test, prediction_train

prediction_gru = prediction(model_gru)

# Plot test data vs prediction
def plot_future(prediction_test, prediction_train, model_name):
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(scaler.inverse_transform(dataset),label='Dataset')
    '''# plot the forecasts in red -> 09/10: provato ma non va; vedi meglio domani!
    for i in range(len(prediction_train)):
        off_s = len(dataset) + i - 1
        print(len(prediction_train[i]))
        off_e = off_s + len(prediction_train[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [dataset[off_s]] + prediction_train[i]
        plt.plot(xaxis, yaxis, color='red')'''
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookback:len(prediction_train)+lookback, :] = prediction_train
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(prediction_train)+(lookback*2)+1:len(dataset)-1, :] = prediction_test
    plt.plot(trainPredictPlot,label='Train')
    plt.plot(testPredictPlot,label='Test')
    plt.title('Test data and prediction for ' + model_name)
    plt.legend(loc='upper left')
    plt.xlabel('Time (day)')
    plt.ylabel('Daily positives')


plot_future(prediction_gru[0], prediction_gru[1], 'Vanilla GRU')
plt.show()

# Calculate MAE and RMSE
def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()

    print(model_name + ':')
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))
    print('')

evaluate_prediction(prediction_gru[0], y_test, 'Vanilla GRU')