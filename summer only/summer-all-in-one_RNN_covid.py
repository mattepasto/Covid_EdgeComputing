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
df = pd.read_csv(r'.\datasets\dpc-covid19-ita-andamento-nazionale-from-june.csv', index_col = 'data', parse_dates = ['data'], usecols=['nuovi_positivi','data'])
dataset = df.values

# Scale data
# The input to scaler.fit -> array-like, sparse matrix, dataframe of shape (n_samples, n_features)
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

lookback = 7   # 5; optimal value for vanilla is 15
# A “lookback period” defines how many previous timesteps are used in order to predict the subsequent timestep. 
# In this regard, we are using a one-step prediction model. The lookback period is set to 5 in this instance. 
# This means that we are using the time steps at t-4, t-3, t-2, t-1, and t to predict the value at time t+1.

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

#n_neurons = 32  # 8 # 16   ---> EVERY MODEL A DIFFERNT NUM OF NEURONS!
n_features = 1
X_train, y_train = create_dataset(train, lookback)
X_test, y_test = create_dataset(test, lookback)
# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))  # trying to set time steps to 3 but got an error -> check dedicated file
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

print('X_train.shape: ', X_train.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape) 
print('y_test.shape: ', y_test.shape)

# Create Vanilla LSTM model
def create_lstm(units):
    model = Sequential()
    # Input layer
    model.add(LSTM(units = units, batch_input_shape=(n_batch, n_features, lookback), stateful=True)) # return_sequences=True
    model.add(Dropout(0.01)) # 23/09: Every LSTM layer should be accompanied by a Dropout layer. 
                            # This layer will help to prevent overfitting by ignoring randomly selected neurons during training, 
                            # and hence reduces the sensitivity to the specific weights of individual neurons. 
                            # 20% is often used as a good compromise between retaining model accuracy and preventing overfitting.
    # Output layer
    model.add(Dense(1))
    #Compile model
    model.compile(optimizer='adam',loss='mse')
    return model

model_lstm = create_lstm(8)

# Create Stacked LSTM model: we use two hidden layers
def create_stacked_lstm(units):
    model = Sequential()
    # Input layer
    model.add(LSTM(units = units, return_sequences=True, batch_input_shape=(n_batch,X_train.shape[1], X_train.shape[2]),stateful=True))
    model.add(Dropout(0.01))
    # First hidden layer
    model.add(LSTM(units = units))#, return_sequences=True))
    model.add(Dropout(0.01))
    '''# Second hidden layer
    model.add(LSTM(units = units))
    model.add(Dropout(0.2))'''
    # Output layer
    model.add(Dense(1))
    #Compile model
    model.compile(optimizer='adam',loss='mse')
    return model

model_stacked_lstm = create_stacked_lstm(2)

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

model_gru = create_gru(8)

# Create stacked GRU model
def create_stacked_gru(units):
    model = Sequential()
    # Input layer 
    model.add(GRU (units = units, return_sequences = True, 
                 input_shape = [X_train.shape[1], X_train.shape[2]]))
    model.add(Dropout(0.01)) 
    # Hidden layer
    model.add(GRU(units = units))                 
    model.add(Dropout(0.01))
    # Output layer
    model.add(Dense(units = 1)) 
    #Compile model
    model.compile(optimizer='adam',loss='mse')
   
    return model

model_stacked_gru = create_stacked_gru(4)

# Create BiLSTM model
def create_bilstm(units):
    model = Sequential()
    # Input layer
    model.add(Bidirectional(LSTM(units = units, input_shape=(X_train.shape[1], X_train.shape[2]))))
    '''# Hidden layer
    model.add(Bidirectional(LSTM(units = units)))'''    # if we add the layer again, put return_sequence = TRUE above
    # Output layer
    model.add(Dense(1))
    #Compile model
    model.compile(optimizer='adam',loss='mse')
    return model

model_bilstm = create_bilstm(1)

# Create BiGRU model
def create_bigru(units):
    model = Sequential()
    # Input layer
    model.add(Bidirectional(GRU(units = units, input_shape=(X_train.shape[1], X_train.shape[2]))))
    '''# Hidden layer
    model.add(Bidirectional(LSTM(units = units)))'''    # if we add the layer again, put return_sequence = TRUE above
    # Output layer
    model.add(Dense(1))
    #Compile model
    model.compile(optimizer='adam',loss='mse')
    return model

model_bigru= create_bigru(1)

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

history_lstm = fit_model(model_lstm)
history_stacked_lstm = fit_model(model_stacked_lstm)
history_gru = fit_model(model_gru)
history_stacked_gru = fit_model(model_stacked_gru)
history_bilstm = fit_model(model_bilstm)
history_bigru = fit_model(model_bigru)

# Plotting losses for each model
plot_loss (history_lstm, 'Vanilla LSTM')
#plt.show()
plot_loss (history_stacked_lstm, 'Stacked LSTM')
#plt.show()
plot_loss (history_gru, 'Vanilla GRU')
#plt.show()
plot_loss (history_stacked_gru, 'Stacked GRU')
#plt.show()
plot_loss (history_bilstm, 'Bidirectional LSTM')
#plt.show()
plot_loss (history_bigru, 'Bidirectional GRU')
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

prediction_lstm = prediction(model_lstm)
prediction_stacked_lstm = prediction(model_stacked_lstm)
prediction_gru = prediction(model_gru)
prediction_stacked_gru = prediction(model_stacked_gru)
prediction_bilstm = prediction(model_bilstm)
prediction_bigru = prediction(model_bigru)

# Plot test data vs prediction
def plot_future(prediction_test, prediction_train, model_name):
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(scaler.inverse_transform(dataset),label='Dataset')
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

plot_future(prediction_lstm[0], prediction_lstm[1], 'Vanilla LSTM')
#plt.show()
plot_future(prediction_stacked_lstm[0], prediction_stacked_lstm[1], 'Stacked LSTM')
#plt.show()
plot_future(prediction_gru[0], prediction_gru[1], 'Vanilla GRU')
#plt.show()
plot_future(prediction_stacked_gru[0], prediction_stacked_gru[1], 'Stacked GRU')
#plt.show()
plot_future(prediction_bilstm[0], prediction_bilstm[1], 'Bidirectional LSTM')
#plt.show()
plot_future(prediction_bigru[0], prediction_bigru[1], 'Bidirectional GRU')
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

evaluate_prediction(prediction_lstm[0], y_test, 'Vanilla LSTM')
evaluate_prediction(prediction_stacked_lstm[0], y_test, 'Stacked LSTM')
evaluate_prediction(prediction_gru[0], y_test, 'Vanilla GRU')
evaluate_prediction(prediction_stacked_gru[0], y_test, 'Stacked GRU')
evaluate_prediction(prediction_bilstm[0], y_test, 'Bidirectional LSTM')
evaluate_prediction(prediction_bigru[0], y_test, 'Bidirectional GRU')