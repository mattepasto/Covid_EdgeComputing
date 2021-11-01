from math import sqrt
from keras.callbacks import EarlyStopping
from numpy import concatenate
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = read_csv(r'.\datasets\dpc-covid19-ita-andamento-nazionale.csv', header=0, index_col=0)
# dropping NaN columns
dataset.drop('stato', axis=1, inplace=True)
dataset.drop('casi_da_sospetto_diagnostico', axis=1, inplace=True)
dataset.drop('casi_da_screening', axis=1, inplace=True)
dataset.drop('note', axis=1, inplace=True)
dataset.drop('note_test', axis=1, inplace=True)
dataset.drop('note_casi', axis=1, inplace=True)
values = dataset.values
values=values[:,[0,1,2,10,7,8,6]]	# in the column 10, there are the number of swabs; in the 8th, deaths 

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
print(reframed)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[7,8,9,10,11,12]], axis=1, inplace=True) # not dropping 13 (nuovi positivi è la settima colonna, ci sono 7 colonne t-1 che tengo più quella di riferimento che vogliamo al tempo t ossia la settima)
print(reframed.head())
 
# split into train and test sets
values = reframed.values
train_size = int(len(values) * 0.67)
test_size = len(values) - train_size
train = values[:train_size, :]
test = values[train_size:len(values), :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]	# taking the last column as output
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

n_neurons = 8	# optimum value for better error; #2
n_batch = 1
# design network
model = Sequential()	# from keras.models
model.add(LSTM(n_neurons, batch_input_shape=(n_batch,train_X.shape[1], train_X.shape[2]), stateful=True)) #input_shape=(train_X.shape[1], train_X.shape[2]))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')	# mae = computes the mean absolute error between labels and predictions.
											# adam = if you pass adam like this, the default paramters are used (https://keras.io/api/optimizers/)
# fit network
#early_stop = EarlyStopping(monitor = 'val_loss', patience = 10, verbose=1) 
history = model.fit(train_X, train_y, epochs=200, batch_size=1, validation_data=(test_X, test_y), 
				verbose=2, shuffle=False)#, callbacks = [early_stop])	# verbose to see progress

# https://keras.io/api/models/model_training_apis/ (fit method)
# epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. 
# Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". 
# The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.

# batch_size: Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32.
# validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. 
# shuffle: whether to shuffle the training data before each epoch)

# plot history
pyplot.plot(history.history['loss'], label='Train loss')
pyplot.plot(history.history['val_loss'], label='Validation loss')
pyplot.legend()
pyplot.show()
 
# make a prediction for test_X
yhat = model.predict(test_X, batch_size = 1)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# make a prediction for train_X
xhat = model.predict(train_X, batch_size = 1)
train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
# invert scaling for forecast
inv_xhat = concatenate((xhat, train_X[:, 1:]), axis=1)
inv_xhat = scaler.inverse_transform(inv_xhat)
inv_xhat = inv_xhat[:,0]
'''# invert scaling for actual
test_X = test_X.reshape((len(test_X), 1))
inv_x = concatenate((test_X, train_X[:, 1:]), axis=1)
inv_x = scaler.inverse_transform(inv_x)
inv_x = inv_x[:,0]'''
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# Plot test data vs prediction
def plot_future(prediction_test, prediction_train):
    
    plt.figure(figsize=(10, 6))
    new_pos = dataset.iloc[:,6]		# new_pos are in the 6th column in the csv
    sliced=new_pos.values
    plt.plot(sliced,label='Dataset')
    # shift train predictions for plotting
    trainPredictPlot = np.empty((len(values)))
    trainPredictPlot[:] = np.nan
    trainPredictPlot = concatenate((prediction_train, trainPredictPlot[:(len(values)-len(prediction_train))]))
    # shift test predictions for plotting
    testPredictPlot = np.empty(len(values))
    testPredictPlot[:] = np.nan
    testPredictPlot = concatenate((testPredictPlot[:(len(values)-len(prediction_test))], prediction_test))
    
    plt.plot(trainPredictPlot,label='Train')
    plt.plot(testPredictPlot,label='Test')
    plt.title('Test data and prediction for multi-var LSTM')
    plt.legend(loc='upper left')
    plt.xlabel('Time (day)')
    plt.ylabel('Daily positives')
    plt.show()

plot_future(inv_yhat, inv_xhat)