import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVR
import pandas as pd
import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from scipy import spatial
from math import *
import sys
import time
import math
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import scipy.stats as st
import datetime
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout
from keras.layers import  CuDNNLSTM

def singleTurbineData(turbine_iter):
	df = pd.read_csv('la-haute-borne-data-2013-2016.csv', sep=';')
	df['Date_time'] = df['Date_time'].astype(str).str[:-6] #remove timezone (caused me an hour of pain)
	df.Date_time=pd.to_datetime(df['Date_time'])
	df=df.fillna(method='ffill')
	turbines=df.Wind_turbine_name.unique()
	df['sin']=np.sin(df['Wa_c_avg']/360*2*math.pi)
	df['cos']=np.cos(df['Wa_c_avg']/360*2*math.pi)
	df=df[df['Wind_turbine_name']==turbines[turbine_iter]]
	df=df.sort_values(by='Date_time')
	df = df.reset_index()
	return df,turbines[turbine_iter]


def updateDataset(df, test_date,train_start_date,end_date,trainSet,testSet,recordsBack):
	'''
	this function cleans up assumed datasets lengths.  Because some values are missing from the dataset, I update counts
	for test and train dataset variables based on the length of the value I receive from "currentTurbine"
	'''	
	currentTurbine=df[(df['Date_time']>= train_start_date) & (df['Date_time']<end_date)]
	
	if(len(currentTurbine.Date_time.values)==(trainSet+testSet+recordsBack)):
		return currentTurbine
	else:
		print("Adjusting dataset, value(s) missing from time series.")
		s = currentTurbine.Date_time.eq(test_date)
		location=s.index[s][-1]
		currentTurbine=df.loc[location-trainSet-recordsBack:location+testSet-1]
		if(len(currentTurbine.Date_time.values)==(trainSet+testSet+recordsBack)):
			return currentTurbine
		else:
			print("Exiting...")                    
			sys.exit("Error Retrieving data")



def createGraph(weighted,actual, rmse):
	X = np.arange(0,len(actual))
	figure = plt.figure()
	tick_plot = figure.add_subplot(1, 1, 1)
	tick_plot.plot(X, actual,  color='green', linestyle='-', marker='*', label='Actual')
	tick_plot.plot(X, weighted,  color='blue',linestyle='-', marker='*', label='Predictions')
	plt.xlabel('Time (ten minute increments for a day)')
	plt.ylabel('Angle')
	plt.legend(loc='upper left')
	plt.title('Wind Angles and SVR Predictions\nError:  '+str(rmse))
	plt.show()

def setupTrainTestSets(train_test_data,total,recordsBack, trainSet,cos=False):
	from sklearn.preprocessing import normalize
	i=0
	training_data = []
	wind_direction_actual = []
	wind_speed=[]
	temperature=[]
	actual=[]

	while i <total:
		wind_speed.append(train_test_data.Ws_avg.values[i:recordsBack+i])
		temperature.append(train_test_data.Ot_avg.values[i:recordsBack+i])		
		if(cos):
			training_data.append(train_test_data.cos.values[i:recordsBack+i])
			wind_direction_actual.append(train_test_data.cos.values[recordsBack+i])
		else:
			training_data.append(train_test_data.sin.values[i:recordsBack+i])
			wind_direction_actual.append(train_test_data.sin.values[recordsBack+i])
		
		actual.append(train_test_data['Wa_c_avg'].values[recordsBack+i])
		i+=1

	training_data=np.array(training_data)
	wind_direction_actual=np.array(wind_direction_actual)
	
	wind_speed=np.array(wind_speed)
	wind_speed=normalize(wind_speed)
	
	temperature=np.array(temperature)
	temperature=normalize(temperature)

	
	training_data=np.reshape(training_data, (training_data.shape[0], training_data.shape[1],1))
	wind_speed=np.reshape(wind_speed, (wind_speed.shape[0], wind_speed.shape[1],1))
	temperature=np.reshape(temperature, (temperature.shape[0], temperature.shape[1],1))

	
	training_data=np.concatenate((training_data, wind_speed), axis=2)
	training_data=np.concatenate((training_data, temperature), axis=2)

	trainX_initial=training_data[:trainSet-144]
	trainY_initial=wind_direction_actual[:trainSet-144]

	validationX=training_data[trainSet-144:trainSet]
	validationY=wind_direction_actual[trainSet-144:trainSet]

	trainX_full=training_data[:trainSet]
	trainY_full=wind_direction_actual[:trainSet]

	testX=training_data[trainSet:]
	testY=wind_direction_actual[trainSet:]
	actual=np.array(actual[trainSet:])


	return trainX_initial, trainY_initial, validationX, validationY, trainX_full, trainY_full, testX,testY, actual


'''
initialize variables
'''
def dataSetup(test_date):
        testSet=24*6 #test 1 day of values

        previousDays_rows=365
        trainSet=previousDays_rows*24*6 

        total=trainSet+testSet

        previousDays_columns=6
        recordsBack=previousDays_columns*24*6

        test_date=test_date+datetime.timedelta(days=0)
        train_start_date=test_date+datetime.timedelta(days=-(previousDays_rows+previousDays_columns))
        end_date=test_date+datetime.timedelta(minutes = 10*testSet)

        df, turbine_name=singleTurbineData(0)	

        currentTurbine=updateDataset(df,test_date,train_start_date,end_date,trainSet,testSet,recordsBack)
        return currentTurbine,total,recordsBack, trainSet

def convertToDegrees(sin_prediction,cos_prediction):
	'''
	Converting sine and cosine back to its circular angle depends on finding which of the the 4 circular quadrants the 
	prediction will fall into. If sin and cos are both GT 0, degrees will fall in 0-90.  If sin>0 cos<0, degrees will fall into 90-180, etc. 
	'''
	inverseSin=np.degrees(np.arcsin(sin_prediction))
	inverseCos=np.degrees(np.arccos(cos_prediction))
	radians_sin=[]
	radians_cos=[]
	for a,b,c,d in zip(sin_prediction, cos_prediction, inverseSin, inverseCos):
		if(a>0 and b>0):
			radians_sin.append(c)
			radians_cos.append(d)	
		elif(a>0 and b<0):
			radians_sin.append(180-c)
			radians_cos.append(d)	
		elif(a<0 and b<0):
			radians_sin.append(180-c)
			radians_cos.append(360-d)	
		elif(a<0 and b>0):
			radians_sin.append(360+c)
			radians_cos.append(360-d)
	radians_sin=np.array(radians_sin)
	radians_cos=np.array(radians_cos)
	return radians_sin, radians_cos



def calcWeightedDegreePredictions(sin_error,cos_error,radians_sin,radians_cos):
	errorTotal=cos_error+sin_error
	sinWeight=(errorTotal-sin_error)/errorTotal
	cosWeight=(errorTotal-cos_error)/errorTotal
	weighted=np.add(sinWeight*radians_sin, cosWeight*radians_cos)
	return weighted





def train_predict():
        model = Sequential()
        model.add(CuDNNLSTM(128*trainX_initial.shape[2]*2, input_shape=(recordsBack,trainX_initial.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_absolute_error', optimizer='adam')


        checkpointer=ModelCheckpoint('weights.h5', monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        earlystopper=EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
        model.fit(trainX_initial, trainY_initial, validation_data=(validationX, validationY),epochs=20, batch_size=testX.shape[0], verbose=2, shuffle=False,callbacks=[checkpointer, earlystopper])
        
        model.load_weights("weights.h5")

        validationPredict=model.predict(validationX)
        validation_mae=mean_absolute_error(validationY, validationPredict)
        
        model.fit(trainX_initial, trainY_initial, validation_data=(validationX, validationY),epochs=1, batch_size=testX.shape[0], verbose=2)


        testPredict = model.predict(testX)

        testPredict[testPredict > 1] = 1
        testPredict[testPredict <-1] = -1
        return testPredict, validation_mae





results = pd.DataFrame(columns=['test_date','degrees_mae','rmse'])

for i in range(1,30):
        date_to_test=datetime.datetime(2016, 1, i)
        currentTurbine,total,recordsBack, trainSet=dataSetup(date_to_test)

        trainX_initial, trainY_initial, validationX, validationY, trainX_full, trainY_full, testX,testY,actual=setupTrainTestSets(currentTurbine,total,recordsBack, trainSet)
        testPredict_sin,validation_mae_sin=train_predict()
        rmse = math.sqrt(mean_squared_error(testY, testPredict_sin))
        mae=mean_absolute_error(testY, testPredict_sin)

        trainX_initial, trainY_initial, validationX, validationY, trainX_full, trainY_full, testX,testY,actual=setupTrainTestSets(currentTurbine,total,recordsBack, trainSet,cos=True)
        testPredict_cos,validation_mae_cos=train_predict()
        rmse = math.sqrt(mean_squared_error(testY, testPredict_cos))
        mae=mean_absolute_error(testY, testPredict_cos)

        radians_sin, radians_cos=convertToDegrees(testPredict_sin,testPredict_cos)
        weighted=calcWeightedDegreePredictions(validation_mae_sin,validation_mae_cos,radians_sin,radians_cos)
        degrees_mae=mean_absolute_error(actual, weighted)

        mse = mean_squared_error(actual, weighted)
        rmse = sqrt(mse)

        results = results.append({'test_date':str(date_to_test)[:10],'degrees_mae': degrees_mae, 'rmse': rmse}, ignore_index=True)
        print(results)
        guesses_pandas = pd.DataFrame(weighted)
        actual_pandas = pd.DataFrame(actual)
        guesses_file="multi_LSTM_results/"+str(date_to_test)[:10]+"guesses.csv"
        actual_file="multi_LSTM_results/"+str(date_to_test)[:10]+"actual.csv"
        guesses_pandas.to_csv(guesses_file)
        actual_pandas.to_csv(actual_file)
        results.to_csv("multi_LSTM_results/year_results.csv")

