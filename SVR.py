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



def train_predict(train_test_data, c_,g_, cos=False):
	i=0
	x = []
	y = []
	while i <total:
		if(cos):

			x.append(train_test_data.cos.values[i:recordsBack+i])
			y.append(train_test_data.cos.values[recordsBack+i])
		else:
			x.append(train_test_data.sin.values[i:recordsBack+i])
			y.append(train_test_data.sin.values[recordsBack+i])
		i+=1

	svr_rbf = SVR(kernel='rbf', C=c_ ,gamma=g_)
	y_rbf = svr_rbf.fit(x[:trainSet], y[:trainSet]).predict(x)
	y_rbf[y_rbf > 1] = 1
	y_rbf[y_rbf < -1] = -1
	mae = mean_absolute_error(y[trainSet:], y_rbf[trainSet:])
	mse = mean_squared_error(y[trainSet:], y_rbf[trainSet:])

	rmse = sqrt(mse)
	return y_rbf[trainSet:], rmse


def WindDirectionWithDates(train_test_data, test_date, end_date):

	actual_values=train_test_data[(train_test_data['Date_time']>= test_date) & (train_test_data['Date_time']<end_date)].Wa_c_avg.values
	return np.array(actual_values)




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



def calcWeightedRMSEDegreePredictions(sin_rmse,cos_rmse,radians_sin,radians_cos):
	rmseTotal=cos_rmse+sin_rmse
	sinWeight=(rmseTotal-sin_rmse)/rmseTotal
	cosWeight=(rmseTotal-cos_rmse)/rmseTotal
	weighted=np.add(sinWeight*radians_sin, cosWeight*radians_cos)
	return weighted


def createGraph(weighted,actual, rmse):
	X = np.arange(0,testSet)
	figure = plt.figure()
	tick_plot = figure.add_subplot(1, 1, 1)
	tick_plot.plot(X, actual,  color='green', linestyle='-', marker='*', label='Actual')
	tick_plot.plot(X, weighted,  color='blue',linestyle='-', marker='*', label='Predictions')
	plt.xlabel('Time (ten minute increments for a day)')
	plt.ylabel('Angle')
	plt.legend(loc='upper left')
	plt.title('Wind Angles and SVR Predictions\nRMSE:  '+str(rmse))
	plt.show()



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




def printResults(weighted,actual, test_date):
	mae = mean_absolute_error(weighted, actual)
	mse = mean_squared_error(weighted, actual)
	rmse = sqrt(mse)
	print(str(test_date)[:16]+" RMSE weighted degree prediction:"+str(rmse))
	print(str(test_date)[:16]+" MAE weighted degree prediction:"+str(mae))
	return rmse,mae


def updateDataset(df, test_date,train_start_date,end_date):
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
			sys.exit("Error Retrieving data")


def runFull(plot=False):
	C_=[1000]
	gamma_=[0.00001]
	results = pd.DataFrame(columns=['turbine_name','trainSet_days_size','test_date','C','Gamma','Sin_RMSE','Cos_RMSE','Degrees_RMSE','Degrees_mae'])
	for turbine_iter in range(0,4):
		for x in range(0,1000):
			for c in C_:
				for g in gamma_:
					test_date=datetime.datetime(2016, 1, 1)
					test_date=test_date+datetime.timedelta(days=x)
					train_start_date=test_date+datetime.timedelta(days=-(previousDays_rows+previousDays_columns))
					end_date=test_date+datetime.timedelta(minutes = 10*testSet)

					df, turbine_name=singleTurbineData(turbine_iter)	

					
					currentTurbine=updateDataset(df,test_date,train_start_date,end_date)
					
					actual=WindDirectionWithDates(currentTurbine, test_date, end_date)
					sin_prediction, sin_rmse=train_predict(currentTurbine,c,g)
					cos_prediction, cos_rmse=train_predict(currentTurbine,c,g,cos=True)


					radians_sin, radians_cos=convertToDegrees(sin_prediction,cos_prediction)
					weighted=calcWeightedRMSEDegreePredictions(sin_rmse,cos_rmse,radians_sin,radians_cos)

					degrees_rmse, mae=printResults(weighted,actual, test_date)
					results = results.append({'turbine_name': turbine_name,'trainSet_days_size': previousDays_rows, 'test_date':str(test_date)[:16],'C': c, 'Gamma': g, 'Sin_RMSE': sin_rmse, 'Cos_RMSE': cos_rmse, 'Degrees_RMSE': degrees_rmse,"Degrees_mae":mae}, ignore_index=True)
					print(results)
					guesses_pandas = pd.DataFrame(weighted)
					actual_pandas = pd.DataFrame(actual)
					guesses_file="SVR_results/"+str(test_date)[:10]+"guesses.csv"
					actual_file="SVR_results/"+str(test_date)[:10]+"actual.csv"
					guesses_pandas.to_csv(guesses_file)
					actual_pandas.to_csv(actual_file)
					results.to_csv("SVR_results/year_results.csv")

					if(plot):
						plot(weighted,actual)

def runOneDay(plot=True,turbine_iter=0,x=0,c=1000,g=.00001 ):
	test_date=datetime.datetime(2016, 1, 2)
	test_date=test_date+datetime.timedelta(days=x)
	train_start_date=test_date+datetime.timedelta(days=-(previousDays_rows+previousDays_columns))
	end_date=test_date+datetime.timedelta(minutes = 10*testSet)

	df, turbine_name=singleTurbineData(0)	


	currentTurbine=updateDataset(df,test_date,train_start_date,end_date)

	actual=WindDirectionWithDates(currentTurbine, test_date, end_date)
	sin_prediction, sin_rmse=train_predict(currentTurbine,c,g)
	cos_prediction, cos_rmse=train_predict(currentTurbine,c,g,cos=True)
        

	radians_sin, radians_cos=convertToDegrees(sin_prediction,cos_prediction)
	weighted=calcWeightedRMSEDegreePredictions(sin_rmse,cos_rmse,radians_sin,radians_cos)

	degrees_rmse=printResults(weighted,actual, test_date)
	
	actual_p = pd.DataFrame(actual)
	actual_p.to_csv('actual.csv')
	
	weighted_p=pd.DataFrame(weighted)
	weighted_p.to_csv('SVR.csv')
	
	if(plot):
		createGraph(weighted,actual,degrees_rmse)



'''
initialize variables
'''
testSet=24*6 #test 1 day of values

previousDays_rows=90
trainSet=previousDays_rows*24*6 #train on 90 previous days of data

total=trainSet+testSet

previousDays_columns=6
recordsBack=previousDays_columns*24*6


#main method
runFull()
#runOneDay()

