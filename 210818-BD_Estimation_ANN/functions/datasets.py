# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import glob
import cv2
import os

def load_welding_attributes_P(inputPath):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	df = pd.read_excel(inputPath, header=0, sheet_name=None)

	P_df = df['P']

	#P_index_list = ['P' + str(i + 1) for i in range(15)]
	#P_index_list.append('BD')
	#P_df = P_df[P_index_list]
	#P_df['P16'] = P_df['P9']/P_df['P1']

	# return the data frame
	return P_df

def load_welding_attributes_X(inputPath):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	df = pd.read_excel(inputPath, header=0, sheet_name=None)

	X_df = df['X']

	X_index_list = ['X' + str(i + 1) for i in range(11)]
	X_index_list.append('ND')
	X_df = X_df[X_index_list]

	# return the data frame
	return X_df

def process_welding_attributes(training_flag, df, features, scaler_folder_name):
	# features: 분석에 사용하고 싶은 feature들만 index형태로 받기
	# performin min-max scaling each continuous feature column to
	# the range [0, 1]
	if training_flag == True:
		scaler = MinMaxScaler()
		scaler.fit(df[features])
		scaler_filename = scaler_folder_name + "/scaler.save"
		joblib.dump(scaler, scaler_filename)
	else:
		scaler_filename = scaler_folder_name + "/scaler.save"
		scaler = joblib.load(scaler_filename)
	trainContinuous = scaler.transform(df[features])

	# construct our training and testing data points by concatenating
	# the categorical features with the continuous features
	trainX = np.hstack([trainContinuous])
	# trainX = np.hstack([df[features]])

	# return the concatenated training and testing data
	return trainX


def CalcRMSE(_predicted, _observed):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	error = _predicted - _observed
	RMSE = (np.average(error ** 2))**0.5

	# return the data frame
	return RMSE

def CalcRsquared(_predicted, _observed):

	error = _predicted - _observed
	mean = np.mean(_observed)
	sse = np.sum(error ** 2)
	sst = np.sum((_observed - mean) ** 2)
	Rsquared = 1 - sse / sst

	return (mean, Rsquared)