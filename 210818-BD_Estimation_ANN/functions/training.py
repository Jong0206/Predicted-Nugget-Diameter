# import the necessary packages
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from functions import datasets
from functions import models
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def traning_Pdata(_numberOfIteration, _inputPath, _featureList, _nodeList, _model_description, _testDataRate=0):
	mean_list = []
	Rsquared_list = []
	RMSE_list = []
	loss_list = []

	for k in range(_numberOfIteration):

		#모델 저장용 디랙토리 생성
		directory = "Models/{}-{}".format(_model_description, int(k))
		if not os.path.exists(directory):
			os.makedirs(directory)

		# construct the path to the input .txt file that contains information
		# on each house in the dataset and then load the dataset
		print("[INFO] loading welding attributes...")
		# inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])

		P_df = datasets.load_welding_attributes_P(_inputPath)

		# construct a training and testing split with 75% of the data used
		# for training and the remaining 25% for evaluation
		# print("[INFO] constructing training/testing split...")
		if (_testDataRate < 0.000001):
			train = P_df
			test = P_df
		else:
			(train, test) = train_test_split(P_df, test_size=_testDataRate)
		# , random_state = 42

		# find the largest house price in the training set and use it to
		# scale our house prices to the range [0, 1] (this will lead to
		# better training and convergence)
		maxNI = P_df["ND"].max()
		# P_trainY = P_df["NI"] / maxNI
		trainY = train["ND"]
		testY = test["ND"]
		trainY = np.hstack([trainY])
		testY = np.hstack([testY])

		# process the house attributes data by performing min-max scaling
		# on continuous features, one-hot encoding on categorical features,
		# and then finally concatenating them together
		print("[INFO] processing data...")

		trainX = datasets.process_welding_attributes(True, train, _featureList, directory)
		testX = datasets.process_welding_attributes(False, test, _featureList, directory)

		# create our MLP and then compile the model using mean absolute
		# percentage error as our loss, implying that we seek to minimize
		# the absolute percentage difference between our price *predictions*
		# and the *actual prices*
		if (len(_nodeList) == 2):
			model = models.create_mlp_2layer(trainX.shape[1], _nodeList[0], _nodeList[1], regress=True)
		if (len(_nodeList) == 3):
			model = models.create_mlp_3layer(trainX.shape[1], _nodeList[0], _nodeList[1], _nodeList[2], regress=True)

		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
			initial_learning_rate=1e-3,
			decay_steps=100,
			decay_rate=0.99)

		optimizer = Adam(learning_rate=lr_schedule, decay=1e-3 / 200)
		model.compile(loss="mean_squared_error", optimizer=optimizer)
		history = model.fit(trainX, trainY, epochs=1000, batch_size=4)

		'''

        opt = Adam(lr=1e-3, decay=1e-3 / 200)
        model.compile(loss="mean_squared_error", optimizer=opt)



        def lr_scheduler(epoch, lr):
            if epoch > 1200:
                lr = 0.000000000001
                return lr
            if epoch > 900:
                lr = 0.0000000001
                return lr
            if epoch > 600:
                lr = 0.00000001
                return lr
            if epoch > 300:
                lr = 0.000001
                return lr
            return lr
        callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)]

        model.summary()

        # train the model
        print("[INFO] training model...")
        # model.fit(P_trainX, P_trainY, validation_data=(testX, testY),epochs=200, batch_size=8)
        model.fit(P_trainX, P_trainY, epochs=1500, batch_size=4, callbacks=callbacks)
        '''

		# first_layer_weights = model.layers[0].get_weights()[0]
		# first_layer_biases  = model.layers[0].get_weights()[1]
		# second_layer_weights = model.layers[1].get_weights()[0]
		# second_layer_biases  = model.layers[1].get_weights()[1]
		# third_layer_weights = model.layers[2].get_weights()[0]
		# third_layer_biases  = model.layers[2].get_weights()[1]

		# tf.keras.utils.plot_model(
		#     model, to_file='model.png', show_shapes=False, show_layer_names=True,
		#     rankdir='TB', expand_nested=False, dpi=96)

		# 학습된 모델 저장
		#modelFile = "/Models/{}/model-{}-{}".format(_model_description, _model_description, int(k))
		modelFile = "./{}".format(directory)
		model.save(modelFile)

		# 학습한 모델을 이용한 예측 테스트
		print("[INFO] predicting ND...")
		preds = model.predict(testX)

		predicted = preds.flatten()
		observed = testY
		mean, Rsquared = datasets.CalcRsquared(predicted, observed)
		RMSE = datasets.CalcRMSE(predicted, observed)

		result = np.array([testY, preds.flatten()])
		result = result.transpose()
		Labels = ['Label', 'Predicted']
		result_df = pd.DataFrame.from_records(result, columns=Labels)

		# 예측 테스트 저장
		directory = "Results/{}".format(_model_description)
		if (Rsquared > 0 and Rsquared < 1):
			if not os.path.exists(directory):
				os.makedirs(directory)
			export_path = "Results/{}/result-{}-{}.csv".format(_model_description, _model_description, int(k))
			# result_df.to_excel(export_path, sheet_name='Sheet1')
			result_df.to_csv(export_path)

			#loss_list.append(history.history['loss'])
			#print(loss_list)
			mean_list.append(mean)
			Rsquared_list.append(Rsquared)
			RMSE_list.append(RMSE)


			#plt.plot(history.history['loss'], 'r-')
			#plt.rc('font', size=12)
			#plt.xlabel('epochs')
			#plt.ylabel('loss')
			#plt.ylim([0, 1.5])
			#plt.ylabel('val_loss')
			#plt.show()

	a = len(mean_list)
	f = open('Results/{}/result-{}-Summary.txt'.format(_model_description, _model_description), mode='wt', encoding='utf-8')
	for i in range(len(mean_list)):
		f.writelines("[INFO] mean: {:.2f} Rsquared: {:.2f} RMSE: {:.2f} \n".format(mean_list[i], Rsquared_list[i], RMSE_list[i]))

	av_mean = np.average(mean_list)
	av_Rsquared = np.average(Rsquared_list)
	max_Rsquared = np.max(Rsquared_list)
	av_RMSE = np.average(RMSE_list)
	max_RMSE = np.max(RMSE_list)
	f.writelines("[Summary] mean: {:.2f} Rsquared_MEAN: {:.2f} Rsquared_MAX: {:.2f} RMSE_MEAN: {:.2f} RMSE_MAX: {:.2f}\n".format(av_mean, av_Rsquared, max_Rsquared, av_RMSE, max_RMSE))
	#f.writelines("[Summary] loss: {}\n".format(loss_list))


	f.close()