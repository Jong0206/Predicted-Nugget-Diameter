from tensorflow.keras.models import load_model
from functions import datasets

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

settingFp = open('Setting.txt', "r")
featureList = []
mean_list = []
Rsquared_list = []
RMSE_list = []
loss_list = []

Setting_temp = settingFp.readline()#*****Common Setting(5)*****

#머신러닝에 사용할 Feature
Setting_temp = settingFp.readline()
Setting_temp = Setting_temp.split()
del Setting_temp[0]
for i in Setting_temp:
	if not i == 0:
		featureList.append(i)
for i in range(15):
	Setting_temp = settingFp.readline()#Common Setting 및 Training Setting의 나머지 줄

Setting_temp = settingFp.readline()#*****Estimation Setting(5)*****
Setting_temp = settingFp.readline()
Setting_temp = Setting_temp.split()
Model_name = Setting_temp[1]

Setting_temp = settingFp.readline()
Setting_temp = Setting_temp.split()
Test_dataset = Setting_temp[1]

Setting_line = settingFp.readline()
Setting_temp = Setting_line.split()
Estimated_result = Setting_temp[1]

settingFp.close()

#저장한 모델 불러오기
model_path = "./Models/{}".format(Model_name)
print("[INFO] loading network...")
model = load_model(model_path)

# 학습한 모델을 이용한 예측 테스트
print("[INFO] predicting ND...")
inputPath = os.path.sep.join(["Dataset/", Test_dataset])

P_df = datasets.load_welding_attributes_P(inputPath)
testX = datasets.process_welding_attributes(False, P_df, featureList, model_path)
test = P_df
testY = test['ND']
preds = model.predict(testX)

directory = "Estimation"
if not os.path.exists(directory):
	os.makedirs(directory)
export_path = "Estimation/{}".format(Estimated_result)
predicted = preds.flatten()
result = np.array([preds.flatten()])
result = result.transpose()

mean, Rsquared = datasets.CalcRsquared(predicted, testY)
Rsquared_list.append(Rsquared)
#Rsquared_list = np.array(Rsquared_list)
#Rsquared_list = Rsquared_list.transpose()

RMSE = datasets.CalcRMSE(predicted, testY)
RMSE_list.append(RMSE)
#RMSE_list = np.array(RMSE_list)
#RMSE_list = RMSE_list.transpose()
del P_df[P_df.columns[0]]
P_df.insert(1,'ND_Estimated', result)
P_df.insert(15, 'R^2', (Rsquared_list+[None]*len(P_df))[:len(P_df)], True)
P_df.insert(16, 'RMSE', (RMSE_list + [None]*len(P_df))[:len(P_df)], True)
P_df.to_csv(export_path)


'''
test = 1

testY = test["BD"]
testY = np.hstack([testY])

predicted = preds.flatten()
observed = testY
mean, Rsquared = datasets.CalcRsquared(predicted, observed)
RMSE = datasets.CalcRMSE(predicted, observed)

result = np.array([testY, preds.flatten()])
result = result.transpose()
Labels = ['Label', 'Predicted']
result_df = pd.DataFrame.from_records(result, columns=Labels)

# 예측 테스트 저장
directory = "Estimation/{}".format(_model_description)
if (Rsquared > 0 and Rsquared < 1):
    if not os.path.exists(directory):
        os.makedirs(directory)
    export_path = "Estimation/{}/result-{}.csv".format(_model_description, _model_description)
    # result_df.to_excel(export_path, sheet_name='Sheet1')
    result_df.to_csv(export_path)

f = open('Estimation/{}/result-{}-Summary.txt'.format(_model_description, _model_description), mode='wt', encoding='utf-8')
f.writelines("[INFO] mean: {:.2f} Rsquared: {:.2f} RMSE: {:.2f} \n".format(mean, Rsquared, RMSE))
f.close()
'''