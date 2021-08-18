# USAGE
# python mlp_regression.py --dataset Houses-dataset/Houses\ Dataset/

# import the necessary packages
from functions import training
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=False,
	help="path to input dataset of house images")
args = vars(ap.parse_args())

settingFp = open('Setting.txt', "r")

featureList = []
nodeList = []

Setting_temp = settingFp.readline()#*****Common Setting(5)*****

#머신러닝에 사용할 Feature
Setting_temp = settingFp.readline()
Setting_temp = Setting_temp.split()
del Setting_temp[0]
for i in Setting_temp:
	if not i == 0:
		featureList.append(i)
for i in range(4):
	Setting_temp = settingFp.readline()#Common Setting의 나머지 줄

Setting_line = settingFp.readline()#*****Training Setting(10)*****
Setting_line = settingFp.readline()
Setting_temp = Setting_line.split()
trainingData = Setting_temp[1]

Setting_line = settingFp.readline()
Setting_temp = Setting_line.split()
model_description = Setting_temp[1]

Setting_temp = settingFp.readline()
Setting_temp = Setting_temp.split()
del Setting_temp[0]
for i in Setting_temp:
	if not i == 0:
		nodeList.append(int(i))

Setting_line = settingFp.readline()
Setting_temp = Setting_line.split()
numberOfIteration = int(Setting_temp[1])

Setting_line = settingFp.readline()
Setting_temp = Setting_line.split()
batchFlag = int(Setting_temp[1])

settingFp.close()

#Setting_line = settingFp.readline()
#Setting_temp = Setting_line.split()
#SmootingWindowSize_For_dP = int(Setting_temp[1])

inputPath = os.path.sep.join(["Dataset/", trainingData])

if batchFlag == 0:
	training.traning_Pdata(numberOfIteration, inputPath, featureList, nodeList, model_description, _testDataRate=0)

else:
	for idx in range(2):

		idx = idx + 6
		if(idx < 4):
			n = idx + 1
			node1 = 16 * n
			node2 = 4 * n
			nodeList = [node1, node2]
			model_description = 'P1_9-2layer(16-4-n={})-wo0-pattern-sim-3-6-7-9-10-11-13-15-16-20-21-24-25%'.format(int(n))

		else:
			n = idx - 3
			node1 = 16 * n
			node2 = 8 * n
			node3 = 4 * n
			nodeList = [node1, node2, node3]
			model_description = 'H101-105-126-128-P123-+-Meterial-3layer(16-8-4-n={})-25%'.format(int(n))

		inputPath = os.path.sep.join(["Dataset/", trainingData])
		numberOfIteration = 3

		training.traning_Pdata(numberOfIteration, inputPath, featureList, nodeList, model_description, _testDataRate=0.25)
