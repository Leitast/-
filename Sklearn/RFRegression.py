import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import xlrd
import xlwt
import random
import copy
from scipy.stats import pearsonr
import csv
import os
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict

scores = []

member = 0

###########1.读取数据部分##########
#载入数据并且打乱数据集
def load_data(StartPo,EndPo,TestProportion,FeatureNum,Shuffle,FilePath):
    # 样本起始行数，结束行数，测试集占总样本集比重,特征数，是否打乱样本集     #如果Testproportion为0或1就训练集=测试集
    # 打开excel文件
    workbook = xlrd.open_workbook(str(FilePath))       #excel路径
    sheet = workbook.sheet_by_name('Sheet1')             #sheet表
    Sample = []#总样本集
    train = []#训练集
    test = []#测试集
    TestSetSphere = (EndPo-StartPo+1)*TestProportion  #测试集数目
    TestSetSphere = int(TestSetSphere)#测试集数目
    #获取全部样本集并打乱顺序
    for loadi in range(StartPo-1,EndPo):
        RowSample = sheet.row_values(loadi)
        Sample.append(RowSample)
    if Shuffle == 1:  #是否打乱样本集
        random.shuffle(Sample)  #如果shuffle=1，打乱样本集
    #如果Testproportion为0就训练集=测试集
    if TestProportion == 0 or TestProportion == 1:
        TrainSet = np.array(Sample)          #变换为array
        TestSet = np.array(Sample)
    else:
        #设置训练集
        for loadtraina in Sample[:(EndPo-TestSetSphere)]:
            GetTrainValue = loadtraina
            train.append(GetTrainValue)
        #设置测试集
        for loadtesta in range(-TestSetSphere-1,-1):
            GetTestValue = Sample[loadtesta]
            test.append(GetTestValue)
        #变换样本集
        TrainSet = np.array(train)                  #变换为array
        TestSet = np.array(test)
   #分割特征与目标变量
    x1 , y1 = TrainSet[:,:FeatureNum] , TrainSet[:,-1]
    x2 , y2 = TestSet[:,:FeatureNum] , TestSet[:,-1]
    return x1 , y1 , x2 , y2


###########2.回归部分##########
def regression_method(model):
    model.fit(x_train,y_train)
    #准确度得分
    score = model.score(x_test, y_test)
    #print("score:"+str(score))
    resulttrain = model.predict(x_train)
    result = model.predict(x_test)
   # print(result)
   # print("Pre")
   # print(y_test)
   # print("Train")
   # print(resulttrain)
   # print("Trainsssssssssssssss")
   # print(x_train)
    ResidualSquare = (result - y_test)**2     #计算残差平方
    #print(member)
    #ZhengShu =abs(result-y_test)
    #Rzhengshu = sum(ZhengShu)
    RSS = sum(ResidualSquare)   #计算残差平方和
    MSE = np.mean(ResidualSquare)       #计算均方差
   # MAE = np.mean(ZhengShu)
    num_regress = len(result)   #回归样本个数
    pearson_R = pearsonr(y_test,result)
    RMSE  = math.sqrt(MSE)
    print(f'n={num_regress}')
    print(f'R^2={score}')
    print(f'Pearson_R={pearson_R}')
    print(f'MSE={MSE}')
    print(f'RMSE={RMSE}')
    print(f'RSS={RSS}')
    #print(f'MAE={MAE}')
############绘制折线图##########
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('RandomForestRegression R^2: %f'%score)
    plt.legend()        # 将样例显示出来
    plt.show()
    return result , resulttrain

def IncMSE(MSE,x_test, y_test,FeatureNum,Set_Times,model):    #获取MSE，x测试集，y测试集，特征数，随机求IncMSE次数，模型（随机森林）
    x_MSE = copy.deepcopy(x_test)      #深拷贝不破坏原列表
    y_MSE = copy.deepcopy(y_test)
    TestNum = len(y_MSE)
    #########多次生成随机数，多次计算IncMSE（由于随机有不确定性，所以要多次随机）
    IncMSE_Set = []
    IncMSE_Times = 1
    while IncMSE_Times <= Set_Times:     #多次生成随机数，多次计算IncMSE（由于随机有不确定性，所以要多次随机）
        IncMSE_x = []
        for i in range(0,FeatureNum):
            MSE_Replace = np.random.random(TestNum)
            x_MSE[:,i] = MSE_Replace           #替换第i个特征
            MSE_Score = model.score(x_MSE,y_MSE)
            MSE_Result = model.predict(x_MSE)
            MSE_ResidualSquare = (MSE_Result - y_MSE)**2   #计算残差平方
            MSE_RSS = sum(MSE_ResidualSquare)   #计算残差平方和
            MSE_MSE = np.mean(MSE_ResidualSquare)   #计算均方差
            IncMSE = MSE_MSE - MSE
            IncMSE_x.append(IncMSE)
            x_MSE = copy.deepcopy(x_test)   #复原原特征，深拷贝不破坏原列表
        IncMSE_Set.append(IncMSE_x)          #多次计算IncMSE后的数据
        IncMSE_Times += 1
    IncMSE_SetArray = np.array(IncMSE_Set)    #变换为array
    ########计算每个特征的IncMSE平均数########
    X_IncMSE_Average = []
    for j in range(0,FeatureNum):
        X_IncMSE_Set = IncMSE_SetArray[:,j]
        X_IncMSE = np.mean(X_IncMSE_Set)       #求多次IncMSE的平均值（由于随机有不确定性，所以要多次随机）
        X_IncMSE_Average.append(X_IncMSE)
    X_IncMSE_Average_Sum = sum(X_IncMSE_Average)
    ########计算每个特征的IncMSE平均数的百分比########
    print('IncMSE:')
    for k in range(0,FeatureNum):
        X_Percent = X_IncMSE_Average[k]/X_IncMSE_Average_Sum       #计算每个特征IncMSE的百分比
        print(f'    x{k+1} = {X_IncMSE_Average[k]}   {X_Percent*100}%')        #输出各特征的IncMSE的平均数与其百分比

##########3.绘制验证散点图########
def scatter_plot(TureValues,PredictValues):
    #设置参考的1：1虚线参数
    print("PredictValues"+PredictValues)
    xxx = [-0.5,1.5]
    yyy = [-0.5,1.5]
    #绘图
    plt.figure()
    plt.plot(xxx , yyy , c='0' , linewidth=1 , linestyle=':' , marker='.' , alpha=0.3)#绘制虚线
    plt.scatter(TureValues , PredictValues , s=20 , c='r' , edgecolors='k' , marker='o' , alpha=0.8)#绘制散点图，横轴是真实值，竖轴是预测值
    plt.xlim((0,1))   #设置坐标轴范围
    plt.ylim((0,1))
    plt.title('RandomForestRegressionScatterPlot')
    plt.show()


###########4.预设回归方法##########
####随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=400)   #esitimators决策树数量


########5.设置参数与执行部分#############
#设置数据参数部分
x_train , y_train , x_test , y_test = load_data(2,98234,0.2,5,1,r'D:\workspace\PyCharmProject\yaoy\POI\20200202.xlsx')   #行数以excel里为准
#起始行数2，结束行数121，训练集=测试集，特征数量17,不打乱样本集
y_pred , y_trainpred= regression_method(model_RandomForestRegressor)        #括号内填上方法，并获取预测值

#f1 = open('new.csv', 'w')

# output_data = os.path.abspath(r'.\0430\fcn_beautiful0430.csv')
#
# with open(output_data, mode='w', encoding='utf-8', newline='') as f:
#     writer = csv.writer(f)
# num1 = 0
# num2 = 0
# for i in y_trainpred:
#     num1 = num1 + 1
#     print(i)
#
# print("分隔符")

# for i in y_pred:
#     num2 = num2 + 1
#     print(i)

# print(num1)
# print(num2)
# #重要性得分
importance = model_RandomForestRegressor.feature_importances_
# #indices = np.argsort(importance)[::-1]
print("x的重要性:"+str(x_train.shape[1]))
for f in range(x_train.shape[1]) :
    #print(importance[indices[f]])
    print(importance[f])

# print('—————————————————————————————————')
# IncMSE(y_pred,x_test,y_test,13,50,model_RandomForestRegressor)
#
# scatter_plot(y_test,y_pred)  #生成散点图
#
# print('--------------------------------------------------------------')
# r = model_RandomForestRegressor.fit(x_train, y_train)
# acc = r2_score(y_test, model_RandomForestRegressor.predict(x_test))
# for i in range(13) :
#     x_t = x_test.copy()
#     np.random.shuffle(x_t[:,i])
#     shuff_acc = r2_score(y_test, model_RandomForestRegressor.predict(x_t))
#     scores.append((acc-shuff_acc)/acc)
#
# for score in scores:
#     print(score)

