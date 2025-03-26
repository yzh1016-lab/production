import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

##载入数据
data_path=r'C:\Users\DELL\Desktop\data.xlsx'
def get_data(data_path,train_size,seq_len):
    data_df=pd.read_excel(data_path)
    data_lst=data_df.values.tolist()
    data_lst = data_lst[::-1]
    X=[]
    Y=[]
    for i in range(len(data_lst)-seq_len):
        X.append(data_lst[i:i+seq_len])
        Y.append(data_lst[i+seq_len])
    trainX = X[:int(len(X) * train_size)]
    trainY = Y[:int(len(Y) * train_size)]
    testX = X[int(len(X) * train_size):]
    testY = Y[int(len(Y) * train_size):]
    trainX=np.array(trainX).reshape(len(trainX),seq_len,-1)##一定不要忘记reshape()一下！！！
    trainY=np.array(trainY)
    testX=np.array(testX).reshape(len(testX),seq_len,-1)
    testY=np.array(testY)
    return trainX,trainY,testX,testY

##网络搭建
class CNN_LSTM(nn.Module):
    def __init__(self,conv_input,input_size,hidden_size,num_layers,output_size):
        super().__init__()
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.conv=nn.Conv1d(conv_input,conv_input,1)
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x=self.conv(x)
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        m0=torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        output,_=self.lstm(x,(h0,m0))
        ##每次只输出最后一个时间步所得到的结果
        out=self.fc(output[:,-1,:])##别忘记加逗号！！！！！！
        return out

##调参
trainX,trainY,testX,testY=get_data(data_path,0.8,10)
Xtrain=torch.tensor(trainX)
Ytrain=torch.tensor(trainY)
epochs=100
learn_rate=1e-3
conv_input=10
input_size=1
hidden_size=128
num_layers=2
output_size=1

##训练网络
def train():
    model=CNN_LSTM(conv_input,input_size,hidden_size,num_layers,output_size)
    #选用Adam梯度下降方式
    optimizer=torch.optim.Adam(model.parameters(),lr=learn_rate)
    #选用MSEloss损失函数
    criterion=nn.MSELoss()
    train_loss=[]
    for epoch in range(epochs):
        #训练声明
        model.train()
        #每次训练前，进行梯度清零，0
        optimizer.zero_grad()
        #数据转为float类型
        Xtrain = Xtrain.float()
        Ytrain=Ytrain.float()
        #前向传播
        out=model.forward(Xtrain)
        #计算误差
        loss=criterion(out,Ytrain[:])
        train_loss.append(loss)
        #计算梯度
        loss.backward()
        #反向传播
        optimizer.step()
        #评价声明
        model.eval()
        if epoch%10==0:
            print(f'第{epoch}次训练，误差为：{loss}')
    torch.save(model.state_dict(),r'E:\pycharm专用文件\论文项目\CNN_LSTM时间序列分析\CNN_LSTM-时间序列分析')

# train()

##预测
#创建模型实例
model=CNN_LSTM(conv_input,input_size,hidden_size,num_layers,output_size)
#载入模型
model.load_state_dict(torch.load(r'E:\pycharm专用文件\论文项目\CNN_LSTM时间序列分析\CNN_LSTM-时间序列分析'))
Xtest=torch.tensor(testX)
#一定不要忘记转为float类型！！！
Xtest=Xtest.float()
#前向传播
predict=model.forward(Xtest)
predict=predict.detach().numpy()
predict=predict.tolist()##不转为float类型会报错
testY=testY.tolist()
pre=[i[0] for i in predict]##预测值
t_y=[j[0] for j in testY]##真实值

##画图
plt.title('CNN_LSTM--时间序列分析')
x=[i for i in range(len(predict))]
plt.plot(x,pre,marker='o',markersize=1,label='predict')
plt.plot(x,t_y,marker='x',markersize=1,label='True_Y')
plt.legend()
plt.show()


