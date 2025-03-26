import jieba
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time

######################################################开始建设项目##########################################################
##(1)、数据处理阶段##
##构建词典
def built_voca_dict():
    ##1、数据导入
    fname = "E:\pycharm专用文件\论文项目\唐诗三百首.txt"
    ##2、数据清理(视情况而定)
    cleaned_text = []
    for line in open(fname, "r"):
        if line.strip():  ##过滤掉空白行
            cleaned_text.append(line[:-2])  ##这里去掉每一行最后的两个换行字符"\n"

    ##3、分词
    all_sentences = []  ##暂存分词结果
    index_to_word = []  ##索引到词的映射
    word_to_index = {}  ##词到索引的映射
    for line in cleaned_text:
        words = jieba.lcut(line)  ####注意！！！：这里是lcut(),而不是cut()。lcut能将结果从迭代器里直接拿出来
        ##便于最后将句子转化为索引表示
        all_sentences.append(words)
        for word in words:
            if word not in index_to_word:
                index_to_word.append(word)
    for i, word in enumerate(index_to_word):
        word_to_index[word] = i

    ##4、将所有的语料进行索引表示(构建词典)
    corpus_index = []  ##语料的索引表示(构建词典)
    for sentence in all_sentences:
        temp_lst = []
        for word in sentence:
            temp_lst.append(word_to_index[word])

        ##在每行后面都添加一个空格
        temp_lst.append(word_to_index[" "])  ##这里空格的索引是4
        corpus_index.extend(temp_lst)  ##用空格的索引‘4’代替‘[]’符号来分隔每一行

    return index_to_word, word_to_index, len(index_to_word), corpus_index

##全局调用
index_to_word,word_to_index,word_len,corpus_index=built_voca_dict()


##(2)、构建数据集对象##
####编写数据集类####
class LyricsDataset():
    def __init__(self, corpus_index, num_chars):
        self.corpus_index = corpus_index  ##语料数据
        self.num_chars = num_chars  ##语料长度——————决定建模时一次取多少长度的语料数据
        self.word_count = len(corpus_index)  ##词的数量
        self.number = self.word_count // self.num_chars  ##句子的数量

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        start = min(max(idx, 0), self.word_count - self.num_chars - 2)  ##防止start超出范围
        x = self.corpus_index[start:start + self.num_chars]
        y = self.corpus_index[start + 1:start + self.num_chars + 1]  ##将x往右移动一位！！！
        ##例如x=[0, 4, 1, 2, 3]
        ##例如y=[4, 1, 2, 3, 4]
        return torch.tensor(x), torch.tensor(y)
        ##用来将列表 x 和 y 转换成 PyTorch 张量（Tensor）的操作。PyTorch 是一个流行的深度学习库，它使用张量作为基本的数据结构来进行各种数学运算。

##(3)、构建循环神经网络##
class TextGenerator(nn.Module):
    def __init__(self):
        super(TextGenerator,self).__init__()    ##这行代码确保了父类nn.Module被正确初始化，从而使新类能够继承和使用nn.Module的所有功能。
        ##初始化词嵌入层
        self.ebd=nn.Embedding(num_embeddings=word_len,embedding_dim=128)
        ##初始化循环网络
        self.rnn=nn.RNN(input_size=128,hidden_size=128,num_layers=1)
        ##初始化输出层,预测的标签的数量为词典中词的总数(比如这里的word_len)
        self.out=nn.Linear(128,word_len)

    ##构建前向计算过程，输入为当前时刻的输入和上一时刻的隐藏状态
    def forward(self,inputs,hidden):

        ##此处embed的形状为[1,num_chars=5,128]
        embed=self.ebd(inputs)

        ##正则化

        ##送入循环网络层
        ##output里面包含的是每一个时刻的输出
        output,hidden=self.rnn(embed.transpose(0,1),hidden)

        ##将output送到全连接层，进行最后的结果输出
        output=self.out(output)

        return output,hidden

    ##初始化隐藏层
    def init_hidden(self):
        return torch.zeros(1,1,128)

##(4)、训练模型
def train():
    ##构建词典
    index_to_word, word_to_index, word_len, corpus_index = built_voca_dict()
    ##数据集
    lyrics=LyricsDataset(corpus_index,32)
    ##模型构建
    model=TextGenerator()
    ##损失函数(交叉熵损失)
    criterion=nn.CrossEntropyLoss()
    ##优化方法(自适应的梯度下降算法————Adam)
    optimer=optim.Adam(model.parameters(),lr=1e-3)  ##传入参数和学习率
    ##训练轮数
    epoch=200   ##训练200次
    ##迭代打印
    item_num=300

    ##开始训练
    for epoch_idx in range(epoch):
        ##初始化数据加载器
        dataloader=DataLoader(lyrics,shuffle=True)
        ##训练时间
        start=time.time()
        ##迭代次数
        item_num=0
        ##迭代损失
        total_loss=0.0

        for x,y in dataloader:

            ##初始化隐藏状态
            hidden=model.init_hidden()
            ##送入网络中进行计算
            output,_=model(x,hidden)
            ##计算损失output、y的形状应分别为[32,7683]、[32]
            loss=criterion(output.squeeze(),y.squeeze())      ##注意：这里要把'output'和‘y’用squeeze()函数来分别降成二维、一维的，否则会报错！
            ##梯度清零
            optimer.zero_grad()
            ##反向传播
            loss.backward()
            ##参数更新
            optimer.step()

            item_num+=1
            total_loss+=loss.item()

            infor=f"'epoch:'{epoch_idx};'loss:'{total_loss/item_num};'time:'{time.time()-start}"
            print(infor)

    ##最后将训练好的模型进行保存
    torch.save(model.state_dict(),r"E:\pycharm专用文件\论文项目\RNN\text_generator")    ##模型的保存路径

##进行训练
##train()

##(5)、预测函数
def predict(start_word,sentence_length):
    ##构建词典
    index_to_word, word_to_index, word_len, corpus_index = built_voca_dict()
    ##加载模型
    model=TextGenerator()
    model.load_state_dict(torch.load(r"E:\pycharm专用文件\论文项目\RNN\text_generator"))
    model.eval()    ##模型评估
    ##初始化隐藏状态
    hidden=model.init_hidden()

    ##首先，将start_word转化为索引
    word_idx=word_to_index[start_word]
    generate_sentence=[word_idx]
    for _ in range(sentence_length):
        output,hidden=model(torch.tensor([[word_idx]]),hidden)
        ##选择概率最大的词作为预测值输出
        word_idx=torch.argmax(output)
        generate_sentence.append(word_idx)

    ##最后将索引序列generate_sentence转化为词序列
    predict_words=''
    for idx in generate_sentence:
        predict_words+=index_to_word[idx]
    return predict_words

print('请输入你想输入的词语：')
start_word=input()
print('请设置你想得到的返回语句的最大长度：')
sentence_length=int(input())
infor=predict(start_word,sentence_length)
print(infor)