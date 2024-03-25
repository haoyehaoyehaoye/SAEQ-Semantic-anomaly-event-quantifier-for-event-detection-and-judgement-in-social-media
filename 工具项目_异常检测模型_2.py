from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoModel, AutoTokenizer, BertTokenizer,BertModel
import csv
from transformers import BertModel, BertTokenizer
import warnings
import emoji
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
class Config():
    def __init__(self,batch_size):
        self.hidden_size=1024  #Bert原始的输出
        self.result=2        #最后输出的
        self.conv_length=87  #计算卷积的结果,直接看forward里面的print结果
        self.batch_size=batch_size
#定义卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv11 = nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)  #这就是残差连接 用1*1的卷积保证维度
    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)  #残差
        out = out1 + out2
        return out
class BertClassify(nn.Module):
    def __init__(self,config):
        super(BertClassify, self).__init__()
        #self.bert = AutoModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        #self.bert = AutoModel.from_pretrained(model)
        self.dropout = nn.Dropout(0.3)
        self.bert = BertModel.from_pretrained('bert-large-uncased')
        self.linear_1 = nn.Linear(config.hidden_size,256)
        self.linear_2 = nn.Linear(256,64)
        self.linear_3 = nn.Linear(64,config.result)
        self.soft_layer = nn.Softmax(dim=1)  #dim1的时候才有效果，dim=0没效果也没报错
    def forward(self, X):
        input_ids = X[0]
        outputs = self.bert(input_ids=input_ids)
        pooled_output = outputs.pooler_output  # 获取 [CLS] 标记的输出
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.linear_1(pooled_output)
        pooled_output = self.linear_2(pooled_output)
        logits = self.linear_3(pooled_output)
        probabilities = self.soft_layer(logits)
        return probabilities
class MyDataset(Dataset):
    def __init__(self, sentences, labels=None, with_labels=True):
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased') #在dataset中加载分词器，将文本转换为数值,如果是下载的，就是.txt
        self.with_labels = with_labels
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sent = self.sentences[index]
        #print(sent)
        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',  # 填充到最大值
                                      truncation=True,  # 截断到最大值
                                      max_length=128,# 最大值
                                      return_tensors='pt')  # 返回pytorch类型

        token_ids = encoded_pair['input_ids'].squeeze(0)  
        # # input_ids为将文本转化为数字，前后加一个特殊字符101(开始字符)，102(结束字符)，0(填充字符)
        #print(token_ids)
        if self.with_labels:
            label = self.labels[index]
            #print(label)
            return token_ids,torch.tensor(label)  #label是个一维列表，不能直接传入
        else:
            return token_ids.unsqueeze(0)
def getDataCsv():
   #file='./drive/MyDrive/chinese_roberta_wwm_large_ext_pytorch/rec_data.csv'  #这是谷歌上面的地址
   sentence_list=[]
   labels_list=[]
   file = './GPT最终实验代码/异常事件检测数据集.xlsx'
   dataframe = pd.read_excel(file)
   for index, row in dataframe.iterrows():
       labels_list.append(row[0])
       text_without_emoji = emoji.demojize(row[1])  #特殊表情包转换为文本
       text_without_emoji = text_without_emoji.replace('_', "")
       text_without_emoji = text_without_emoji.replace(':', "")
       text_without_emoji = text_without_emoji.replace('#', "")
       #print(text_without_emoji)
       sentence_list.append(text_without_emoji)
   #print(sentence_list)
   return sentence_list,labels_list

class MyLoss(nn.Module):    #自己写loss function
    def __init__(self):
        super().__init__()
    def forward(self, pred, true):
        loss = torch.log(torch.cosh(pred-true))
        return torch.sum(loss)
def train_model ():
    forward_loss=10000
    flag_stop=0
    my_module.train()
    train = DataLoader(dataset=MyDataset(sentences= sentence_list[0:400:1], labels= labels_list[0:400:1],with_labels=True), batch_size=batch_size, shuffle=True, num_workers=0)
    for i in range(epoches):  #循环每轮
        sum_loss=0  #记录每轮的总损失
        for data in enumerate(train):
            optimizer.zero_grad()
            # print(data[0]) #data[0]记录的是第几批数据。 比如一共2w个数据，batch为64，所以每轮有312个batch. 
            # data[1]存放的是数据，包含内容与标签两个部分
            sentence,labels=data[1]
            sentence=sentence.to(device=devices)
            labels=labels.to(device=devices)
            # print(sentence.size())
            # print(labels.size())
            pred = my_module([sentence]).to(device=devices) #[]升维,要求三维
            #print(pred,labels)
            loss = loss_fn(pred,labels)
            #sum_loss += loss.item()
            sum_loss += loss
            loss.backward()
            
            optimizer.step()
            #print(f'完成第{data[0]}批次，损失为{loss.item()}')
            print(f'完成第{data[0]}批次')
        print('====')
        print("{}轮总损失为{}".format(i,sum_loss))
        if(forward_loss-sum_loss<0.05):
            print("第{}次epoch出现低增长".format(i))
            flag_stop=flag_stop+1
        #if(flag_stop>3 or i>=20 ):
        if(i>=30 ):
            print("此时停止")
            torch.save(my_module,"./model/row_V2_400.pth")
            break
        forward_loss = sum_loss
        torch.save(my_module,"./model/row_V2_400.pth")
def test_model ():
    forward_loss=0
    flag_stop=0
    #my_module.eval()
    my_module.to(device=devices)
    sentences_test=sentence_list[400:500:1]
    labels_test=labels_list[400:500:1]
    test_data = MyDataset(sentences=sentences_test,with_labels=False)  #调用dataset，只负责文本数据转换类型，不使用dataloader
    count=0
    for i in range(len(sentences_test)):
        #print(test_data.__getitem__(i).shape)  本身是[128]  .unsqueeze(0)升维  要求三维
        #print(test_data.__getitem__(i).unsqueeze(0).unsqueeze(0).shape)
       # print(sentences_test[i])
        #print(test_data.__getitem__(i))
        #tokens_text = BertTokenizer.from_pretrained('bert-base-uncased').convert_ids_to_tokens(test_data.__getitem__(i).squeeze().tolist()) #把标识转回字符串
        #print(tokens_text)
        pred = my_module(test_data.__getitem__(i).unsqueeze(0).to(device=devices)) 
        print(pred.data)
        pre_label = max(enumerate(pred.data[0]), key=lambda x: x[1])[0]
        print(pre_label,labels_test[i])
        if(pre_label ==labels_test[i] ):
            count+=1
    print(count/100)


if __name__ == '__main__':
    warnings.simplefilter("ignore")
    #全局通用设置
    torch.cuda.empty_cache() #释放显存
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #devices = torch.device('cpu')
    print(devices)
    batch_size = 64
    config=Config(batch_size=batch_size)
    setup_seed(50)
    # 所有的样本循环多少次
    epoches = 50
    sentence_list,labels_list = getDataCsv()
    print(f'共有数据{len(sentence_list)}')
    torch.cuda.empty_cache() #释放显存
    pth='./model/row_V2_400.pth'
    #my_module = BertClassify(config).to(device=devices)  #原始模型
    my_module = torch.load(pth).to(device=devices)   #训练的模型
    loss_fn = nn.CrossEntropyLoss()   #交叉损失熵

    loss_fn = loss_fn.to(device=devices)
    optimizer = torch.optim.SGD(params=my_module.parameters(),lr=0.0001)
    train_model()
    #test_model()   #0.91的准确率







