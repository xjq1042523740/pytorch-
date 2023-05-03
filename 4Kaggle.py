#!/usr/bin/env python
# coding: utf-8

# In[1]:


import hashlib
import os
import tarfile
import zipfile
import requests

DATA_HUB=dict()
DATA_URL='http://d2l-data.s3-accelerate.amazonaws.com/'


# In[2]:


def download(name,cache_dir=os.path.join('..','data')):
    #下载一个DATA_HUB中的文件，返回本地文件名
    assert name in DATA_HUB,f"{name}不在{DATA_HUB}中"
    url,sha1_hash=DATA_HUB[name]
    os.makedirs(cache_dir,exist_ok=True)  #在当前目录下创建文件夹
    fname=os.path.join(cache_dir,url.split('/')[-1])  #将文件夹与下载的数据文件名拼接起来
    if os.path.exists(fname):  #如果当前路径下存在fname这个文件
        sha1=hashlib.sha1()   #哈希算法的sha1算法函数
        with open(fname,'rb') as f:  #打开该文件夹
            while True:
                data=f.read(1048576)  #一次读取1024*1024即1M大小的文件
                if not data:   #如果已近没有数据了就跳出
                    break
                sha1.update(data)   #数据更新哈希值
        if sha1.hexdigest()==sha1_hash:  #如果最后的哈希值和完整数据的哈希值相同则不用再下载一次
            return fname
    print(f'正在从{url}下载{fname}...')  
    r=requests.get(url,stream=True,verify=True)  
    #如果没有该文件，或该文件不完整，就下载该文件，stream=True，verify=True表示从该网站分段下载，通过验证就下载一次
    with open(fname,'wb') as f:
        f.write(r.content)
    return fname


# In[3]:


def download_extract(name,folder=None):
    #下载并解压zip/tar文件
    fname=download(name)
    base_dir=os.path.dirname(fname)   #base_dir：下载文件所在的目录
    data_dir,ext=os.path.splitext(fname) #data_dir:解压缩文件后的文件夹的完整路径。ext：下载文件的扩展名
    if ext=='.zip':
        fp=zipfile.ZipFile(fname,'r')
    elif ext in ('.tar','.gz'):
        fp=tarfile.open(fname,'r')
    else:
        assert False,f"只有zip/tar文件可以被解压"
    fp.extractall(bases_dir)
    #它的作用是将压缩文件中的所有文件提取到指定的目录(bases_dir)中
    return os.path.join(base_dir,folder) if folder else data_dir

def download_all():
    for name in DATA_HUB:
        download(name)


# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


# In[5]:


#数据预处理
#将要下载的数据库网址，哈希值存入字典
DATA_HUB['kaggle_house_train']=(DATA_URL+'kaggle_house_pred_train.csv',
                               '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test']=(DATA_URL+'kaggle_house_pred_test.csv',
                              'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')


# In[6]:


#下载数据
train_data=pd.read_csv(download('kaggle_house_train'))
test_data=pd.read_csv(download('kaggle_house_test'))


# In[7]:


print(train_data.shape)
print(test_data.shape)


# In[8]:


print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]])


# In[9]:


#将测试集和训练集合并在一起
all_features=pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))


# In[10]:


#筛选出数值特征，并将他们标准归一化
numeric_features=all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features]=all_features[numeric_features].apply(lambda x:
                                                                   (x-x.mean())/(x.std()))
all_features[numeric_features]=all_features[numeric_features].fillna(0)


# In[11]:


#将离散值根据one-hot编码
all_features=pd.get_dummies(all_features,dummy_na=True)
all_features.shape


# In[12]:


#转化为tensor张量
n_train=train_data.shape[0]
train_features=torch.tensor(all_features[:n_train].values,dtype=torch.float32)
test_features=torch.tensor(all_features[n_train:].values,dtype=torch.float32)
train_labels=torch.tensor(train_data.SalePrice.values.reshape(-1,1),dtype=torch.float32)


# In[13]:


loss=nn.MSELoss()
in_features=train_features.shape[1]

def get_net():
    net=nn.Sequential(nn.Linear(in_features,1))
    return net

def log_rmse(net,features,labels):
    #log后的均方误差
        clipped_preds=torch.clamp(net(features),1,float('inf'))
        rmse=torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
        return rmse.item()


# In[14]:


def train(net,train_features,train_labels,test_features,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
    train_ls,test_ls=[],[]
    train_iter=d2l.load_array((train_features,train_labels),batch_size)
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X,y in train_iter:
            optimizer.zero_grad()
            y_hat=net(X)
            l=loss(y_hat,y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls 


# In[15]:


def get_k_fold_data(k,i,X,y):
    assert k>1
    fold_size=X.shape[0]//k
    X_train,y_train=None,None
    for j in range(k):
        idx=slice(j*fold_size,(j+1)*fold_size)
        X_part,y_part=X[idx,:],y[idx]
        if j==i:
            X_valid,y_valid=X_part,y_part
        elif X_train is None:
            X_train,y_train=X_part,y_part
        else:
            X_train=torch.cat([X_train,X_part],0)
            y_train=torch.cat([y_train,y_part],0)
    return X_train,y_train,X_valid,y_valid


# In[16]:


def k_fold(k,X_train,y_train,num_epochs,learning_rate,weight_decay,batch_size):
    train_l_sum,valid_l_sum=0,0
    for i in range(k):
        data=get_k_fold_data(k,i,X_train,y_train)
        net=get_net()
        train_ls,valid_ls=train(net,*data,num_epochs,learning_rate,weight_decay,batch_size)
        train_l_sum+=train_ls[-1]  #将最后一个epoch调整完权重后的均方误差加起来
        valid_l_sum+=valid_ls[-1]
        if i==0:
            d2l.plot(list(range(1,num_epochs+1)),[train_ls,valid_ls],xlabel='epoch',ylabel='rmse',
                    xlim=[1,num_epochs],legend=['train','valid'],yscale='log')
            
        print(f'折{i+1},训练log rmse {float(train_ls[-1]):f}\t',
             f'验证log rmse {float(valid_ls[-1]):f}')
    return train_l_sum/k,valid_l_sum/k


# In[17]:


k,num_epochs,lr,weight_decay,batch_size=5,100,5,0,64
train_l,valid_l=k_fold(k,train_features,train_labels,num_epochs,lr,weight_decay,batch_size)
print(f'{k}折验证：平均训练log rmse：{float(train_l):f}',
     f'平均验证 log rmse：{float(valid_l):f}')


# In[20]:


def train_and_pred(train_features,test_features,train_labels,test_data,num_epochs,lr,weight_decay,batch_size):
    net=get_net()
    train_l,_=train(net,train_features,train_labels,None,None,num_epochs,lr,weight_decay,batch_size)
    d2l.plot(np.arange(1,num_epochs+1),[train_l],xlabel='epoch',ylabel='log rmse',xlim=[1,num_epochs],yscale='log')
    print(f'训练log均方误差为：{float(train_l[-1]):f}')
    preds=net(test_features).detach().numpy()   #方便后面转化为csv的格式
    test_data['SalePrice']=pd.Series(preds.reshape(1,-1)[0])
    submission=pd.concat([test_data['Id'],test_data['SalePrice']],axis=1)
    submission.to_csv('submission.csv',index=False)


# In[21]:


train_and_pred(train_features,test_features,train_labels,test_data,num_epochs,lr,weight_decay,batch_size)


# In[ ]:




