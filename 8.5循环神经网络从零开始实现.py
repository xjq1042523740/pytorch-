#!/usr/bin/env python
# coding: utf-8

# In[1]:



import torch
from d2l import torch as d2l
import math
from torch import nn
from torch.nn import functional as F


# In[2]:


batch_size,num_steps=32,35
train_iter,vocab=d2l.load_data_time_machine(batch_size,num_steps)


# In[3]:


F.one_hot(torch.tensor([0,2]),len(vocab))


# In[4]:


X=torch.arange(10).reshape((2,5))
#我们经常转换输入的维度，以便获得形状为 （时间步数，批量大小，词表大小）的输出。
#将时间步数放在第一位是为了方便使用矩阵编程，若批量大小在第一位则需要用两个for循环才能完成一个batch前向传播，
#∵每一个子序列的每一步都用到了前一步的hiddens状态，不转置无法一步到位
#后面的RNNModelScratch会用到
F.one_hot(X.T,28).shape


# In[5]:


#获取RNN模型参数
def get_params(vocab_size,num_hiddens,device):
    num_inputs=num_outputs=vocab_size
    
    def normal(shape):
        return torch.randn(size=shape,device=device)*0.01
    
    W_xh=normal((num_inputs,num_hiddens))
    W_hh=normal((num_hiddens,num_hiddens))
    b_h=torch.zeros(num_hiddens,device=device)
    W_hq=normal((num_hiddens,num_outputs))
    b_q=torch.zeros(num_outputs,device=device)
    
    params=[W_xh,W_hh,b_h,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# In[6]:


#初始化一个批量的RNN隐藏层的状态a
def init_rnn_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),)


# In[7]:


def rnn(inputs,state,params):
    W_xh,W_hh,b_h,W_hq,b_q=params
    H,=state
    outputs=[]
    for X in inputs:
        H=torch.tanh(torch.mm(X,W_xh)+torch.mm(H,W_hh)+b_h)
        Y=torch.mm(H,W_hq)+b_q
        outputs.append(Y)
    return torch.cat(outputs,dim=0),(H,)


# In[8]:


class RNNModelScratch:
    def __init__(self,vocab_size,num_hiddens,device,get_params,init_state,forward_fn):
        self.vocab_size,self.num_hiddens=vocab_size,num_hiddens
        self.params=get_params(vocab_size,num_hiddens,device)
        self.init_state,self.forward_fn=init_state,forward_fn
    
    def __call__(self,X,state):
        X=F.one_hot(X.T,self.vocab_size).type(torch.float32)
        return self.forward_fn(X,state,self.params)
    
    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hiddens,device)


# In[9]:


X = torch.arange(10).reshape((2, 5))
num_hiddens=512
net=RNNModelScratch(len(vocab),num_hiddens,d2l.try_gpu(),get_params,init_rnn_state,rnn)
state=net.begin_state(X.shape[0],d2l.try_gpu())
Y,new_state=net(X.to(d2l.try_gpu()),state)
Y.shape,len(new_state),new_state[0].shape


# In[ ]:




