
# coding: utf-8
# 批量归一化(使用指数加权平均数和方差)
#指数加权平均：以气温位例：moving mean 表示前t-1天的加权平均气温值，mean表示第t天的气温值
#使用批量归一化的目的是为了训练的时候防止梯度消失和梯度爆炸阻碍训练，影响训练速度，
#
#测试的时候为了参数和训练的参数尽量保持一致我们就用整个训练集的加权平均参数来代替平均值和方差，这样才会得到理想的结果
#就相当于把他看成一个参数，训练的时候这个参数定为A，而测试的时候改了这个参数为B，最后的结果不会是我们想要的值。
#将这个批量归一化看成一个非线性的激活函数的一部分，如果改了他的参数就相当于训练的激活函数和测试的激活函数不一样。
# In[13]:


import torch
from torch import nn
from d2l import torch as d2l

def batch_normal(X,gamma,beta,moving_mean,moving_var,eps,momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        #预测模式（评估模式）直接使用传入的(训练好的)移动平均所得的均值和方差
        X_hat=(X-moving_mean)/torch.sqrt(moving_var+eps)
    else:
        #训练模式下，需要调整移动平均所得的均值和误差
        assert len(X.shape) in (2,4)     #全连接层输入是二维的，卷积层输入是4维的
        if len(X.shape)==2:
           #计算本批次的均值。方差以便后面指数加权使用
            mean=X.mean(dim=0)
            var=((X-mean)**2).mean(dim=0)
        else:
            mean=X.mean(dim=(0,2,3),keepdim=True)   #保持维度以便后面求方差用(自动广播)
            var=((X-mean)**2).mean(dim=(0,2,3),keepdim=True)
        #训练模式下不断更新指数加权平均值和方差，最后得到整个训练集的加权均值和方差，以便后面测试的时候使用
        #训练的时候由于得不到整个训练集的指数加权均值和方差，所以使用批量的均值训练
        #测试的时候使用训练集的均值而不是用测试集的均值是因为要保证训练和测试的参数一致
        X_hat=(X-mean)/torch.sqrt(var+eps)
        moving_mean=momentum*moving_mean+(1.0-momentum)*mean
        moving_var=momentum*moving_var+(1.0-momentum)*var
    Y=gamma*X_hat+beta
    return Y,moving_mean.data,moving_var.data


# In[14]:


class BatchNorm(nn.Module):
    def __init__(self,num_features,num_dims):
        super(BatchNorm,self).__init__()
        assert num_dims in (2,4)
        if num_dims==2:
            shape=(1,num_features)
        else:
            shape=(1,num_features,1,1)
        #需要训练的参数
        self.gamma=nn.Parameter(torch.ones(shape))
        self.beta=nn.Parameter(torch.zeros(shape))
        #不需要训练的非模型参数
        self.moving_mean=torch.zeros(shape)
        self.moving_var=torch.ones(shape)
    
    def forward(self,X):
        #判断X是不是在内存（存储在cpu上）上，如果在显存（存储在GPU上）上，则将moving_mean,moving_var移动到显存上去
        #gamma和beta不用在这里移动是因为他是parameters参数，调用net.to(device)的时候会移动过去
        if X.device!=self.moving_mean.device:
            self.moving_mean=self.moving_mean.to(X.device)
            self.moving_var=self.moving_var.to(X.device)
        #保存更新过的moving_mean、moving_var
        Y,self.moving_mean,self.moving_var=batch_normal(X,self.gamma,self.beta,
                                                        self.moving_mean,self.moving_var,eps=1e-5,momentum=0.9)
        return Y


# In[15]:


#Lenet
net=nn.Sequential(nn.Conv2d(1,6,kernel_size=5,stride=1),BatchNorm(6,num_dims=4),nn.Sigmoid(),
                  nn.AvgPool2d(kernel_size=2,stride=2),
                  nn.Conv2d(6,16,kernel_size=5),BatchNorm(16,num_dims=4),nn.Sigmoid(),
                  nn.AvgPool2d(kernel_size=2,stride=2),
                  nn.Flatten(),
                  nn.Linear(256,120),BatchNorm(120,num_dims=2),nn.Sigmoid(),
                  nn.Linear(120,84),BatchNorm(84,num_dims=2),nn.Sigmoid(),
                  nn.Linear(84,10))


# In[16]:


lr,num_epochs,batch_size=1.0,10,256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr=lr,device=d2l.try_gpu())


# In[12]:




