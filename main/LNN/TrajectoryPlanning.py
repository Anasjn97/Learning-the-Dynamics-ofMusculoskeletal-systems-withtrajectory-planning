# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 13:24:35 2021

@author: Anas Jnini
"""

from torchdyn.models import *
from torchdyn.datasets import *
from torchdyn import *
import scipy.io as sio
import numpy as np
import math

import numpy as np
import time
import scipy.io as sio


from torch.autograd import grad
import torch.utils.data as data
#Load The model learned using Jax and initialize the Torch NN

device = torch.device("cpu")

modelm=np.load('params.npy',allow_pickle=True)

l1=np.asarray(modelm[0][0])
w1=torch.nn.parameter.Parameter(torch.tensor(np.transpose(l1)),requires_grad=True)

l2=np.asarray(modelm[0][1])
b1=torch.nn.parameter.Parameter(torch.tensor(np.transpose(l2)),requires_grad=True)

l3=np.asarray(modelm[2][0])
w2=torch.nn.parameter.Parameter(torch.tensor(np.transpose(l3)),requires_grad=True)

l4=np.asarray(modelm[2][1])
b2=torch.nn.parameter.Parameter(torch.tensor(np.transpose(l4)),requires_grad=True)

l5=np.asarray(modelm[4][0])
w3=torch.nn.parameter.Parameter(torch.tensor(np.transpose(l5)),requires_grad=True)

l6=np.asarray(modelm[4][1])
b3=torch.nn.parameter.Parameter(torch.tensor(np.transpose(l6)),requires_grad=True)

#%% Import Jax Neural Network and Initialize the Neural Network in Pytorch
hdim=128
L1=nn.Linear(4,hdim)
L1.weight=w1
L1.bias=b1


L2=nn.Linear(hdim,hdim)
L2.weight=w2
L2.bias=b2


L3=nn.Linear(hdim,1)
L3.weight=w3
L3.bias=b3





net1 = nn.Sequential(
            L1,
            nn.Softplus(),
            L2,
            nn.Softplus(),
            L3)

pi=np.pi
v=np.asarray([[-0.15,pi/2+0.4,0,0],[0.1,pi/2+0.6,0,0],[0.3,pi/2+0.3,0,0],[0.4,pi/2-0.05,0,0],[0.35,pi/2-0.45,0,0],[0.05,pi/2-0.45,0,0],[-0.20,pi/2-0.35,0,0],[-0.35,pi/2-0.05,0,0]])  
  
trajectory=[]




class LNN2(nn.Module):
    """Lagrangian Neural ODE

    :param net: function parametrizing the vector field.
    :type net: nn.Module
    """
    def __init__(self, net,g,i):
        super().__init__()
        self.net = net
        self.g=g #Target position
        self.i=i
    def forward(self, x):
        self.n = n = x.shape[1]//2
        bs = x.shape[0]
        x = x.requires_grad_(True)
        qqd_batch = tuple(x[i, :] for i in range(bs))
        jac = tuple(map(partial(jacobian, self._lagrangian, create_graph=True), qqd_batch))
        hess = tuple(map(partial(hessian, self._lagrangian, create_graph=True), qqd_batch))
        qdd_batch = tuple(map(self._qdd, zip(jac, hess, qqd_batch)))
        qd, qdd = x[:, n:], torch.cat([qdd[None] for qdd in qdd_batch])
        return torch.cat([qd, qdd], 1)

    def _lagrangian(self, qqd):
        return self.net(qqd).sum()

    def _qdd(self, inp):
        n = self.n ; jac, hess, qqd = inp
        #print(qqd)
        g=self.g.bias.detach()
        er=g-qqd[0:2]
        k=15
        alpha=15
        de=-qqd[2:4]
              
        
        e=er

        V=0.5*k*(de+alpha*e)+alpha*de  #Desired acceleration
        self.i=self.i+1
        
        u=hess[n:, n:]@V+ hess[n:, :n]@qqd[n:]-jac[:n] #Lyapunov controler
        
       
        qqd=hess[n:, n:].pinverse()@(jac[:n] - hess[n:, :n]@qqd[n:]+u)
        return qqd
    








for  j in range(len(v)):  
    
    f = nn.Linear(2, 2)
    f.weight=torch.nn.parameter.Parameter(torch.tensor([[0,0],[0,0]]).float())
    f.bias=torch.nn.parameter.Parameter(torch.tensor(v[j,0:2]).float())
    #f.bias=torch.nn.parameter.Parameter(torch.tensor([-0.52,pi/2]).float())
    i=0
    m=NeuralDE(LNN2(net1,f,i), solver='dopri5').to(device)
    horizon=150
    X0=torch.tensor([[0,1.57,0,0]])
    s_span = torch.linspace(0, (horizon-1)/100, horizon)
    traj=m.trajectory(X0, s_span).cpu().detach()
    a=np.concatenate((np.asarray(traj),np.flip(np.asarray(traj),axis=0)),axis=0)
    np.save(r'C:\Users\DellG5\Desktop\EPFL\PDM\lagrangian_nns-mastertraj.npy',a)
    trajectory+=[np.asarray(traj)]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), dpi=1080)
    axes[0,0].plot(s_span,traj[:,0,0],label='LNN')
    axes[0,0].plot(s_span,np.asarray(f.bias[0].detach())*np.ones(len(traj)), color='green',label='Truth')
    #axes[0,0].plot((T_span,q1), color='yellow',label='Truth')
    axes[0,0].set_xlabel('Time [s]')
    axes[0,0].set_ylabel('$q_1$')

    
    #axes[0,1].plot((T_span,q2), color='yellow',label='Truth')    
    axes[0,1].plot(s_span,traj[:,0,1],label='LNN')
    axes[0,1].plot(s_span,np.asarray(f.bias[1].detach())*np.ones(len(traj)), color='green',label='Truth')
    axes[0,1].set_xlabel('Time [s]')
    axes[0,1].set_ylabel('$q_2$')
    plt.tight_layout() 

    #axes[1,0].plot((T_span,dq1), color='yellow',label='Truth')
    axes[1,0].plot(s_span,traj[:,0,2],label='LNN')
    axes[1,0].plot(s_span,0*np.ones(len(traj)), color='green',label='Truth')
    axes[1,0].set_xlabel('Time [s]')
    axes[1,0].set_ylabel('$dq_1$')
    plt.tight_layout() 
        
    #axes[1,1].plot((T_span,dq2), color='yellow',label='Truth')
    axes[1,1].plot(s_span,traj[:,0,3],label='LNN')
    axes[1,1].plot(s_span,0*np.ones(len(traj)), color='green',label='Truth')
    axes[1,1].set_xlabel('Time [s]')
    axes[1,1].set_ylabel('$dq_2$')
    plt.tight_layout() 
        
    
    #plt.savefig(str(j)+'.png')
    if(j==0):
        b=a
        
    if (j>0):
        b=np.concatenate((b,a),axis=0)
    
    
np.save(r'lagrangian_nns-mastertraj.npy',b)
    


