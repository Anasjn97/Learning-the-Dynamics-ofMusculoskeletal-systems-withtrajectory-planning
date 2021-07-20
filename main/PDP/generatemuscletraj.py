from arm_files.arm_flexor import Arm2DVecEnv, Arm2DEnv
import numpy as np
from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import math
import scipy.io as sio
import numpy as np
import time
import matplotlib.pyplot as plt

# --------------------------- load environment ----------------------------------------
arm = JinEnv.RobotArm()
arm.initDyn(g=9.81)

# --------------------------- create PDP SysID object ----------------------------------------
# create a pdp object
dt = 0.01
armid = PDP.SysID()
armid.setAuxvarVariable(arm.dyn_auxvar)
armid.setStateVariable(arm.X)
armid.setControlVariable(arm.U)
dyn = arm.X + dt * arm.f
armid.setDyn(dyn)
#%%
# --------------------------- generate experimental data ----------------------------------------
# true_parameter

# generate the rand inputs
hor=75
batch_inputs = armid.getRandomInputs(n_batch=10, lb=[-5,-5], ub=[5,5], horizon=hor)
batch_states = []
batch_statep = []
allow_pickle=True

env = Arm2DEnv(visualize=False)
observation = env.reset()
acceleration=[]
actuation=[]
for j in range(3):
    elbow=[]
    shoulder=[]
    delbow=[]
    dshoulder=[]
    states = numpy.zeros((hor + 1, 4))
    statesp = numpy.zeros((hor + 1, 4))
    acc=numpy.zeros((hor + 1, 2))
    act=numpy.zeros((hor, 2))
    for i in range(hor+1):
        actions = env.action_space.sample()
        observation, reward, done, info = env.step(actions)
        states[i, :] = [-pi/2+observation['joint_pos'].get('r_shoulder')[0],observation['joint_pos'].get('elbow')[0],observation['joint_vel'].get('r_shoulder')[0],observation['joint_vel'].get('elbow')[0]]
        statesp[i, :] = [observation['joint_vel'].get('r_shoulder')[0],observation['joint_vel'].get('elbow')[0],observation['joint_acc'].get('r_shoulder')[0],observation['joint_acc'].get('elbow')[0]]
        acc[i,:] = [observation['joint_acc'].get('r_shoulder')[0],observation['joint_acc'].get('elbow')[0]]
        if i>0:
             act[i-1,:] = [observation['forces'].get('shoulder_flexion_actuator')[0],observation['forces'].get('elbow_flexion_actuator')[0]]
            
    
    
    batch_states += [states]
    batch_statep += [statesp]
    acceleration+=[acc]  
    actuation+=[act]
    observation = env.reset(random_target=True, obs_as_dict=True)



#%%


robotarm_iodata = {'batch_inputs': actuation,
                   'batch_states': batch_states,
                   'batch_statesp': batch_statep,
                   'acceleration':acceleration,
                   'actuation':actuation}
sio.savemat('data/robotarm_iodata.mat', {'robotarm_iodata': robotarm_iodata})        
        
# --------------------------- Verify equations ----------------------------------------        
