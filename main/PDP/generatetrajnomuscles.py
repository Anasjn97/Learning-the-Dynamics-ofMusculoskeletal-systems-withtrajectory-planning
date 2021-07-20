from arm_files.arm_torque import Arm2DVecEnv, Arm2DEnv
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
arm.initDyn(g=9.806)

# --------------------------- create PDP SysID object ----------------------------------------
# create a pdp object
dt = 0.01
armid = PDP.SysID()
armid.setAuxvarVariable(arm.dyn_auxvar)
armid.setStateVariable(arm.X)
armid.setControlVariable(arm.U)
dyn = arm.X + dt * arm.f
armid.setDyn(dyn)

# --------------------------- generate experimental data ----------------------------------------
# true_parameter

# generate the rand inputs
hor=1000
batch_inputs = armid.getRandomInputs(n_batch=1, lb=[-5,-5], ub=[5,5], horizon=hor)
batch_states = []
batch_statep = []
def modify_default_Coord(osim_file, coord, value):
    """ Modify a coordinate default value in osim file
     INPUTS: - osim_file: string, path to the .osim file

                - coord: string, coordinate name
                - value: float, new value (in radian)
OUTPUTS: - osim_file: string, path to the modified .osim file
"""
    file = open(osim_file, 'r')
    lines = file.readlines()
    new_lines = lines
    #print(range(len(lines)))
    for l in range(len(lines)):
        line = lines[l]
        if(len(line.split())>0):
            if len(line.split()[0].split('<')) > 1 and line.split()[0].split('<')[1] == 'Coordinate':
                if line.split()[1].split('"')[1] == coord:
                    new_lines[l + 4] = '				                            <default_value>'+str(value)+'</default_value>\n'
                    #print(line.split())
                    with open(osim_file, 'w') as file:
                        file.writelines(new_lines)
    return osim_file

osimf=r'C:\Users\DellG5\Desktop\EPFL\PDM\main_anas\predictiveArmControl-main\arm_files\twoActuators.osim'
coord1=r"r_shoulder_elev"
coord2=r"elbow_flexion"


env = Arm2DEnv(visualize=False)
observation = env.reset()

acceleration=[]
actuation=[]
hor=1000
for j in range(1):
    
    x=np.random.normal(loc=0.78, scale=0.1, size=None)
    x2=np.random.normal(loc=1.13*0.1, scale=0.1, size=None)
    modify_default_Coord(osimf, coord1, x)
    modify_default_Coord(osimf, coord2, x2)
    
    env = Arm2DEnv(visualize=False)
    observation = env.reset()
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


    


#%%

true_parameter = [0.325, observation['body_mass'].get('r_humerus')[0], 0.2483, observation['body_mass'].get('ulna')[0]+observation['body_mass'].get('radius')[0]]
# save the data
robotarm_iodata = {'batch_inputs': actuation,
                   'batch_states': batch_states,
                   'batch_statesp': batch_statep,
                   'true_parameter': true_parameter,
                   'acceleration':acceleration,
                   'actuation':actuation}
sio.savemat('data/robotarm_iodata.mat', {'robotarm_iodata': robotarm_iodata})        
        
# --------------------------- Verify equations ----------------------------------------        
