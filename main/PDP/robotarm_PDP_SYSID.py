from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import numpy as np
import time
import scipy.io as sio

# --------------------------- load environment ----------------------------------------
arm = JinEnv.RobotArm()
# Setup PDP environment
#l1, m1, l2, m2, g = 0.3378,1.864572,0.2483 ,1.33889,9.81
arm.initDyn(c1=0,c2=0,c3=0,c4=0,g=9.806)
# --------------------------- create PDP SysID object ----------------------------------------
dt = 0.01
armid = PDP.SysID()
armid.setAuxvarVariable(arm.dyn_auxvar)
armid.setStateVariable(arm.X)
armid.setControlVariable(arm.U)

dyn = arm.X + dt * arm.f
armid.setDyn(dyn)


#%%
# --------------------------- load the data ----------------------------------------
load_data = sio.loadmat('data/robotarm_iodata.mat')
data = load_data['robotarm_iodata'][0, 0]

true_parameter=[0.5,1.86,0.5,1.33] #Initialize the Parameters

n_batch = len(data['batch_inputs'])
batch_inputs = []
batch_states = []
current_parameterv=[]
for i in range(n_batch):
    batch_inputs += [data['batch_inputs'][i]]
    batch_states += [data['batch_states'][i]]

# --------------------------- load the data ----------------------------------------
for j in range(1):
    start_time = time.time()
    # learning rate
    lr = 1e-6
    # initialize
    loss_trace, parameter_trace = [], []
    sigma = 1 
    initial_parameter = np.array(true_parameter) + sigma * np.random.rand(len(true_parameter)) - sigma / 2
    current_parameter = initial_parameter
    
    
    
    
    
    for k in range(int(10e5)):
        # one iteration of PDP
        loss, dp = armid.step(batch_inputs, batch_states, current_parameter)
        # update
        current_parameter -= lr * dp
        loss_trace += [loss]
        parameter_trace += [current_parameter]
        if (np.isnan(loss)==1) or ((loss>10000)and k>1000):
            break
        
        # print
        if k % 100 == 0:
            print('Trial:', j, 'Iter:', k, 'loss:', loss)
            print(current_parameter)  
          
    
    
    # save
    current_parameterv+=[current_parameter]
    print(np.mean(current_parameterv,0))
    save_data = {'trail_no': j,
                 'loss_trace': loss_trace,
                 'parameter_trace': parameter_trace,
                 'learning_rate': lr,
                 'time_passed': time.time() - start_time},
    sio.savemat('./results/PDP_SysID_results_trial_' + str(j) + '.mat', {'results': save_data})

    
    
