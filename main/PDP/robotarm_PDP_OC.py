from PDP import PDP
from JinEnv import JinEnv
from casadi import *
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt

# --------------------------- load environment ----------------------------------------



#Target Positions
v=np.asarray([[pi/2-0.15,pi/2+0.4,0,0],[pi/2+0.1,pi/2+0.6,0,0],[pi/2+0.3,pi/2+0.3,0,0],[pi/2+0.4,pi/2-0.05,0,0],[pi/2+0.35,pi/2-0.45,0,0],[pi/2+0.05,pi/2-0.45,0,0],[pi/2-0.20,pi/2-0.35,0,0],[pi/2-0.35,pi/2-0.05,0,0]])  
  
def OCSOLv(dt,horizon,ini_state,a,t):
    arm = JinEnv.RobotArm()
    
    
    l1, m1, l2, m2, g = 0.32463838,1.94784329,0.2571893,1.0723924,9.81
    arm.initDyn(l1=l1, m1=m1, l2=l2, m2=m2, g=9.81)
    wq1, wq2, wdq1, wdq2, wu = 0.1, 0.1, 0.01, 0.01, 0.001
    
    arm.initCost(wq1=wq1, wq2=wq2, wdq1=wdq1, wdq2=wdq2, wu=wu,a=a)
    armoc = PDP.ControlPlanning()
    armoc.setStateVariable(arm.X)
    armoc.setControlVariable(arm.U)
    dyn = arm.X + dt * arm.f
    armoc.setDyn(dyn)
    armoc.setPathCost(arm.path_cost)
    armoc.setFinalCost(arm.final_cost)
    
    true_cartpoleoc = PDP.OCSys()
    true_cartpoleoc.setStateVariable(arm.X)
    true_cartpoleoc.setControlVariable(arm.U)
    true_cartpoleoc.setDyn(dyn)
    true_cartpoleoc.setPathCost(arm.path_cost)
    true_cartpoleoc.setFinalCost(arm.final_cost)
    true_sol = true_cartpoleoc.ocSolver(ini_state=ini_state, horizon=t)
    true_state_traj = true_sol['state_traj_opt']
    true_control_traj = true_sol['control_traj_opt']
    u=int(t/horizon)
    qs=np.empty(shape=(u,4))
    qs=np.append(qs,[ini_state],axis=0)
    for s in range(u):
        qs[s,:]=true_state_traj[(s+1)*horizon,:]

    print(ini_state)
    print(qs[s,:])
    output={'qs':qs,
            'solved_traj':true_state_traj}
    return output



def OCControl(dt,horizon,ini_state,a):
    
# --------------------------- create PDP Control/Planning object ----------------------------------------
    arm = JinEnv.RobotArm()
    l1, m1, l2, m2, g = 0.32463838,1.94784329,0.2571893,1.0723924,9.81 #Input Learned parameters
    arm.initDyn(l1=l1, m1=m1, l2=l2, m2=m2, g=10)
    wq1, wq2, wdq1, wdq2, wu = 0.2, 0.2, 0.01, 0.01, 0.0001 #Specify weights
    arm.initCost(wq1=wq1, wq2=wq2, wdq1=wdq1, wdq2=wdq2, wu=wu,a=a)
    #arm.initCosty2(wq1=wq1, wq2=wq2, wdq1=wdq1, wdq2=wdq2,l1=l1,l2=l2, wu=wu,a=[pi/4,pi/2,0,0,-0.17,0.4])
    armoc = PDP.ControlPlanning()
    armoc.setStateVariable(arm.X)
    armoc.setControlVariable(arm.U)
    dyn = arm.X + dt * arm.f
    armoc.setDyn(dyn)
    armoc.setPathCost(arm.path_cost)
    armoc.setFinalCost(arm.final_cost)
    
    
    # --------------------------- create PDP true OC object ----------------------------------------
    true_cartpoleoc = PDP.OCSys()
    true_cartpoleoc.setStateVariable(arm.X)
    true_cartpoleoc.setControlVariable(arm.U)
    true_cartpoleoc.setDyn(dyn)
    true_cartpoleoc.setPathCost(arm.path_cost)
    true_cartpoleoc.setFinalCost(arm.final_cost)
    true_sol = true_cartpoleoc.ocSolver(ini_state=ini_state, horizon=horizon)
    true_state_traj = true_sol['state_traj_opt']
    true_control_traj = true_sol['control_traj_opt']
    #print(true_state_traj[horizon-1,:],'vs',a)
    #print(true_sol['cost'])
    #print(true_state_traj[0,:])
    #print(true_state_traj[horizon-1,:])
    traj=np.empty(shape=(1,4))
    #traj=np.append(traj,[ini_state],axis=0)available
    trajopt=np.empty(shape=(1,4))
    #trajopt=np.append(traj,[ini_state],axis=0)
    for j in range(1):
        # learning rate
        lr = 1e-1
        loss_trace, parameter_trace = [], []
        armoc.recmat_init_step(horizon, -1)
        initial_parameter = np.random.randn(armoc.n_auxvar)
        current_parameter = initial_parameter
        max_iter =10000
        start_time = time.time()
        for k in range(int(max_iter)):
            # one iteration of PDP
            loss, dp = armoc.recmat_step(ini_state, horizon, current_parameter)
            # update
            current_parameter -= lr * dp
            loss_trace += [loss]
            parameter_trace += [current_parameter]
            # print
            if k % 1000 == 0:
                print('trial:', j ,'Iter:', k, 'loss:', loss)
            # if (loss <0.01) == 1:
            #     break
            #     print('early exit')    
    
        # solve the trajectory
        sol = armoc.recmat_unwarp(ini_state, horizon, current_parameter)
        
        traj=np.append(traj,sol['state_traj'][0:(horizon-1),:],axis=0)
        trajopt=np.append(trajopt,true_state_traj[0:(horizon-1),:],axis=0)
        save_data = {'parameter_trace': parameter_trace,
                     'loss_trace': loss_trace,
                     'learning_rate': lr,
                     'solved_solution': sol,
                     'true_solution': true_sol,
                     'solved_traj':traj,
                     'optimal_traj':trajopt,
                     'time_passed': time.time() - start_time,
                     'robotarm': {'l1': l1,
                                 'm1': m1,
                                 'l2': l2,
                                 'm2': m2,
                                 'wq1': wq1,
                                 'wq2': wq2,
                                 'wdq1': wdq1,
                                 'wdq2': wdq2,
                                 'wu': wu},
                     'dt': dt,
                     'horizon': horizon}
       
    return save_data
        
  
l1, m1, l2, m2, g =  0.32463838,1.94784329,0.2471893,1.0723924, 10

t=50
dt = 0.05
horizon = 5
ini_state = [pi/2, pi/2, 0, 0]
#ini_state=[1.19539789e+00 ,1.47250691e+00, 6.59584303e-03, 1.32690461e-03]
# ini_state = [1.0781,0.6854,0,0] 
# ini_state = [0.6854,pi/7,0,0]
# ini_state=[0.6781,pi/2,0,0]
#a=[pi/4+1,pi/2-2,0,0]
a=[3*(pi/8),pi/2-0.1,0,0]
a=[1.42, 1.97, 0, 0]
output=OCSOLv(dt=dt,horizon=horizon,ini_state=ini_state,a=a,t=t)
qs=output['qs']
save_data=[]
ini_state = [pi/2, pi/2, 0, 0]
for i in range(len(v)):
    a=v[i]
    for k in range(10):
        ini_state = [pi/2, pi/2, 0, 0]
        for j in range(len(qs)):
            temp=OCControl(dt=dt,horizon=horizon,ini_state=ini_state,a=a)
            save_data+=[temp]
            
            ini_state=temp['solved_traj'][4,:]
            print(temp['solved_traj'][-1,:])
            print(ini_state)
       # a=qs[j,:]
print(a)


trajectoryv=[]
for k in range(88):  
     trajectory=np.empty(shape=(1,4))*0
     trajectoryideal=np.empty(shape=(1,4))*0    
     for j in range(10):
         trajectory=np.append(trajectory,save_data[k*j]['solved_traj'],axis=0)
         trajectoryideal=np.append(trajectoryideal,save_data[k*j]['optimal_traj'],axis=0)
     trajectoryv+=[trajectory]
    
    
save={'save_data':save_data}


sio.savemat('./data/PDP_Recmat_results_trial' + str(0) + '.mat', {'results': save})
