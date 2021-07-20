1-Generate trajectories using eiter generatemuscletraj.py for the arm with muscles or generatetrajnomuscle.py for the arm without muscles and specifying the chosen hyperparameters.

2- Start The system Identification task with robotarm_PDP_SYSID.py

3- Run robotarm_PDP_OC.py Specify the desired goals and obtain the desired trajectories.


This program uses the built in Environment, Jinenv use to identify and control robot arms. https://github.com/wanxinjin/Pontryagin-Differentiable-Programming


To solve system identification problems, you will need SysID class from the module ./PDP/PDP.py:

SysID: which is to integrate the controlled (autonomous) system in forward pass, obtain the corresponding auxiliary control system, and then integrate the auxiliary control system in backward pass. The procedure to instantiate a SysID object is fairly straightforward, including seven steps:
Step 1: set state variable of your dynamics ----> setStateVariable
Step 2: set input variable of your dynamics ----> setControlVariable
Step 3: set (unknown) parameters in dynamics----> setAuxvarVariable
Step 4: set dynamics (difference) equation----> setDyn
Step 5: integrate the dynamics equation in forward pass -----> integrateDyn
Step 6: get the auxiliary control system ------> getAuxSys
Step 7: integrate the auxiliary control system in backward pass ------> integrateAuxSys




ControlPlanning. The procedure to instantiate a ControlPlanning object is fairly straightforward, including the following nine steps:

Step 1: set state variable of your system ----> setStateVariable
Step 2: set input variable of your system ----> setControlVariable
Step 3: set dynamics (difference) equation of your system ----> setDyn
Step 4: set path cost function of your system ----> setPathCost
Step 5: set final cost function of your system -----> setFinalCost
Step 6: set policy parameterization with (unknown) parameters -----> for planning, you can use setPolyControl (parameterize the policy as Lagrangian polynomial), or for feedback control, you can use setNeuralPolicy (parameterize the policy as feedback controller)
Step 7: integrate the control system in forward pass -----> integrateSys
Step 8: get the auxiliary control system ------> getAuxSys
Step 9: integrate the auxiliary control system in backward pass ------> integrateAuxSys