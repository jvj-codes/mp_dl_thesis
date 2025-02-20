# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:51:43 2025

@author: JVJ
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
import numpy as np
import torch
torch.set_warn_always(True)

# local
from rl_project.MonetaryPolicy import misc
from rl_project.EconDLSolvers import DLSolverClass, torch_uniform
from rl_project.EconDLSolvers import auxilliary
from rl_project.IncomeChannels import model_funcs

class IncomeChannelsModelClass(DLSolverClass):
    
    #########
	# setup #
	#########
    
    def setup(self, full=None):
        
        #parameters, active mp
        
        par = self.par
        sim = self.sim
        
        par.full = full if not full is None else torch.cuda.is_available()
        par.seed = 1
        
        # a. model
        par.T = 5 # time 
        
        ## parameters of the model
        
        par.beta = 0.99     #discount factor
        par.sigma = 3.0     #inverse of intertemporal elasticity of consumption and money holdings
        par.phi = 1         #inverse of Frisch elasticity of labor supply
        par.chi = 0.1       #relative preference weight of money holdings
        par.A = 1.3         #taylor rule coefficient
        par.p_star = 1.01   #target gross high-inflation rate (4% per annum)
        par.rho_p = 0.3     #persistence of inflation 
        par.gamma = 1.03    #gross wage growth
        par.eta = 0.7       #fraction of wealth to determine limit of debt
        par.vartheta = 0.4  #fraction of wage income to determine limit of debt
        
        ## mean and std of shocks
        
        par.sigma_psi = 0.0005 #wage shock (std std.) log_normal distributed
        par.mu_psi = 1         #wage shock mean one 
        
        par.sigma_Q = 2 #equity returns shock (std. dev) gamma distributed
        par.mu_Q = 1    #equity returns shock mean one
        
        par.sigma_R = 0.0005 #monetary policy shock (std. dev) normal mean = 0
        par.mu_R = 0.0       #monetary policy shock mean
        par.NR = 4           #monetary policy quadrature nodes
        
        par.sigma_pi = 0.0005 #inflation shock (std. dev) log_normal distributed
        par.mu_pi = 0.0       #inflation shock mean
        par.Npi               #inflation shock quardrature nodes
        
        par.sigma_un = 0.00005   #unemployment (std dev) normal distributed
        par.mu_un = 0.05         #unemployment on average 5pct.         
        
        
        ## initial states
        
        par.m_min = 1.670
        par.m_max = 1.750
        
        par.b_min = 3.960
        par.b_max = 4.040
        
        par.c_min = 0.995
        par.c_max = 1.005
        
        par.pi_min = 0.000 # minimum inflation of 0% 
        par.pi_max = 0.080 # maximum inflation of 8%
        
        par.n_min = 0.990
        par.n_max = 1.010
        
        # b. solver settings
        
        # states, actions and shocks
        par.Nstates = 5
        par.Nstates_pd = 5
        par.Nshocks = 5
        par.Nactions = 3
        
        # outcomes and actions
        par.Noutcomes = 3 # consumption, labor, money holdings, bond holdings
        par.KKT = False ## use KKT conditions (for DeepFOC)
        par.NDC = 0 # number of discrete choices
        
        # c. simulation
        sim.N = 100_000 # number of agents 
        sim.reps = 10 # number of repetitions
        
    def allocate(self):
        
        # a. unpack
        par = self.par
        sim = self.sim
        train = self.train
        
        dtype = train.dtype
        device = train.device
        
        if not par.full: # for solving without GPU
            par.T= 3
            sim.N = 10_000
        
        # b. quadrature
        par.epsn_t, par.epsn_t_w = misc.log_normal_gauss_hermite(sigma = par.sigma_tau, n = par.Ntau, mu = par.mu_tau) #returns nodes of length n, weights of length n
        
        par.epsn_R, par.epsn_R_w = misc.normal_gauss_hermite(sigma = par.sigma_R, n = par.NR, mu = par.mu_R)
        
        par.epsn_y, par.epsn_y_w = misc.normal_gauss_hermite(sigma = par.sigma_y, n = par.Ny, mu = par.mu_y)
        
        par.epsn_t_w = torch.tensor(par.epsn_t_w,dtype=dtype,device=device)
        par.epsn_t = torch.tensor(par.epsn_t,dtype=dtype,device=device)
        
        par.epsn_R_w = torch.tensor(par.epsn_R_w,dtype=dtype,device=device)
        par.epsn_R = torch.tensor(par.epsn_R,dtype=dtype,device=device)
        
        par.epsn_y_w = torch.tensor(par.epsn_y_w,dtype=dtype,device=device)
        par.epsn_y = torch.tensor(par.epsn_y,dtype=dtype,device=device)
        
        # c. simulation
        sim.states = torch.zeros((par.T,sim.N,par.Nstates),dtype=dtype,device=device)
        sim.states_pd = torch.zeros((par.T,sim.N,par.Nstates_pd),dtype=dtype,device=device)
        sim.shocks = torch.zeros((par.T,sim.N,par.Nshocks),dtype=dtype,device=device)
        sim.outcomes = torch.zeros((par.T,sim.N,par.Noutcomes),dtype=dtype,device=device)
        sim.actions = torch.zeros((par.T,sim.N,par.Nactions),dtype=dtype,device=device)
        sim.reward = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
        
    def setup_train(self):
        
        par = self.par
        train = self.train
        dtype = train.dtype
        device = train.device
        
        # a. neural network
        if not par.full:
            train.Neurons_policy = np.array([100,100])
            train.Nneurons_value = np.array([100,100])
            
        # b. policy activation functions and clipping
        if par.KKT:
            raise NotImplementedError
            #train.policy_activation_final = ['sigmoid', 'sigmoid', 'softplus']
            #train.min_actions = torch.tensor([0.995, 3.960, 0.990],dtype=dtype,device=device) #AMP minimum action value c, b, n
            #train.max_actions = torch.tensor([1.005, 4.040, 1.010],dtype=dtype,device=device) #AMP maximum action value c, b, n
        else: 
            train.policy_activation_final = ['sigmoid', 'sigmoid', 'sigmoid']
            train.min_actions = torch.tensor([1.005, 4.00, 0.990],dtype=dtype,device=device) #AMP minimum action value c, b, n
            train.max_actions = torch.tensor([1.015, 4.08, 1.010],dtype=dtype,device=device) #AMP maximum action value c, b, n
            
            train.epsilon_sigma = np.array([0.1,0.1,0.1])
            train.epsilon_sigma_min = np.array([0.0,0.0,0.0])
        
        # c. misc
        train.terminal_actions_known = False # use terminal actions
        train.use_input_scaling = False # use input scaling
        train.do_exo_actions = -1 
        
        if train.algoname == 'DeepVPD':
            train.learning_rate_value_decay = 0.999
            train.learning_rate_policy_decay = 0.999
            train.tau = 0.4
            train.start_train_policy = 50
            
        else:
            raise NotImplementedError
            
    def allocate_train(self):
        
        par = self.par
        train = self.train
        dtype = train.dtype
        device = train.device
        
        train.states = torch.zeros((par.T,train.N,par.Nstates),dtype=dtype,device=device)
        train.states_pd = torch.zeros((par.T,train.N,par.Nstates_pd),dtype=dtype,device=device)
        train.shocks = torch.zeros((par.T,train.N,par.Nshocks),dtype=dtype,device=device)
        train.outcomes = torch.zeros((par.T,train.N,par.Noutcomes),dtype=dtype,device=device)
        train.actions = torch.zeros((par.T,train.N,par.Nactions),dtype=dtype,device=device)
        train.reward = torch.zeros((par.T,train.N),dtype=dtype,device=device)
        
    def quad(self):
        
        par = self.par
        
        epsn_t,epsn_R,epsn_y = torch.meshgrid(par.epsn_t,par.epsn_R,par.epsn_y, indexing='ij')
        
        epsn_t_w,epsn_R_w,epsn_y_w = torch.meshgrid(par.epsn_t_w,par.epsn_R_w,par.epsn_y_w, indexing='ij')
        
        quad = torch.stack((epsn_t.flatten(), epsn_R.flatten(), epsn_y.flatten()), dim=1)
        quad_w = epsn_t_w.flatten() * epsn_R_w.flatten() * epsn_y_w.flatten()
        
        return quad, quad_w

    
    def draw_shocks(self,N):
        
        par = self.par
        
        epsn_t = torch.exp(torch.normal(mean=par.mu_tau, std=par.sigma_tau, size=(par.T, N),))
        epsn_R = torch.normal(par.sigma_R, par.mu_R, size=(par.T,N,))
        epsn_y = torch.normal(par.sigma_y, par.mu_y, size=(par.T,N,))
        
        return torch.stack((epsn_t, epsn_R, epsn_y),dim=-1) 
    
    def draw_initial_states(self,N,training=False): #atm uniform
        
        par = self.par
        
        # a. draw initial real money balance
        m0 = torch_uniform(par.m_min, par.m_max, size = (N,))
        
        # b. draw initial bond holdings
        b0 = torch_uniform(par.b_min, par.b_max, size = (N,))
        
        # c. draw initial consumption
        c0 = torch_uniform(par.c_min, par.c_max, size = (N,))
        
        # d. draw initial inflation t-1
        p0 = torch_uniform(par.p_min, par.p_max, size = (N,))
        
        # e. draw initial hours worked
        n0 = torch_uniform(par.n_min, par.n_max, size = (N,))
        
        # f. initial state for shocks
        
        t0_eps = torch.exp(torch.normal(mean=par.mu_tau, std=par.sigma_tau, size=(N,)))
        r0_eps = torch.normal(par.sigma_R, par.mu_R, size=(N,))
        y0_eps = torch.normal(par.sigma_y, par.mu_y, size=(N,))
        
        par.epsn_R_prev = r0_eps
        
        return torch.stack((m0, b0, c0, p0, n0, t0_eps, r0_eps, y0_eps),dim=-1)
    
    def draw_exploration_shocks(self,epsilon_sigma,N): 
        """ draw exploration shockss """ 
        par = self.par 
        
        eps = torch.zeros((par.T,N,par.Nactions)) 
        for i_a in range(par.Nactions): 
            eps[:,:,i_a] = torch.normal(0,epsilon_sigma[i_a],(par.T,N)) 
            
        return eps
    
    def draw_exo_actions(self, N):
        
        par = self.par 
        
        # a. draw exo consumption
        c_exo = torch_uniform(par.c_min, par.c_max, size = (N,))
        
        # b. draw exo real money balance
        m_exo = torch_uniform(par.m_min, par.m_max, size = (N,))
    
        # c. draw exo hours worked
        n_exo = torch_uniform(par.n_min, par.n_max, size = (N,))
        
        return torch.stack((c_exo, m_exo, n_exo))
    
    outcomes = model_funcs.outcomes
    reward = model_funcs.reward
    discount_factor = auxilliary.discount_factor
	
    terminal_reward_pd = auxilliary.terminal_reward_pd
		
    state_trans_pd = model_funcs.state_trans_pd
    state_trans = model_funcs.state_trans
    exploration = auxilliary.exploration

    
    
        
        
            
            
            