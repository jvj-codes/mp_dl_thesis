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
from rl_project.WealthBehavior import misc
from rl_project.WealthBehavior import model_funcs
from rl_project.EconDLSolvers import DLSolverClass, torch_uniform
from rl_project.EconDLSolvers import auxilliary

class WealthBehaviorModelClass(DLSolverClass):
    
    #########
	# setup #
	#########
    
    def setup(self, full=None):
        
        par = self.par
        sim = self.sim
        
        par.full = full if not full is None else torch.cuda.is_available()
        par.seed = 1
        
        # a. model
        par.T = 5 # time 
        
        ## parameters of the model
        
        par.beta = 0.99     #discount factor
        #par.sigma = 3.0     #inverse of intertemporal elasticity of consumption and money holdings
        par.phi = 1         #inverse of Frisch elasticity of labor supply
        par.chi = 0.1       #relative preference weight of money holdings
        par.j = 0.1         #relative preference weight of house holdings
        par.A = 1.3         #taylor rule coefficient
        par.pi_star = 1.01  #target gross high-inflation rate (4% per annum)
        par.R_star = par.beta / par.pi_star #target nominal interest rate
        par.rhopi = 0.3     #persistence of inflation 
        par.gamma = 1.03    #gross wage growth
        par.eta = 0.7       #fraction of wealth to determine limit of debt
        par.vartheta = 0.4  #fraction of wage income to determine limit of debt
        par.lbda = 0.2      #minimum payment due
        par.varphi = 1.01   #labor supply schedule
        par.rp = 0.03       #risk premium 3%
        par.q0 = 0.001      #initial value house price
        par.q_h = 0.5       #spread importance

        
        # wealth
        par.y_base = 1.0 # base
        
        ## mean and std of shocks
        
        par.psi_sigma = 0.0005 #wage shock (std std.) log_normal distributed
        par.psi_mu = 1         #wage shock mean one 
        par.Npsi = 4
        
        par.Q_loc = 1    #equity returns shock (mean) student-t distributed
        par.Q_nu = 4     #equity returns degrees of freedom
        par.Q_scale = 2  #equity returns shock (std. dev) student-t distributed
        par.NQ = 4       #equity returns shock quadrature nodes
        
        par.R_sigma = 0.0005 #monetary policy shock (std. dev) normal mean = 0
        par.R_mu = 0.0       #monetary policy shock mean
        par.NR = 4           #monetary policy quadrature nodes
        
        par.pi_sigma = 0.0005 #inflation shock (std. dev) log_normal distributed
        par.pi_mu = 0.0       #inflation shock mean
        par.Npi  = 4          #inflation shock quardrature nodes
        
        par.h_sigma = 0.0005 #house price shock (std. dev) log normal dis
        par.h_mu = 1         #house price mean 
        par.Nh = 4
        
        
        ## Interval values of initial states and actions REVISIT
        
        par.pi_min = 0.000 # minimum inflation of 0% 
        par.pi_max = 0.080 # maximum inflation of 8%
        
        par.R_min = 0.000 # minimum interest rate of 0%
        par.R_max = 0.100 # maximum interest rate of 10%
        
        par.eq_min = -0.30 # minium return on equity -30%
        par.eq_max = 0.30  # maximum return on equity 30%
        
        par.w_min = 0.000    # minimum wage
        par.w_max = 0.999 # maximum wage, no bounds?
        
        par.e_min = 0.000 # minimum holdings of equity
        par.e_max = 0.999 # maximum holdings of equity
        
        par.b_min = 0.000 # minimum holdings of bonds
        par.b_max = 0.999 # maximum holdings of bonds
        
        par.m_min = 0.000
        par.m_max = 0.999
        
        par.q_min = 0.000 # minimum house prices
        par.q_max = 0.999 # maximum house prices
        
        
        ###
        par.c_min = 0.000
        par.c_max = 0.999
        
        par.n_min = 0.000
        par.n_max = 0.999
        
        par.h_min = 0.000 # minimum house holdings
        par.h_max = 0.999 # maximum house holdings
        
        par.d_min = 0.000 # minimum debt
        par.d_max = 0.999 # maximum debt (will be determined by constraint ultimately)
        
        # b. solver settings
        
        # states, actions and shocks
        par.Nstates = 10
        par.Nstates_pd = 8
        par.Nshocks = 5
        par.Nactions = 6
        
        # outcomes and actions
        par.Noutcomes = 4 # consumption, labor, money holdings, house holdings
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
            par.T = 5
            sim.N = 10_000
            
        if par.KKT:
            par.Nactions = 7
        else:
            par.Nactions = 6
        
        #life time wealth
        par.y = torch.ones(par.T,dtype=dtype,device=device)	
        #par.y[0] = par.y_base
        
        # c. quadrature
        par.psi, par.psi_w  = misc.log_normal_gauss_hermite(sigma = par.psi_sigma, n = par.Npsi, mu = par.psi_mu)
        par.R, par.R_w      = misc.log_normal_gauss_hermite(sigma = par.R_sigma, n = par.NR, mu = par.R_mu) #returns nodes of length n, weights of length n
        par.pi, par.pi_w    = misc.log_normal_gauss_hermite(sigma = par.pi_sigma, n = par.Npi, mu = par.pi_mu)
        par.q, par.q_w      = misc.student_t_gauss_hermite(nu = par.Q_nu, loc = par.Q_loc, scale = par.Q_scale, n = par.NQ)
        par.h, par.h_w      = misc.log_normal_gauss_hermite(sigma = par.h_sigma, n = par.Nh, mu = par.h_mu)
        

        par.psi_w  = torch.tensor(par.psi_w,dtype=dtype,device=device)
        par.psi = torch.tensor(par.psi,dtype=dtype,device=device)
        
        par.R_w  = torch.tensor(par.R_w,dtype=dtype,device=device)
        par.R = torch.tensor(par.R,dtype=dtype,device=device)
        
        par.pi_w  = torch.tensor(par.pi_w,dtype=dtype,device=device)
        par.pi = torch.tensor(par.pi,dtype=dtype,device=device)
        
        par.q_w  = torch.tensor(par.q_w,dtype=dtype,device=device)
        par.q = torch.tensor(par.q,dtype=dtype,device=device)
        
        par.h_w  = torch.tensor(par.h_w,dtype=dtype,device=device)
        par.h = torch.tensor(par.h,dtype=dtype,device=device)
        
        # d. simulation
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
            ## c, h, n, a, b, d, mu_t
            train.policy_activation_final = ['sigmoid', 'sigmoid', 'sigmoid','sigmoid','sigmoid','sigmoid','softplus']
            train.min_actions = torch.tensor([par.c_min, par.h_min, par.n_min, par.e_min, par.b_min, par.d_min, 0.00],dtype=dtype,device=device) #minimum action value c, h, n, a, b, d
            train.max_actions = torch.tensor([par.c_max, par.h_max, par.n_max, par.e_max, par.b_max, par.d_max, np.inf],dtype=dtype,device=device) #maximum action value c, h, n, a, b, d
            
            train.epsilon_sigma = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.0]) #revisit this
            train.epsilon_sigma_min = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            
        else: 
            ## c, h, n, a, b, d
            train.policy_activation_final = ['sigmoid', 'sigmoid', 'sigmoid','sigmoid','sigmoid','sigmoid']
            train.min_actions = torch.tensor([par.c_min, par.h_min, par.n_min, par.e_min, par.b_min, par.d_min],dtype=dtype,device=device) #minimum action value c, h, n, a, b, d
            train.max_actions = torch.tensor([par.c_max, par.h_max, par.n_max, par.e_max, par.b_max, par.d_max],dtype=dtype,device=device) #maximum action value c, h, n, a, b, d
            
            train.epsilon_sigma = np.array([0.1,0.1,0.1,0.1,0.1,0.1]) #revisit this explorations (policy function)
            train.epsilon_sigma_min = np.array([0.0,0.0,0.0,0.0,0.0,0.0]) 
        
        # c. misc
        train.terminal_actions_known = False # use terminal actions
        train.use_input_scaling = False # use input scaling
        train.do_exo_actions = -1 
        
        if train.algoname == 'DeepVPD':
            train.learning_rate_value_decay = 0.999
            train.learning_rate_policy_decay = 0.999
            train.tau = 0.4
            train.start_train_policy = 50
            
        elif train.algoname == 'DeepFOC':
            train.eq_w = torch.tensor([3.0, 3.0, 3.0, 3.0, 5.0],dtype=dtype,device=device) #revisit this
            
        elif train.algoname == 'DeepSimulate':
            pass

        else:
            NotImplementedError
            
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
        
        psi,R,pi,q,h = torch.meshgrid(par.psi,par.R,par.pi,par.q,par.h, indexing='ij')
        
        psi_w,R_w,pi_w,q_w,h_w = torch.meshgrid(par.psi_w,par.R_w,par.pi_w,par.q_w,par.h_w, indexing='ij')
        
        quad = torch.stack((psi.flatten(), R.flatten(), pi.flatten(), q.flatten(), h.flatten()), dim=1)
        quad_w = psi_w.flatten() * R_w.flatten() * pi_w.flatten() * q_w.flatten() * h_w.flatten()
        
        return quad, quad_w

    
    def draw_shocks(self,N):
        
        par = self.par
        
        psi = torch.exp(torch.normal(mean=par.psi_mu,std=par.psi_sigma, size=(par.T,N,)))
        epsn_R = torch.exp(torch.normal(mean=par.R_mu, std=par.R_sigma, size=(par.T,N,)))
        epsn_pi = torch.exp(torch.normal(par.pi_mu, par.pi_sigma, size=(par.T,N,)))
        dist_Q = torch.distributions.StudentT(df=par.Q_nu, loc=par.Q_loc, scale=par.Q_scale)
        epsn_Q = dist_Q.sample((par.T,N))
        epsn_h = torch.exp(torch.normal(par.h_mu, par.h_sigma, size=(par.T,N,)))
        return torch.stack((psi, epsn_R, epsn_pi, epsn_Q, epsn_h),dim=-1) 
    
    def draw_initial_states(self,N,training=False): #atm uniform
        
        par = self.par
        
        # a. draw initial inflation pi
        pi0 = torch_uniform(par.pi_min, par.pi_max, size = (N,))
        
        # b. draw initial nominal interest rate
        R0 = torch_uniform(par.R_min, par.R_max, size = (N,))
        
        # c. draw initial equity return
        eq0 = torch_uniform(par.eq_min, par.eq_max, size = (N,))
        
        # d. draw initial wage
        w0 = torch_uniform(par.w_min, par.w_max, size = (N,))
        
        # e. draw initial equity holdings
        e0 = torch_uniform(par.e_min, par.e_max, size = (N,))
        
        # f. draw initial bond holdings
        b0 = torch_uniform(par.b_min, par.b_max, size = (N,))     
        
        # f. draw initial money holdings
        m0 = torch_uniform(par.m_min, par.m_max, size = (N,))  
        
        # h. draw initial debt
        d0 = torch_uniform(par.d_min, par.d_max, size = (N,))
        
        # i. draw initial house prices
        q0 = torch_uniform(par.q_min, par.q_max, size = (N,))
        
        # i. draw initial house holdings
        h0 = torch_uniform(par.h_min, par.h_max, size = (N,))
        
        return torch.stack((pi0,R0,eq0,w0,e0,b0,m0,d0,q0,h0),dim=-1)
    
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
        
        # c. draw exo real money balance
        h_exo = torch_uniform(par.h_min, par.h_max, size = (N,))
    
        # d. draw exo hours worked
        n_exo = torch_uniform(par.n_min, par.n_max, size = (N,))
        
        return torch.stack((c_exo, h_exo, m_exo, n_exo))
    
    outcomes = model_funcs.outcomes
    reward = model_funcs.reward
    discount_factor = auxilliary.discount_factor
	
    terminal_reward_pd = auxilliary.terminal_reward_pd
		
    state_trans_pd = model_funcs.state_trans_pd
    state_trans = model_funcs.state_trans
    exploration = auxilliary.exploration

    
    
        
        
            
            
            