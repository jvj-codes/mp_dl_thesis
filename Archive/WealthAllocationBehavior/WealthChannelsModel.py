# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:51:43 2025

@author: JVJ
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
import numpy as np
import torch
import math
torch.set_warn_always(True)

# local
from rl_project.WealthAllocationBehavior import misc
from rl_project.WealthAllocationBehavior import model_funcs
from rl_project.EconDLSolvers import DLSolverClass, torch_uniform
from rl_project.EconDLSolvers import auxilliary

class WealthBehaviorModelClass(DLSolverClass):
    
    #########
	# setup #
	#########
    
    def setup(self, full=None):
        
        par = self.par
        sim = self.sim
        train = self.train
        dtype = train.dtype
        device = train.device
        
        par.full = full if not full is None else torch.cuda.is_available()
        par.seed = 1
        
        # a. model
        par.T = 100 # time 
        
        ## parameters of the model
        par.beta = 0.99     #discount factor
        par.j = 0.1         #relative preference weight of house holdings
        par.A = 1.3         #taylor rule coefficient
        par.pi_star = math.log(1.80) #target gross high-inflation rate (
        par.R_star = par.beta / math.exp(par.pi_star) #target nominal interest rate
        par.rhopi = 0.3     #persistence of inflation 
        par.gamma = 1.03    #gross wage growth
        par.vartheta = 0.7  #fraction of wealth to determine limit of debt
        par.eta = 0.4       #labor supply schedule
        par.lbda = 0.2      #minimum payment due
        par.varphi = 1.01   #labor supply schedule
        par.q0 = 0.001      #initial value house price
        par.q_h = 0.5       #spread importance
        par.eps_rp = 0.02   #risk premium

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
        par.h_mu = 0         #house price mean 
        par.Nh = 4
        
        
        ## Interval values of initial states and actions REVISIT
        
        par.pi_min = 0.000 # minimum inflation of 0% 
        par.pi_max = 0.080 # maximum inflation of 8%
        
        par.w_min = 0.000    # minimum wage
        par.w_max = 0.999 # maximum wage, no bounds?
        

        # b. solver settings
        
        # states, actions and shocks
        par.Nstates = 7
        par.Nstates_pd = 8
        par.Nshocks = 5
        par.Nactions = 5
        
        # outcomes and actions
        par.Noutcomes = 3 # consumption, house holdings, labor
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
            par.T = 100
            sim.N = 10_000
            
        if par.KKT:
            par.Nactions = 7 #revisit when focs are done
        else:
            par.Nactions = 5
        
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
        
        sim.R = np.nan
        
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
            ## n, alpha_d, alpha_e, alpha_b, alpha_h
            train.policy_activation_final = ['sigmoid', 'sigmoid', 'sigmoid','sigmoid','sigmoid','sigmoid','softplus']
            train.min_actions = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],dtype=dtype,device=device) #minimum action value n, alpha_d, alpha_e, alpha_b, alpha_h
            train.max_actions = torch.tensor([0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, np.inf],dtype=dtype,device=device) #maximum action value n, alpha_d, alpha_e, alpha_b, alpha_h
            
        else: 
            ## n, alpha_d, alpha_e, alpha_b, alpha_h
            train.policy_activation_final = ['sigmoid', 'sigmoid', 'sigmoid','sigmoid','sigmoid']
            train.min_actions = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000],dtype=dtype,device=device) #minimum action value n, alpha_d, alpha_e, alpha_b, alpha_h
            train.max_actions = torch.tensor([0.9999, 0.9999, 0.9999, 0.9999, 0.9999],dtype=dtype,device=device) #maximum action value n, alpha_d, alpha_e, alpha_b, alpha_h
            
        
        # c. misc
        train.terminal_actions_known = False # use terminal actions
        train.use_input_scaling = False # use input scaling
        train.do_exo_actions = -1 
        
        if train.algoname == 'DeepVPD':
            train.learning_rate_value_decay = 0.999
            train.learning_rate_policy_decay = 0.999
            train.tau = 0.4
            train.start_train_policy = 50
            
        if train.algoname == 'DeepFOC':
            train.eq_w = torch.tensor([3.0, 3.0, 3.0, 3.0, 5.0],dtype=dtype,device=device) #revisit this
            
        if train.algoname == 'DeepSimulate':
            pass
        else:
            if par.KKT:
                train.epsilon_sigma = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.0]) #revisit this
                train.epsilon_sigma_min = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            else:
                train.epsilon_sigma = np.array([0.1,0.1,0.1,0.1,0.1]) #revisit this
                train.epsilon_sigma_min = np.array([0.0,0.0,0.0,0.0,0.0])

            
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
        epsn_q = dist_Q.sample((par.T,N))
        epsn_h = torch.exp(torch.normal(par.h_mu, par.h_sigma, size=(par.T,N,)))
        return torch.stack((psi, epsn_R, epsn_pi, epsn_q, epsn_h),dim=-1) 
    
    def draw_initial_states(self,N,training=False): #atm uniform
        
        par = self.par
        
        # a. draw initial inflation pi
        pi0 = torch_uniform(par.pi_min, par.pi_max, size = (N,))
        
        # b. draw initial nominal interest rate
        R0 = torch.exp(torch.normal(mean=par.R_mu, std=par.R_sigma, size=(N,))) 
        R0 = R0*(par.R_star -1)*(pi0/par.pi_star)**((par.A*par.R_star)/(par.R_star -1)) + 1
        
        # c. draw initial equity return
        R_e0 = torch.zeros((N,))
        
        # d. draw initial wage
        w0 = torch.ones((N,))
        
        # e. draw initial money holdings
        m0 = torch.ones((N,))
        
        # f. draw initial debt
        d0 = torch.zeros((N,))
        
        # g. draw initial house prices
        epsn_h0 = torch.exp(torch.normal(par.h_mu, par.h_sigma, size=(N,)))
        q0 = (par.q0 + par.q_h*(R_e0 - R0) + epsn_h0)/(1+pi0)
        
        
        return torch.stack((w0,m0,d0,q0,pi0,R0,R_e0),dim=-1)
    
    def draw_exploration_shocks(self,epsilon_sigma,N): 
        """ draw exploration shockss """ 
        par = self.par 
        
        eps = torch.zeros((par.T,N,par.Nactions)) 
        for i_a in range(par.Nactions): 
            eps[:,:,i_a] = torch.normal(0,epsilon_sigma[i_a],(par.T,N)) 
            
        return eps
    
    def draw_exo_actions(self, N):
        
        par = self.par 
        
        # a. draw exo labor time spend
        n_exo = torch_uniform(0.00001, 0.999, size = (N,))
        
        # b. draw exo debt share
        alph_d_exo = torch_uniform(0.00001, 0.999, size = (N,))
    
        # c. draw exo bond share
        alph_b_exo = torch_uniform(0.00001, 0.999, size = (N,))
        
        # d. draw exo equity share
        alph_e_exo = torch_uniform(0.00001, 0.999, size = (N,))
        
        # e. draw exo hpise share
        alph_h_exo = torch_uniform(0.00001, 0.999, size = (N,))
        return torch.stack((n_exo, alph_d_exo, alph_e_exo, alph_b_exo, alph_h_exo))
    
    outcomes = model_funcs.outcomes
    reward = model_funcs.reward
    discount_factor = auxilliary.discount_factor
	
    #terminal_reward_pd = auxilliary.terminal_reward_pd
		
    state_trans_pd = model_funcs.state_trans_pd
    state_trans = model_funcs.state_trans
    exploration = auxilliary.exploration

    
    
        
        
            
            
            