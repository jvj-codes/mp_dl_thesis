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
import misc
import model_funcs
from EconDLSolvers import DLSolverClass, torch_uniform

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
        par.T = 62 #lifetime (working + retirement)
        par.T_retired = 49 #retirement at year
        
        ## parameters of the model
        par.beta = 0.99           #discount factor
        par.j = 0.01              #relative preference weight of house holdings
        par.A = 1.3               #taylor rule coefficient
        par.pi_star = 0.02        #target inflation rate
        #par.gamma = 1.025          #1.03 gives decreasing labor     #wage growth 
        par.R_star = 0.03         #(1+par.pi_star)/par.beta -1  #target nominal interest rate
        par.rhopi = 0.9#0.3       #persistence of inflation 
        par.vartheta = 2.50       #fraction of wealth to determine limit of debt
        #par.eta = 1.1 #1.5  #1.1       #labor supply schedule
        par.lbda = 0.2            #minimum payment due
        par.q_h = 0.05            #spread importance
        par.eps_rp = 0.02         #risk premium # 0.06
        par.Rpi = 0.35            #impact of interest rate on inflation lagged
        par.theta = 1.065
        par.sigma_int =  3.0000 #inverse of intertemporal elasticity of consumption and money holdings

        ### income
        par.kappa_base = 1.0 # base  # Martin - Life cycle is kappa in paper
        par.kappa_growth = 0.02 # income growth
        par.kappa_growth_decay = 0.1 # income growth decay
        par.kappa_retired = 0.7 # replacement rate

        par.rho_p_base = 0.95 # shock, persistence
        par.rho_p_low = 0.70   
        par.rho_p_high = 0.99

        par.sigma_xi_base = 0.1 # shock, permanent , std
        par.sigma_xi_low = 0.05
        par.sigma_xi_high = 0.15
        par.Nxi = 4 # number of qudrature nodes

        par.sigma_psi_base = 0.1 # shock, transitory std
        par.sigma_psi_low = 0.05
        par.sigma_psi_high = 0.15
        par.Npsi = 4 # number of qudrature nodes
        
        # exploration of initial states
        par.explore_states_endo_fac = 1.0
        par.explore_states_fixed_fac_low = 1.0
        par.explore_states_fixed_fac_high = 1.0
        

        ## mean and std of shocks
        par.mu_m0 = 1.0 # initial cash-on-hand, mean
        par.sigma_m0 = 0.1 # initial cash-on-hand, std
        par.mu_p0 = 1.0 # initial durable, mean
        par.sigma_p0 = 0.1 # initial durable, std
        
        par.Q_loc = 10    #equity returns shock (mean) student-t distributed
        par.Q_nu = 4        #equity returns degrees of freedom
        par.Q_scale = 20  #equity returns shock (std. dev) student-t distributed 20 works
        par.NQ = 4          #equity returns shock quadrature nodes
        
        par.R_sigma = 0.0005  #monetary policy shock (std. dev) log normal mean = 0
        par.R_mu = 1          #monetary policy shock mean
        par.NR = 4            #monetary policy quadrature nodes
        
        par.pi_sigma = 0.0005  #inflation shock (std. dev) distributed
        par.pi_mu = 1e-9           #inflation shock mean
        par.Npi  = 4           #inflation shock quardrature nodes
        
        par.h_sigma = 0.0005    #house price shock (std. dev) log normal dis
        par.h_mu = 1 #0.00         #house price shock mean 
        par.Nh = 4
        
        
        ## Interval values of initial states and actions REVISIT
        
        par.pi_min = 0.01 # minimum inflation of 1% 
        par.pi_max = 0.02 # maximum inflation of 2%
        
        par.R_min = 0.03   
        par.R_max = 0.04 
        

        # b. solver settings
        
        # states, actions and shocks
        par.Nshocks = 6
        par.Nactions = 4
        par.Nstates_fixed = 1 # number of fixed states
        par.Nstates_dynamic = 8 # number of dynamic states
        par.Nstates_dynamic_pd = 9 # number of dynamic post-decision states
        
        # outcomes and actions
        par.Noutcomes = 6 # consumption, house holdings, labor, funds
        par.KKT = False ## use KKT conditions (for DeepFOC)
        par.NDC = 0 # number of discrete choices
        
        # c. simulation
        sim.N = 100_000 #100_000 # number of agents 
        sim.reps = 10 # number of repetitions
        
    def allocate(self):
        
        # a. unpack
        par = self.par
        sim = self.sim
        train = self.train
        
        dtype = train.dtype
        device = train.device
        
        if not par.full: # for solving without GPU
            par.T = 80 #lifetime (working + retirement)
            par.T_retired = 67 #retirement years
            sim.N = 10_000
            
        if par.KKT:
            par.Nactions = 7 #revisit when focs are done
        else:
            par.Nactions = 4
            #par.Nactions = 4
            
        # b. life cycle income
        par.kappa = torch.zeros(par.T,dtype=dtype,device=device)
        par.kappa[0] = par.kappa_base
    
        for t in range(1,par.T_retired):
            par.kappa[t] = par.kappa[t-1]*(1+par.kappa_growth*(1-par.kappa_growth_decay)**(t-1))

        par.kappa[par.T_retired:] = par.kappa_retired * par.kappa_base
        
        
        # b.2: states
        par.Nstates_fixed_pd = par.Nstates_fixed # number of fixed post-decision states
        par.Nstates = par.Nstates_dynamic + par.Nstates_fixed # number of states
        par.Nstates_pd = par.Nstates_dynamic_pd + par.Nstates_fixed_pd +1 # number of post-decision states

        # c. quadrature
        #par.psi, par.psi_w  = misc.log_normal_gauss_hermite(sigma = par.psi_sigma, n = par.Npsi, mu = par.psi_mu)
        _, par.psi_w = misc.log_normal_gauss_hermite(1.0,par.Npsi)
        par.psi, _ = misc.gauss_hermite(par.Npsi)
        _, par.xi_w = misc.log_normal_gauss_hermite(1.0,par.Nxi)
        par.xi, _ = misc.gauss_hermite(par.Nxi)
        
        par.R, par.R_w      = misc.log_normal_gauss_hermite(sigma = par.R_sigma, n = par.NR, mu = par.R_mu) #returns nodes of length n, weights of length n
        par.pi, par.pi_w    = misc.log_normal_gauss_hermite(sigma = par.pi_sigma, n = par.Npi, mu = par.pi_mu)
        par.q, par.q_w      = misc.student_t_gauss_hermite(nu = par.Q_nu, loc = par.Q_loc, scale = par.Q_scale, n = par.NQ)
        par.h, par.h_w      = misc.log_normal_gauss_hermite(sigma = par.h_sigma, n = par.Nh, mu = par.h_mu)
        

        par.psi_w  = torch.tensor(par.psi_w,dtype=dtype,device=device)
        par.psi = torch.tensor(par.psi,dtype=dtype,device=device)
        
        par.xi_w = torch.tensor(par.xi_w,dtype=dtype,device=device)
        par.xi = torch.tensor(par.xi,dtype=dtype,device=device)
        
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
            train.Neurons_policy = np.array([100,100,100])
            train.Nneurons_value = np.array([100,100,100])
            
        # b. policy activation functions and clipping
        if par.KKT:
            ## alpha_d, alpha_e, alpha_b, alpha_h
            train.policy_activation_final = ['sigmoid', 'sigmoid','sigmoid','sigmoid','softplus','softplus']
            train.min_actions = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],dtype=dtype,device=device) #minimum action value n, alpha_d, alpha_e, alpha_b, alpha_h
            train.max_actions = torch.tensor([0.9999, 0.9999, 0.9999, 0.9999, np.inf, np.inf],dtype=dtype,device=device) #maximum action value n, alpha_d, alpha_e, alpha_b, alpha_h
            
        else: 
            ## alpha_d, alpha_e, alpha_b, alpha_h
            train.policy_activation_final = ['sigmoid', 'sigmoid','sigmoid','sigmoid']
            train.min_actions = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000],dtype=dtype,device=device) #minimum action value n, alpha_d, alpha_e, alpha_b, alpha_h
            train.max_actions = torch.tensor([1.0-1e-6, 1.0-1e-6, 1.0-1e-6, 1.0-1e-6],dtype=dtype,device=device) #maximum action value n, alpha_d, alpha_e, alpha_b, alpha_h

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
            train.eq_w = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0, 5.0, 5.0],dtype=dtype,device=device) #revisit this
            
        if train.algoname == 'DeepSimulate':
            pass
        else:
            if par.KKT:
                train.epsilon_sigma = np.array([0.1,0.1,0.1,0.1,0.1,0.0,0.0]) #revisit this
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
        
        if train.algoname == 'DeepFOC':
            train.eq_w = train.eq_w / torch.sum(train.eq_w) # normalize
            
    def quad(self):
        
        par = self.par
        
        psi,xi,R,pi,q,h = torch.meshgrid(par.psi,par.xi,par.R,par.pi,par.q,par.h, indexing='ij')
        
        psi_w,xi_w,R_w,pi_w,q_w,h_w = torch.meshgrid(par.psi_w,par.xi_w,par.R_w,par.pi_w,par.q_w,par.h_w, indexing='ij')
        
        quad = torch.stack((xi.flatten(),psi.flatten(),R.flatten(), pi.flatten(), q.flatten(), h.flatten()), dim=1)
        quad_w = xi_w.flatten()*psi_w.flatten() * R_w.flatten() * pi_w.flatten() * q_w.flatten() * h_w.flatten()
        
        return quad, quad_w

    
    def draw_shocks(self,N):
        
        par = self.par
        
        #psi = torch.exp(torch.normal(mean=par.psi_mu,std=par.psi_sigma, size=(par.T,N,)))
        xi = torch.normal(0.0,1.0,size=(par.T,N))
        psi = torch.normal(0.0,1.0,size=(par.T,N))
        
        epsn_R = torch.exp(torch.normal(mean=par.R_mu, std=par.R_sigma, size=(par.T,N,))) / 100
        epsn_pi = torch.exp(torch.normal(par.pi_mu, par.pi_sigma, size=(par.T,N,))) / 100
        dist_Q = torch.distributions.StudentT(df=par.Q_nu, loc=par.Q_loc, scale=par.Q_scale)
        epsn_q = dist_Q.sample((par.T,N)) / 100
        epsn_h = torch.exp(torch.normal(par.h_mu, par.h_sigma, size=(par.T,N,)))
        return torch.stack((xi, psi, epsn_R, epsn_pi, epsn_q, epsn_h),dim=-1) 
    
    def draw_initial_states(self,N,training=False): #atm uniform
        
        par = self.par
        
        # a. draw initial inflation pi
        pi0 = torch_uniform(par.pi_min, par.pi_max, size = (N,))
        
        # b. draw initial nominal interest rate

        R0 = torch_uniform(par.R_min, par.R_max, size = (N,))
        
        # c. draw initial equity return
        dist_e = torch.distributions.StudentT(df=par.Q_nu, loc=par.Q_loc, scale=par.Q_scale)
        epsn_e = dist_e.sample((N,)) / 100
        R_e0 = pi0 - R0 + epsn_e
        
        # f. draw initial debt
        d0 = torch.zeros((N,))
        
        # g. draw initial house prices
        q0 = torch.ones((N,))/(1+pi0)*50

        R_q0 = torch.zeros((N,))
        
        fac = 1.0 if not training else par.explore_states_endo_fac
        fac_low = 1.0 if not training else par.explore_states_fixed_fac_low
        fac_high = 1.0 if not training else par.explore_states_fixed_fac_high

        # a. draw cash-on-hand
        sigma_m0 = par.sigma_m0*fac
        m0 = par.mu_m0*np.exp(torch.normal(-0.5*sigma_m0**2,sigma_m0,size=(N,)))
        
        # b. draw permanent income
        sigma_p0 = par.sigma_p0*fac
        p0 = par.mu_p0*np.exp(torch.normal(-0.5*sigma_p0**2,sigma_p0,size=(N,)))

        # c. draw sigma_xi
        if par.Nstates_fixed > 0: 
            sigma_xi = torch_uniform(par.sigma_xi_low*fac_low,par.sigma_xi_high*fac_high,size=(N,))

        # d. draw sigma_psi
        if par.Nstates_fixed > 1: sigma_psi = torch_uniform(par.sigma_psi_low*fac_low,par.sigma_psi_high*fac_high,size=(N,))

        # e. draw rho_p  
        if par.Nstates_fixed > 2: rho_p = torch_uniform(par.rho_p_low*fac_low,np.fmin(par.rho_p_high*fac_high,1.0),size=(N,))
        
        # f. store
        if par.Nstates_fixed == 0:
            return torch.stack((m0,p0,d0,q0,pi0,R0,R_e0,R_q0),dim=-1)
        elif par.Nstates_fixed == 1:
            return torch.stack((m0,p0,d0,q0,pi0,R0,R_e0,R_q0,sigma_xi),dim=-1)
        elif par.Nstates_fixed == 2:
            return torch.stack((m0,p0,d0,q0,pi0,R0,R_e0,R_q0,sigma_xi,sigma_psi),dim=-1)
        elif par.Nstates_fixed == 3:
            return torch.stack((m0,p0,d0,q0,pi0,R0,R_e0,R_q0,sigma_xi,sigma_psi,rho_p),dim=-1)
    
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
        #return torch.stack((n_exo,alph_e_exo, alph_b_exo, alph_h_exo))
        return torch.stack((alph_d_exo, alph_e_exo, alph_b_exo, alph_h_exo))
    
    outcomes = model_funcs.outcomes
    reward = model_funcs.reward
    discount_factor = model_funcs.discount_factor
	
    terminal_reward_pd = model_funcs.terminal_reward_pd
		
    state_trans_pd = model_funcs.state_trans_pd
    state_trans = model_funcs.state_trans
    exploration = model_funcs.exploration

    eval_equations_FOC = model_funcs.eval_equations_FOC

    
    
        
        
            
            
            