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
from EconDLSolvers import DLSolverClass, torch_uniform, compute_transfer
from model_funcs import terminal_actions

class WealthIncomeModelClass_CaseStudy(DLSolverClass):
    
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
        par.T = 75         #lifetime (working + retirement)
        par.T_retired = 49 #retirement at year
        
        # a.1 Household parameters
        par.beta = None           #discount factor
        par.eta = 1.1             
        par.sigma_int = 3         #inverse of intertemporal elasticity of consumption and house holdings
        par.nu = 1.5              #disutility factor for labor supply
        par.bequest = 1                 
        
        # a.2 Efficiency & retired time allocation
        par.kappa_base = 1              # income base
        par.kappa_growth = 0.0000       # income growth #kappa before 0.03
        par.kappa_growth_decay = 0.0001 # income growth decay
        par.n_retired = 0.1             #fixed part-time work at retirment
        
        # a.3 Monetary policy 
        par.rho_R = 0.35          #inertia of nominal interest rate: Braun & Ikeda (2025)
        par.R_bar = 1.005         #target nominal interest rate: Ralph Luetticke (2020)
        par.phi_pi = 1.5          #stance on inflation: Ralph Luetticke (2020)
        
        # a.4 Inflation
        par.pi_bar = 1.02         #target inflation: ECB (2025)
        par.rho_pi = 0.32         #inertia of inflation: Danish data
        par.phi_R = 0.015         #impact on inflation: Danish data
        
        # a.5 Iliquid return & cost
        par.Ra_bar = 1.12 # 1.07
        par.phi_Ra = 2.2 #high impact 0.8
        par.rho_a = 0.15 #low persistence
        par.gamma = 0.02 #survey 0.04
        
        # a. 6 Debt
        par.phi = 0.55    #loan-to-value
        par.lbda = 0.05  #Repayment
        par.eps_rp = 0.06 #risk premium
        
        # a.7 Wage
        par.w_bar = 10   # This is purely a scaling choice and does not affect the model’s qualitative dynamics.
        par.rho_w = 0.91 #persistence of income
        par.theta = 1.72 #Andersen, A.L., Johannesen, N., Jørgensen, M. & Peydró, J.-L. (2021). Monetary Policy and Inequality.

        # a. 8 moments for shocks
        par.psi_sigma = 0.08             #wage shock (std. dev) Moira Daly, Dmytro Hryshko, and Iourii Manovskii1 (2022)
        par.psi_mu = 0.05                #wage shock mean
        par.Npsi = 5                     #wage shock quadrature nodes   
        
        par.e_sigma = 0.005               #illiquid returns shock (std. dev)
        par.e_mu = 0.00                  #illiquid returns shock (mean)
        par.Ne = 5                     #illiquid returns shock quadrature nodes
        
        par.R_sigma = 9e-4               #monetary policy shock (std. dev), Wieland and Yang (2016)
        par.R_mu = 0.0                   #monetary policy shock (mean)
        par.NR = 5 #4                    #monetary policy shock quadrature nodes
        
        par.pi_sigma = 0.005              #inflation shock (std. dev) Standard value
        par.pi_mu = 0.00                 #inflation shock (mean)
        par.Npi  = 5 #4                  #inflation shock quardrature nodes
    
        # a.9 initial states
        par.pi0 = 1.02 # minimum inflation of 2%, DST data
        
        par.R0 = 1.03  # minimum interest rate of 2%, DST data

        par.R_a0_min = 1.02
        par.R_a0_max = 1.04

        par.R_a0_min = 1.02
        par.R_a0_max = 1.04

        # Money holdings m₀ ∼ LogN(μ_m0, σ_m0²)
        par.mu_m0 = 0.5822
        par.sigma_m0 = 1.6921/2
        
        # Wage w₀ ∼ LogN(μ_w0, σ_w0²)
        par.sigma_w0 = 0.1446   # from pooled LONS60 fit
        par.mu_w0    = -0.0104  # ln(median_wage) – shift  (E[w]=1)
        
        # Assets a₀ ∼ LogN(μ_a0, σ_a0²)
        par.sigma_a0 = 1.6282/2   # same as wage 
        par.mu_a0 = 0.3871

        
        # Debt d₀ ∼ LogN(μ_d0, σ_d0²) FORUME111
        par.sigma_d0 = 0.5223  
        par.mu_d0    = -3   
        
        par.mu_beta = 0.975
        par.sigma_beta = 0.005

        # a. 10 shock periods for MP
        par.shock_periods = None
        par.experiment = None
        par.R_shock = -0.01 #1 percentage monetary policy
        
        # b. solver settings
        
        # minimum savings calculation for euler errors
        par.Euler_error_min_savings = 1e-3
        par.Delta_MPC = 1e-4 # windfall used in MPC calculation
        
        # states, actions and shocks
        par.Nstates_fixed = 1 # 1 = heterogenous beta, 0 = fixed beta
        par.Nshocks = 4
        
        # outcomes and actions
        par.Noutcomes = 6 # consumption, house holdings, labor, funds
        par.KKT = False ## use KKT conditions (for DeepFOC)
        par.NDC = 0 # number of discrete choices
        
        # c. simulation
        sim.N = 10_000  # number of agents
        sim.reps = 0 # number of repetitions
        sim.sim_R_freq = 50 # reward frequency every 50 reps
        sim.MPC = torch.zeros((par.T,sim.N),dtype=dtype,device=device)

    
        
    def allocate(self):
        
        # a. unpack
        par = self.par
        sim = self.sim
        train = self.train
        
        dtype = train.dtype
        device = train.device
        
        if not par.full: # for solving without GPU
            par.T = 75 #lifetime (working + retirement)
            par.T_retired = 49 #retirement at year
            sim.N = 10_000 #5000 #500 #1000 #5000 #10_000
            
        if par.KKT:
            par.Nactions = 5
        else:
            par.Nactions = 4
        
        if par.Nstates_fixed == 1:
            par.Nstates = 8
            par.Nstates_pd = 10
        else:
            par.Nstates = 7
            par.Nstates_pd = 9
            
        # b. life cycle income
        par.kappa = torch.zeros(par.T,dtype=dtype,device=device)
        par.kappa[0] = par.kappa_base
    
        for t in range(1,par.T):
            par.kappa[t] = par.kappa[t-1]*(1+par.kappa_growth)*(1-par.kappa_growth_decay)**(t-1)
        
        # c. quadrature
        par.psi, par.psi_w  = misc.normal_gauss_hermite(sigma = par.psi_sigma, n = par.Npsi, mu = par.psi_mu)
        par.R, par.R_w      = misc.normal_gauss_hermite(sigma = par.R_sigma, n = par.NR, mu = par.R_mu) #returns nodes of length n, weights of length n
        par.pi, par.pi_w    = misc.normal_gauss_hermite(sigma = par.pi_sigma, n = par.Npi, mu = par.pi_mu)
        par.e, par.e_w      = misc.normal_gauss_hermite(sigma = par.e_sigma, n = par.Ne, mu = par.e_mu) 


        par.psi_w  = torch.tensor(par.psi_w,dtype=dtype,device=device)
        par.psi = torch.tensor(par.psi,dtype=dtype,device=device)
        
        par.R_w  = torch.tensor(par.R_w,dtype=dtype,device=device)
        par.R = torch.tensor(par.R,dtype=dtype,device=device)
        
        par.pi_w  = torch.tensor(par.pi_w,dtype=dtype,device=device)
        par.pi = torch.tensor(par.pi,dtype=dtype,device=device)
        
        par.e_w  = torch.tensor(par.e_w,dtype=dtype,device=device)
        par.e = torch.tensor(par.e,dtype=dtype,device=device)
        
        
        # d. simulation
        sim.states = torch.zeros((par.T,sim.N,par.Nstates),dtype=dtype,device=device)
        sim.states_pd = torch.zeros((par.T,sim.N,par.Nstates_pd),dtype=dtype,device=device)
        sim.shocks = torch.zeros((par.T,sim.N,par.Nshocks),dtype=dtype,device=device)
        sim.outcomes = torch.zeros((par.T,sim.N,par.Noutcomes),dtype=dtype,device=device)
        sim.actions = torch.zeros((par.T,sim.N,par.Nactions),dtype=dtype,device=device)
        sim.reward = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
        sim.euler_error = torch.zeros((par.T-1,sim.N,par.Nactions),dtype=dtype,device=device)
        sim.R = np.nan
        
    def setup_train(self):
        
        par = self.par
        train = self.train
        dtype = train.dtype
        device = train.device
        
        # a. neural network
        if not par.full:
            train.Neurons_policy = np.array([500,500,500])
            train.Nneurons_value = np.array([500,500,500])
            
        # gradient clipping
        train.clip_grad_policy = 10 
        train.clip_grad_value = 10 
            
        # b. policy activation functions and clipping
        if par.KKT:
            ## n, alpha_e, alpha_b
            train.policy_activation_final = ['sigmoid', 'sigmoid', 'sigmoid','sigmoid','softplus']
            train.min_actions = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000],dtype=dtype,device=device) #minimum action value n, alpha_d, alpha_e, alpha_b, alpha_h
            train.max_actions = torch.tensor([0.9999, 0.9999, 0.9999, 0.9999, np.inf],dtype=dtype,device=device) #maximum action value n, alpha_d, alpha_e, alpha_b, alpha_h
            
        else:
            ## n, alpha_e, alpha_b, alpha_d
            train.policy_activation_final = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']
            train.min_actions = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000],dtype=dtype,device=device) #minimum action value n, alpha_d, alpha_e, alpha_b, alpha_h
            train.max_actions = torch.tensor([1.0-1e-6, 1.0-1e-6, 1.0-1e-6, 1.0-1e-6],dtype=dtype,device=device) #maximum action value n, alpha_d, alpha_e, alpha_b, alpha_h

        # c. misc
        train.terminal_actions_known = False # use terminal actions
        train.use_input_scaling = False # use input scaling
        train.do_exo_actions_periods = 10
        train.Delta_time = 10   
        train.Delta_transfer = 5e-5  
        train.K_time_min = 15   
        train.convergence_plot=True

        if train.algoname == 'DeepVPD':
            train.learning_rate_value_decay = 0.999
            train.learning_rate_policy_decay = 0.999
            train.learning_rate_policy = 1e-6
            train.learning_rate_value = 1e-6
            train.start_train_policy = 50          
          
        if train.algoname == 'DeepFOC':
            train.eq_w = torch.tensor([3.0, 3.0, 3.0, 3.0, 5.0],dtype=dtype,device=device) 
            
        if train.algoname == 'DeepSimulate':
            train.learning_rate_policy = 1e-4
        else:
            if par.KKT:
                train.epsilon_sigma = np.array([0.1,0.1,0.1,0.1,0.1]) 
                train.epsilon_sigma_min = np.array([0.0,0.0,0.0,0.0,0.0])
            else:
                train.epsilon_sigma = np.array([0.1,0.1,0.1,0.1]) 
                train.epsilon_sigma_min = np.array([0.0,0.0,0.0,0.0])
            
    def allocate_train(self):
        
        par = self.par
        train = self.train
        dtype = train.dtype
        device = train.device

        if train.algoname == 'DeepFOC':
            train.eq_w = train.eq_w / torch.sum(train.eq_w) # normalize
             
        train.states = torch.zeros((par.T,train.N,par.Nstates),dtype=dtype,device=device)
        train.states_pd = torch.zeros((par.T,train.N,par.Nstates_pd),dtype=dtype,device=device)
        train.shocks = torch.zeros((par.T,train.N,par.Nshocks),dtype=dtype,device=device)
        train.outcomes = torch.zeros((par.T,train.N,par.Noutcomes),dtype=dtype,device=device)
        train.actions = torch.zeros((par.T,train.N,par.Nactions),dtype=dtype,device=device)
        train.reward = torch.zeros((par.T,train.N),dtype=dtype,device=device)
        
    def quad(self):
        
        par = self.par
        
        psi,R,pi,e = torch.meshgrid(par.psi,par.R,par.pi,par.e, indexing='ij')
        
        psi_w,R_w,pi_w,e_w= torch.meshgrid(par.psi_w,par.R_w,par.pi_w,par.e_w, indexing='ij')
        
        quad = torch.stack((psi.flatten(), R.flatten(), pi.flatten(), e.flatten()), dim=1)
        quad_w = psi_w.flatten() * R_w.flatten() * pi_w.flatten() * e_w.flatten()
        
        return quad, quad_w

    
    def draw_shocks(self, N):
        par = self.par
        torch.manual_seed(1337)

        def apply_pct_shock(eps, shock_periods, pct=0.01):
            """
            Add +pct to eps in each period or period-range in shock_periods.
        
            eps : torch.Tensor, shape (T,) or (T, N)
            shock_periods : iterable of ints or (start,end) tuples
            pct : float, e.g. 0.01 for +1 pp
            """
            eps = eps.clone()
            # prepare the delta as a tensor matching eps’s dtype & device
            delta = torch.tensor(pct, dtype=eps.dtype, device=eps.device)
        
            for p in shock_periods:
                if isinstance(p, (tuple, list)):
                    s, e = p
                    # for 1-D eps this will broadcast correctly
                    eps[s:e, ...] += delta
                else:
                    t = p
                    eps[t, ...] += delta
        
            return eps


        # 1) idiosyncratic
        psi    = torch.normal(par.psi_mu,    par.psi_sigma, size=(par.T, N))
        epsn_e = torch.normal(par.e_mu,      par.e_sigma,   size=(par.T, N))

        # 2) policy‐rate: draw baseline, then overlay +1 percentage‐point
        epsn_R = torch.normal(par.R_mu, par.R_sigma, size=(par.T,))
        if par.experiment == "R1pp":
            epsn_R = apply_pct_shock(epsn_R, par.shock_periods, par.R_shock)
        epsn_R = epsn_R.unsqueeze(1).expand(-1, N).clone()

        # 3) inflation
        epsn_pi = torch.normal(par.pi_mu, par.pi_sigma, size=(par.T,))
        epsn_pi = epsn_pi.unsqueeze(1).expand(-1, N).clone()

        # 4) stack into (T, N, 4)
        return torch.stack((psi, epsn_R, epsn_pi, epsn_e), dim=-1)


    
    def draw_initial_states(self,N,training=False): #atm uniform
        
        par = self.par
        
        # a. draw initial illiquid assets
        a0 = torch.exp(torch.normal(mean=par.mu_a0, std=par.sigma_a0, size=(N,))) 
        
        # b. draw initial inflation pi
        pi0 = torch.ones(N)*par.pi0
        
        # c. draw initial nominal interest rate
        R0 = torch.ones(N)*par.R0
        
        # d. draw initial illiquid assets
        R_a0 = torch_uniform(par.R_a0_min, par.R_a0_max, size = (N,))
        
        # e. draw initial wage
        w0 = torch.exp(torch.normal(mean = par.mu_w0, std=par.sigma_w0, size=(N,))) 
        
        # f. draw initial money holdings
        m0 = torch.exp(torch.normal(mean= par.mu_m0, std=par.sigma_m0, size=(N,))) 
        
        # g. draw initialdebt
        d0 = torch.exp(torch.normal(mean= par.mu_d0, std=par.sigma_d0, size=(N,))) 
        
        if par.Nstates_fixed > 0: 
            raw_beta = torch.normal(mean=par.mu_beta, std=par.sigma_beta, size=(N,))
            beta = torch.clamp(raw_beta, min= 0.95, max=0.99)

        # g. store
        if par.Nstates_fixed == 0:
            return torch.stack((a0, w0, m0, pi0, R0, R_a0, d0),dim=-1)
        elif par.Nstates_fixed > 0:
            return torch.stack((a0, w0, m0, pi0, R0, R_a0, d0, beta),dim=-1)
    
    def draw_exploration_shocks(self,epsilon_sigma,N):
        """ draw exploration shockss """ 
        par = self.par 
        
        eps = torch.zeros((par.T,N,par.Nactions)) 
        for i_a in range(par.Nactions): 
            eps[:,:,i_a] = torch.normal(0,epsilon_sigma[i_a],(par.T,N)) 
            
        return eps
    
    def draw_exo_actions(self,N):
        """ draw exogenous actions """

        par = self.par

        exo_actions = torch_uniform(0.01,0.8,size=(par.T,N,par.Nactions))
    
        return exo_actions

    outcomes = model_funcs.outcomes
    reward = model_funcs.reward
    discount_factor = model_funcs.discount_factor
    terminal_actions = model_funcs.terminal_actions
    terminal_reward_pd = model_funcs.terminal_reward_pd
    state_trans_pd = model_funcs.state_trans_pd
    state_trans = model_funcs.state_trans
    exploration = model_funcs.exploration
    eval_equations_FOC = model_funcs.eval_equations_FOC

    def add_transfer(self,transfer):
        """ add transfer to initial states """
        par = self.par
        sim = self.sim

        sim.states[2,:,2] += transfer #cash on hand

    def compute_MPC(self,Nbatch_share=0.01):
        """ compute MPC """
        par = self.par
        sim = self.sim
        with torch.no_grad():

            # a. baseline
            c = sim.outcomes[...,0]
            
            # b. add windfall
            states = deepcopy(sim.states)
            states[:,:,2] += par.Delta_MPC  #add cash on hand
            
            # b. alternative
            c_alt = torch.ones_like(c)
            c_alt[-1] = c[-1] + par.Delta_MPC
            Nbatch = int(Nbatch_share*sim.N)
            for i in range(0,sim.N,Nbatch):
                index_start = i
                index_end = i + Nbatch
                states_ = states[:par.T-1,index_start:index_end,:]
                actions = self.eval_policy(self.policy_NN,states_)
                outcomes = self.outcomes(states_,actions)
                c_alt[:par.T-1,index_start:index_end] = outcomes[:,:,0]

            # c. MPC
            sim.MPC[:,:] = (c_alt-c)/par.Delta_MPC

