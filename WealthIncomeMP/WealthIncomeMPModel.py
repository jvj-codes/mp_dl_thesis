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

class WealthIncomeModelClass(DLSolverClass):
    
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
        par.bequest = 0                 
        
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
        par.Ra_bar = 1.05 #
        par.phi_Ra = 0.8 #high impact
        par.rho_a = 0.15 #low persistence
        par.gamma = 0.05 #survey
        
        # a. 6 Debt
        par.phi = 2.5    #loan-to-value
        par.lbda = 0.05  #Repayment
        
        # a.7 Wage
        par.w_bar = 10   # This is purely a scaling choice and does not affect the model’s qualitative dynamics.
        par.rho_w = 0.75 #persistence of income
        par.theta = 0.10 #Andersen, A.L., Johannesen, N., Jørgensen, M. & Peydró, J.-L. (2021). Monetary Policy and Inequality.

        # a. 8 moments for shocks
        par.psi_sigma = 0.08             #wage shock (std. dev) Moira Daly, Dmytro Hryshko, and Iourii Manovskii1 (2022)
        par.psi_mu = 0.00                #wage shock mean
        par.Npsi = 4                     #wage shock quadrature nodes   
        
        par.e_sigma = 0.04               #illiquid returns shock (std. dev) 
        par.e_mu = 0.00                  #illiquid returns shock (mean)
        par.Ne = 4                       #illiquid returns shock quadrature nodes
        
        par.R_sigma = 9e-4               #monetary policy shock (std. dev), Wieland and Yang (2016)
        par.R_mu = 0.0                   #monetary policy shock (mean)
        par.NR = 4 #4                    #monetary policy shock quadrature nodes
        
        par.pi_sigma = 0.02              #inflation shock (std. dev) Standard value
        par.pi_mu = 0.00                 #inflation shock (mean)
        par.Npi  = 4 #4                  #inflation shock quardrature nodes
    
        # a.9 initial states
        par.pi_min = 1.02 # minimum inflation of 2%, DST data
        par.pi_max = 1.03 # maximum inflation of 3%, DST data
        
        par.R_min = 1.025  # minimum interest rate of 2%, DST data
        par.R_max = 1.045  # maximum interest rate of 4%, DST data

        par.R_a0_min = 1.02
        par.R_a0_max = 1.04  
        
        par.sigma_m0 = 0.5226 #DST data
        par.mu_m0 = 0.1446 #DST data
        
        par.sigma_w0 = 0.1446 #DST data
        par.mu_w0 = -6.7336 #DST data
        
        par.sigma_a0 = 0.1446 #DST data
        par.mu_a0 = -0.6901 #DST data
        
        par.sigma_d0 = 0.1446 #DST data
        par.mu_d0 = -6.7336 #DST data
        
        par.mu_beta = 0.975
        par.sigma_beta = 0.005
        
        # b. solver settings
        
        # minimum savings calculation for euler errors
        par.Euler_error_min_savings = 1e-3
        
        # states, actions and shocks
        par.Nstates_fixed = 0 # 1 = heterogenous beta, 0 = fixed beta
        par.Nshocks = 4
        par.Nactions = 4
        
        # outcomes and actions
        par.Noutcomes = 6 # consumption, house holdings, labor, funds
        par.KKT = False ## use KKT conditions (for DeepFOC)
        par.NDC = 0 # number of discrete choices
        
        # c. simulation
        sim.N = 15_000  # number of agents 
        sim.reps = 0 # number of repetitions
        sim.sim_R_freq = 50 # reward frequency every 50 reps
        
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
            sim.N = 15_000 #5000 #500 #1000 #5000 #10_000
            
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
                train.epsilon_sigma = np.array([0.1,0.1,0.1,0.1]) 
                train.epsilon_sigma_min = np.array([0.0,0.0,0.0,0.0])
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

    
    def draw_shocks(self,N):
      
      par = self.par
      
      psi = torch.normal(mean=par.psi_mu,std=par.psi_sigma, size=(par.T,N,))
      epsn_R = torch.normal(mean=par.R_mu, std=par.R_sigma, size=(par.T,N,))
      epsn_pi = torch.normal(par.pi_mu, par.pi_sigma, size=(par.T,N,))
      epsn_e = torch.normal(par.e_mu, par.e_sigma, size=(par.T,N,))
      return torch.stack((psi, epsn_R, epsn_pi, epsn_e),dim=-1) 

    
    def draw_initial_states(self,N,training=False): #atm uniform
        
        par = self.par
        
        # a. draw initial illiquid assets
        a0 = torch.exp(torch.normal(mean=par.mu_a0, std=par.sigma_a0, size=(N,))) 
        
        # b. draw initial inflation pi
        pi0 = torch_uniform(par.pi_min, par.pi_max, size = (N,))
        
        # c. draw initial nominal interest rate
        R0 = torch_uniform(par.R_min, par.R_max, size = (N,))
        
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

        sim.states[1,:,1] += transfer #cash on hand

    def compute_euler_errors(self,Nbatch_share=0.01):
        """ compute euler error"""

        par = self.par
        sim = self.sim
        train = self.train

        Nbatch = int(Nbatch_share*sim.N)


        for i in range(0,sim.N,Nbatch):

            index_start = i
            index_end = i + Nbatch

            with torch.no_grad():
                # a. get states, actions, outcomes
                states = sim.states[:par.T-1,index_start:index_end]
                a_prev, w, m, pi, R_b, R_e, d_prev = [states[..., i] for i in range(7)]
                
                actions = sim.actions[:par.T-1,index_start:index_end]
                n_act, alpha_e, alpha_b, alpha_d = [actions[..., i] for i in range(4)]

                outcomes = sim.outcomes[:par.T-1,index_start:index_end]
                c, n, s, b, a_pd, d_pd  = [outcomes[..., i] for i in range(6)]
                
                s = torch.clamp(s, min=1e-3)
                c = torch.clamp(c, min=1e-3)
                b = torch.clamp(b, min=1e-3)
                a_pd = torch.clamp(a_pd, min=1e-3)

                # b. post-decision states
                states_pd = self.state_trans_pd(states,actions,outcomes)

                # c. next-period states
                states_plus = self.state_trans(states_pd,train.quad)
                a, w_plus, m_plus, pi_plus, R_b_plus, R_e_plus, d = [states_plus[..., i] for i in range(7)]

                if par.Nstates_fixed > 0:
                    beta = states_plus[..., -1]  # assume beta is last
                else:
                    beta = par.beta
                    
                # d. actions next-period
                actions_plus = self.eval_policy(self.policy_NN,states_plus[:par.T-1],t0=1)
                n_act_plus, alpha_e_plus, alpha_b_plus, alpha_d_plus = [actions_plus[..., i] for i in range(4)]

                # e. next-period outcomes
                outcomes_plus = self.outcomes(states_plus,actions_plus)
                c_plus, n_plus, s_plus, b_plus, a_pd_plus, d_pd_plus = [outcomes_plus[..., i] for i in range(6)]
                
                s_plus = torch.clamp(s_plus, min=1e-3)
                c_plus = torch.clamp(c_plus, min=1e-3, max=10)
                b_plus = torch.clamp(b_plus, min=1e-3)
                a_pd_plus = torch.clamp(a_pd_plus, min=1e-3)
                
                # f. marginal utility current and next period
                marg_util_c_t = model_funcs.marg_util_c(c, par)
                marg_util_n_t = model_funcs.marg_util_n(n, par)

                marg_util_c_plus = model_funcs.marg_util_c(c_plus, par)

                # --- f. Expand relevant variables for time t ---
                kappa = (s - m) / (n_act * w )
                expand_vars = [a_pd, a_prev, R_b, w, alpha_e, alpha_b]
                a_pd_exp, a_prev_exp, R_b_exp, alpha_e_exp, alpha_b_exp, kappa_exp = [misc.expand_dim(v, R_b_plus) for v in expand_vars]
                
                # --- g. Derive FOCS ---
                #  Illiquid alpha_t^e share FOC
                term_e = (R_e_plus/pi_plus -1) - 2*par.gamma*(a_pd_exp - a_prev_exp) + 2*par.gamma*(a_pd_plus - a)
                rhs_e = torch.sum(train.quad_w[None, None, :]*term_e* marg_util_c_plus, dim=-1)
                lhs_e = marg_util_c_t
                euler_err_illiquid = beta[:-1] * rhs_e / lhs_e[:-1] -1

                # Liquid alpha_t^b share FOC
                term_b = R_b_plus / pi_plus - (R_e_plus/pi_plus -1)*alpha_e_exp + alpha_e_exp * 2*par.gamma*(a_pd_plus - a)
                rhs_b = torch.sum(train.quad_w[None, None, :]*term_b* marg_util_c_plus, dim=-1)
                lhs_b = (1-alpha_e) * marg_util_c_t
                euler_err_bond = beta[:-1] * rhs_b / lhs_b[:-1] -1
                
                # Labor supply n_t share FOC
                term_n = alpha_b_exp * (R_b_plus) / pi_plus + (1-alpha_b_exp)*alpha_e_exp*((R_e_plus/pi_plus-1) - 2*par.gamma*(a_pd_exp - a_prev_exp) )
                rhs_n = torch.sum(train.quad_w[None, None, :]*term_n* marg_util_c_plus, dim=-1)
                lhs_n = marg_util_n_t/w*kappa + marg_util_c_t * (1-alpha_e)*(1-alpha_b)
                euler_err_labor =  beta[:-1] * rhs_n / lhs_n[:-1] -1
                
                # Debt allocation alpha_t^d
                term_d = alpha_b_exp * (R_b_plus) / pi_plus + (1-alpha_b_exp)*alpha_e_exp*(R_e_plus/pi_plus-1) - (par.lbda + (R_b_plus/pi_plus  -1))
                rhs_d = torch.sum(train.quad_w[None, None, :]*term_d*marg_util_c_plus, dim=-1)
                lhs_d = marg_util_c_t * (1-alpha_e)*(1-alpha_b)
                euler_err_debt = beta[:-1] * rhs_d / lhs_d[:-1] -1
                
                # j. Batch Euler Error
                euler_error_Nbatch = torch.stack((euler_err_illiquid, euler_err_bond, euler_err_labor, euler_err_debt), dim =-1)
                sim.euler_error[:par.T-1,index_start:index_end] = euler_error_Nbatch

def select_euler_errors(model):
    """ compute mean euler error """

    par = model.par
    sim = model.sim

    savings_indicator = (sim.actions[:par.T_retired,:,2] > par.Euler_error_min_savings)
    return sim.euler_error[:par.T_retired,:][savings_indicator]

def mean_log10_euler_error_working(model):

    euler_errors = select_euler_errors(model).cpu().numpy()
    I = np.isclose(np.abs(euler_errors),0.0)

    return np.mean(np.log10(np.abs(euler_errors[~I])))