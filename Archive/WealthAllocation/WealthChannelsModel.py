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
        sim.sim_R_freq = 50
        
        # a. model
        par.T = 75         #lifetime (working + retirement)
        par.T_retired = 49 #retirement at year
        
        ## parameters of the model
        par.beta = None           #discount factor
        
        par.j = 0.01              #relative preference weight of house holdings
        par.sigma_int = 3               #inverse of intertemporal elasticity of consumption and house holdings
        
        par.A = 1.3               #taylor rule coefficient
        par.R_star = 0.03         #target nominal interest rate
        
        par.pi_star = 0.02        #target inflation rate #baseline 0.02
        par.rhopi = 0.318         #persistence of inflation
        par.Rpi = 0.05            #impact of interest rate on inflation lagged
        
        par.w_bar = math.log(10)            #wage growth  #basline 1.06
        par.rho_w = 0.85          #wage persistence 
        par.vartheta = 2.50       #fraction of wealth to determine limit of debt
        par.eta = 1.1             #labor supply schedule
        par.theta = 0.25          #sensitivity deviation from target interest rate wage
        par.n_retired = 0.1       #fixed part-time work at retirment
        par.nu = 1.5              #disutility factor for labor supply
        
        par.lbda = 0.05           #minimum payment due
        par.eps_rp = 0.06         #risk premium on nominal interest rate
        
        par.rho_q = 0.90          #persistence in house prices
        par.q_h = 0.75            #sensitivity to interest rate devations house prices 
        par.q_bar = math.log(2)                #baseline level house price
        par.tau = 0.005            #cost parameter

        ### income
        par.kappa_base = 1              # income base
        par.kappa_growth = 0.0000       # income growth #kappa before 0.03
        par.kappa_growth_decay = 0.0001 # income growth decay
        
        ##testing
        par.bequest = 0                 
        par.Euler_error_min_savings = 1e-3

        
        ## mean and std of shocks
        par.psi_sigma = 0.02              #wage shock (std std.) logal distributed
        par.psi_mu = 0                #wage shock mean
        par.Npsi = 4                     #quadrature nodes   
        
        par.Q_mu = 0.06                  #equity returns shock (mean)
        par.Q_sigma = 0.01 #3            #equity returns shock (std. dev) 
        par.NQ = 4 #4                    #equity returns shock quadrature nodes
        
        par.R_sigma = 0.0005             #monetary policy shock (std. dev) log normal mean = 0
        par.R_mu = 1 - math.log(100)
        par.NR = 4 #4                    #monetary policy quadrature nodes
        
        par.pi_sigma = 0.005              #inflation shock (std. dev) distributed
        par.pi_mu = 0.00
        par.Npi  = 4 #4                  #inflation shock quardrature nodes
        
        par.h_sigma = 0.02                  #house price shock (std. dev) normal dis
        par.h_mu = 0                     #house price shock mean 
        par.Nh = 4

        
        ## initial states
        
        par.pi_min = 0.02 # minimum inflation of 1% 
        par.pi_max = 0.03 # maximum inflation of 3%
        
        par.R_min = 0.02  # minimum interest rate of 2%
        par.R_max = 0.04  # maximum interest rate of 5%

        par.R_e0_min = 0.01  # minimum interest rate of 2%
        par.R_e0_max = 0.04  # maximum interest rate of 5%
        
        par.sigma_m0 = 0.1 
        par.sigma_w0 = 0.2 
        
        par.d0_alpha = 0.7
        par.d0_beta = 3.0

        par.beta_min = 0.95
        par.beta_max = 0.99

        par.mu_beta = 0.975
        par.sigma_beta = 0.005

        par.mu_q0 = 100
        par.sigma_q0 = 1
        
        # b. solver settings
        
        # states, actions and shocks
        par.Nstates_fixed = 0
            
        par.Nshocks = 5
        par.Nactions = 5
        
        # outcomes and actions
        par.Noutcomes = 7 # consumption, house holdings, labor, funds
        par.KKT = False ## use KKT conditions (for DeepFOC)
        par.NDC = 0 # number of discrete choices
        
        # c. simulation
        sim.N = 15_000  # number of agents 
        sim.reps = 0 # number of repetitions
        
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
            par.Nactions = 7 #revisit when focs are done
        else:
            par.Nactions = 5
            #par.Nactions = 4
        
        if par.Nstates_fixed == 1:
            par.Nstates = 9
            par.Nstates_pd = 11 
        else:
            par.Nstates = 8
            par.Nstates_pd = 10 
            
        # b. life cycle income
        par.kappa = torch.zeros(par.T,dtype=dtype,device=device)
        par.kappa[0] = par.kappa_base
    
        for t in range(1,par.T):
            par.kappa[t] = par.kappa[t-1]*(1+par.kappa_growth)*(1-par.kappa_growth_decay)**(t-1)
        
        # c. quadrature
        par.psi, par.psi_w  = misc.log_normal_gauss_hermite(sigma = par.psi_sigma, n = par.Npsi, mu = par.psi_mu)
        par.R, par.R_w      = misc.log_normal_gauss_hermite(sigma = par.R_sigma, n = par.NR, mu = par.R_mu + math.log(100)) #returns nodes of length n, weights of length n
        par.R  /= 100
        par.pi, par.pi_w    = misc.normal_gauss_hermite(sigma = par.pi_sigma, n = par.Npi, mu = par.pi_mu)
        par.q, par.q_w      = misc.normal_gauss_hermite(sigma = par.Q_sigma, n = par.NQ, mu = par.Q_mu) 
        par.h, par.h_w      = misc.normal_gauss_hermite(sigma = par.h_sigma, n = par.Nh, mu = par.h_mu)

        

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
            ## n, alpha_d, alpha_e, alpha_b, alpha_h
            train.policy_activation_final = ['sigmoid', 'sigmoid', 'sigmoid','sigmoid','sigmoid','softplus','softplus']
            train.min_actions = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],dtype=dtype,device=device) #minimum action value n, alpha_d, alpha_e, alpha_b, alpha_h
            train.max_actions = torch.tensor([0.9999, 0.9999, 0.9999, 0.9999, 0.9999, np.inf, np.inf],dtype=dtype,device=device) #maximum action value n, alpha_d, alpha_e, alpha_b, alpha_h
            
        else: 
            ## n, alpha_d, alpha_e, alpha_b, alpha_h
            train.policy_activation_final = ['sigmoid', 'sigmoid', 'sigmoid','sigmoid','sigmoid']
            
            train.min_actions = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000],dtype=dtype,device=device) #minimum action value n, alpha_d, alpha_e, alpha_b, alpha_h
            train.max_actions = torch.tensor([1.0-1e-6, 1.0-1e-6, 1.0-1e-6, 1.0-1e-6, 1.0-1e-6],dtype=dtype,device=device) #maximum action value n, alpha_d, alpha_e, alpha_b, alpha_h

        # c. misc
        train.terminal_actions_known = False # use terminal actions
        train.use_input_scaling = False # use input scaling
        train.do_exo_actions_periods = 10
        train.Delta_time         = 10   
        train.Delta_transfer     = 5e-5  
        train.K_time_min   = 15   

        if train.algoname == 'DeepVPD':
            train.learning_rate_value_decay = 0.999
            train.learning_rate_policy_decay = 0.999
            train.learning_rate_policy = 1e-6
            train.learning_rate_value = 1e-6
            train.start_train_policy = 50          
          
        if train.algoname == 'DeepFOC':
            train.eq_w = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0],dtype=dtype,device=device) 
            
        if train.algoname == 'DeepSimulate':
            train.learning_rate_policy = 1e-4
        else:
            if par.KKT:
                train.epsilon_sigma = np.array([0.1,0.1,0.1,0.1,0.1,0.0,0.0]) 
                train.epsilon_sigma_min = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            else:
                train.epsilon_sigma = np.array([0.1,0.1,0.1,0.1,0.1]) 
                train.epsilon_sigma_min = np.array([0.0,0.0,0.0,0.0,0.0])
        # convergence plot
        train.convergence_plot=True
            
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
        
        psi,R,pi,q,h = torch.meshgrid(par.psi,par.R,par.pi,par.q,par.h, indexing='ij')
        
        psi_w,R_w,pi_w,q_w,h_w = torch.meshgrid(par.psi_w,par.R_w,par.pi_w,par.q_w,par.h_w, indexing='ij')
        
        quad = torch.stack((psi.flatten(), R.flatten(), pi.flatten(), q.flatten(), h.flatten()), dim=1)
        quad_w = psi_w.flatten() * R_w.flatten() * pi_w.flatten() * q_w.flatten() * h_w.flatten()
        
        return quad, quad_w

    
    def draw_shocks(self,N):
      
      par = self.par
      
      psi = torch.normal(mean=par.psi_mu,std=par.psi_sigma, size=(par.T,N,))
      epsn_R = torch.exp(torch.normal(mean=par.R_mu, std=par.R_sigma, size=(par.T,N,)))
      epsn_pi = torch.normal(par.pi_mu, par.pi_sigma, size=(par.T,N,))
      epsn_q = torch.normal(par.Q_mu, par.Q_sigma, size=(par.T,N,))
      epsn_h = torch.normal(par.h_mu, par.h_sigma, size=(par.T,N,))
      return torch.stack((psi, epsn_R, epsn_pi, epsn_q, epsn_h),dim=-1) 

    
    def draw_initial_states(self,N,training=False): #atm uniform
        
        par = self.par
        
        # a. draw initial inflation pi
        pi0 = torch_uniform(par.pi_min, par.pi_max, size = (N,))
        
        # b. draw initial nominal interest rate
        R0 = torch_uniform(par.R_min, par.R_max, size = (N,))
        
        # c. draw initial equity return
        R_e0 = torch_uniform(par.R_e0_min, par.R_e0_max, size = (N,))
        
        # d. draw initial wage
        #w0 = torch.exp(torch.normal(mean=25*par.sigma_w0**2, std=par.sigma_w0, size=(N,))) 
        w0 = torch.exp(torch.normal(mean = 0.5, std=par.sigma_w0, size=(N,))) 
        
        # e. draw initial money holdings
        m0 = torch.exp(torch.normal(mean=-0.1*par.sigma_m0**2, std=par.sigma_m0, size=(N,))) 
        
        # f. draw initial debt
        d0 = torch.distributions.Beta(par.d0_alpha, par.d0_beta).sample((N,))
        
        # g. draw initial house prices
        q0 = torch.normal(mean=0, std=par.sigma_q0, size=(N,)) 
        
        # h. draw initial house holdings
        h0 = torch.zeros(size=(N,))


        if par.Nstates_fixed > 0: 
            raw_beta = torch.normal(mean=par.mu_beta, std=par.sigma_beta, size=(N,))
            beta = torch.clamp(raw_beta, min= 0.95, max=0.99)

        # h. store
        if par.Nstates_fixed == 0:
            return torch.stack((w0,m0,d0,q0,pi0,R0,R_e0,h0),dim=-1)
        elif par.Nstates_fixed > 0:
            return torch.stack((w0,m0,d0,q0,pi0,R0,R_e0,h0,beta),dim=-1)
    
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
                w, m, d, q, pi, R_b, R_e = [states[..., i] for i in range(7)]

                actions = sim.actions[:par.T-1,index_start:index_end]
                n_act, alpha_d, alpha_e, alpha_b, alpha_h = [actions[..., i] for i in range(5)]

                outcomes = sim.outcomes[:par.T-1,index_start:index_end]
                c, h, n, x, b, e, d_n = [outcomes[..., i] for i in range(7)]
                x = torch.clamp(x, min=1e-3)
                c = torch.clamp(c, min=1e-3)
                b = torch.clamp(b, min=1e-3)
                e = torch.clamp(e, min=1e-3)

                # b. post-decision states
                states_pd = self.state_trans_pd(states,actions,outcomes)

                # c. next-period states
                states_plus = self.state_trans(states_pd,train.quad)
                w_plus, m_plus, d_plus, q_plus, pi_plus, R_b_plus, R_e_plus = [states_plus[..., i] for i in range(7)]

                if par.Nstates_fixed > 0:
                    beta = states_plus[..., -1]  # assume beta is last
                else:
                    beta = par.beta
                    
                # d. actions next-period
                actions_plus = self.eval_policy(self.policy_NN,states_plus[:par.T-1],t0=1)
                n_act_plus, alpha_d_plus, alpha_e_plus, alpha_b_plus, alpha_h_plus = [actions_plus[..., i] for i in range(5)]

                # e. next-period outcomes
                outcomes_plus = self.outcomes(states_plus,actions_plus)
                c_plus, h_plus, n_plus, x_plus, b_plus, e_plus, d_n_plus = [outcomes_plus[..., i] for i in range(7)]
                x_plus = torch.clamp(x_plus, min=1e-3)
                c_plus = torch.clamp(c_plus, min=1e-3, max=10)
                b_plus = torch.clamp(b_plus, min=1e-3)
                e_plus = torch.clamp(e_plus, min=1e-3)
                
                # f. marginal utility current and next period
                marg_util_c = model_funcs.marg_util_c(c, par)
                marg_util_h = model_funcs.marg_util_h(h, par)
                marg_util_n = model_funcs.marg_util_n(n, par)

                marg_util_c_plus = model_funcs.marg_util_c(c_plus, par)
                marg_util_h_plus = model_funcs.marg_util_h(h_plus, par)
                marg_util_n_plus = model_funcs.marg_util_n(n_plus, par)

                expand_vars = [x, alpha_b, alpha_e, alpha_h, alpha_d, m, n, w, d_n, q, pi]
                x_exp, alpha_b_exp, alpha_e_exp, alpha_h_exp, alpha_d_exp, m_exp, n_exp, w_exp, d_n_exp, q_exp, pi_exp = [misc.expand_dim(v, R_b_plus) for v in expand_vars]
                kappa_exp = par.kappa[:par.T-1].unsqueeze(1).unsqueeze(2)
                kappa = par.kappa[:par.T-1].unsqueeze(1)
                
                # g. calculate real returns 
                q_pd_adj = q_exp*(1+pi_exp)                                   #nominel price on house previously
                R_q_plus =  (q_plus*(1+pi_plus) - q_pd_adj)/q_pd_adj  #nominel return on house

                rb_plus = (1 + R_b_plus)/(1 + pi_plus)
                re_plus = (1 + R_e_plus)/(1 + pi_plus)
                rq_plus = (1 + R_q_plus)/(1 + pi_plus)
                rd_plus = (1 + R_b_plus + par.eps_rp)/(1 + pi_plus)
                
                # h. calculate envelope term
                env_term = marg_util_c_plus*(1-alpha_b_plus)*(1-alpha_e_plus)*(1-alpha_h_plus) + marg_util_h_plus * alpha_h_plus * (1-alpha_b_plus)*(1-alpha_e_plus)
                
                # i. euler error
                ## Bond Euler Error
                term_b = rb_plus * x_exp - re_plus * alpha_e_exp * x_exp + rq_plus * x_exp * (alpha_h_exp * alpha_e_exp - alpha_h_exp)- (alpha_h_exp*alpha_e_exp - alpha_h_exp) * par.h_cost * abs(q_plus - q_exp)**(1/2)
                rhs_b = torch.sum(beta * train.quad_w[None, None, :] * term_b * env_term, dim=-1)
                lhs_b1 = marg_util_c * (alpha_e + alpha_h - alpha_h * alpha_e - 1) * x
                lhs_b2 = marg_util_h * (alpha_e * alpha_h - alpha_h) * x
                lhs_b = torch.abs(lhs_b1 + lhs_b2)
                #euler_err_bond = torch.abs(rhs_b/lhs_b -1)
                euler_err_bond = torch.abs(rhs_b - lhs_b)
                #euler_err_bond = torch.abs(lhs_b/rhs_b -1)
                #print(f"mean lhs mean: {lhs_b.mean()}")
                #print(f"mean rhs mean: {rhs_b.mean()}")
                #print(f"euler_err_bond: {euler_err_bond.mean()}")
                #print(f"lhs_b1 mean: {lhs_b1.mean()}")
                #print(f"lhs_b2 mean: {lhs_b2.mean()}")
                #print(f"marg util c mean: {marg_util_c.mean()}")
                #print(f"lhs allocation mean: {(alpha_e + alpha_h - alpha_h * alpha_e - 1).mean()}")

                ## Equity Euler Error
                term_e = re_plus * (1 - alpha_b_exp) * x_exp + rq_plus * x_exp * (alpha_h_exp * alpha_b_exp - alpha_h_exp) - (alpha_h_exp*alpha_b_exp - alpha_h_exp) * par.h_cost * abs(q_plus - q_exp)**(1/2)
                rhs_e = torch.sum(beta * train.quad_w[None, None, :] * term_e * env_term, dim=-1)
                lhs_e = marg_util_c * torch.abs(alpha_b + alpha_h - alpha_h * alpha_b - 1) * x
                lhs_e += marg_util_h * torch.abs(alpha_h*alpha_b - alpha_h) * x
                euler_err_equity = torch.abs(rhs_e - lhs_b)
                
                ## Housing Euler Error
                term_h = rq_plus * x_exp * (1 - alpha_e_exp) * (1 - alpha_b_exp) - (1 - alpha_e_exp) * (1 - alpha_b_exp) * par.h_cost * abs(q_plus - q_exp)**(1/2)
                rhs_h = torch.sum(beta * train.quad_w[None, None, :] * term_h * env_term, dim=-1)
                lhs_h = marg_util_c * torch.abs(alpha_b + alpha_e - alpha_e * alpha_b - 1) * x
                lhs_h += marg_util_h * torch.abs(1-alpha_b - alpha_e + alpha_e*alpha_b) * x
                #euler_err_housing = torch.abs(rhs_h/lhs_h -1)
                #euler_err_housing = torch.abs(torch.abs(rhs_h) - torch.abs(lhs_h))
                euler_err_housing = torch.abs(rhs_h - lhs_h)
                
                ## Labor Euler Error
                labor_term_exp = w_exp * (kappa_exp + alpha_d_exp * par.vartheta)
                labor_term     = w      * (kappa + alpha_d      * par.vartheta)
                
                inv_return = (rb_plus * alpha_b_exp
                             + re_plus * alpha_e_exp * (1 - alpha_b_exp)
                             + rq_plus * alpha_h_exp * (1 - alpha_e_exp) * (1 - alpha_b_exp))
                
                term_n1 = inv_return * labor_term_exp
                term_n2 = -(par.lbda + rd_plus -1) * alpha_d_exp * par.vartheta * w_exp
                rhs_n   = torch.sum(beta * train.quad_w[None,None,:] * (term_n1 + term_n2) * env_term, dim=-1)
                
                lhs_n = (marg_util_c * (1 - alpha_h) * (1 - alpha_e) * (1 - alpha_b)
                        +marg_util_h *  alpha_h    * (1 - alpha_e) * (1 - alpha_b)
                        ) * labor_term + marg_util_n
                
        
                euler_error_labor = torch.abs(rhs_n - lhs_n)
                
                ## Debt Euler Error
                debt_term = par.vartheta * w * n
                debt_term_exp = par.vartheta * w_exp * n_exp
                term_d1 = -(par.lbda + rd_plus -1) * debt_term_exp
                term_d2 = (rb_plus * alpha_b_exp + re_plus * alpha_e_exp * (1 - alpha_b_exp) +  rq_plus * alpha_h_exp * (1 - alpha_e_exp) * (1 - alpha_b_exp)) * debt_term_exp
                rhs_d1 = torch.sum(beta * train.quad_w[None, None, :] * term_d1 * env_term, dim=-1)
                rhs_d2 = torch.sum(beta * train.quad_w[None, None, :] * term_d2 * env_term, dim=-1)
                rhs_d = rhs_d1 + rhs_d2
                lhs_d = marg_util_c * (1 - alpha_h) * (1 - alpha_e) * (1 - alpha_b) * debt_term
                lhs_d += marg_util_h * alpha_h * (1 - alpha_e) * (1 - alpha_b) * debt_term
                
                euler_error_debt = torch.abs(rhs_d - lhs_n)
                
                # j. Batch Euler Error
                euler_error_Nbatch = torch.stack((euler_err_bond, euler_err_equity, euler_err_housing, euler_error_labor, euler_error_debt), dim =-1)
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