# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:13:54 2025

@author: JVJ
"""

import torch
import math
from EconDLSolvers import expand_to_quad, expand_to_states, terminal_reward_pd, discount_factor, exploration

#%% Utility

def utility(c, h, n, par):
	"""
    utility function 
    
    Parameters
    ----------
    c : float
        consumption.
    h : float
        the holdings of housing.
    n : float
        labor hours of work.
    par : tuple
        that consits of parameters,
        eta labor schedule
        j an importance parameter for the holdings of housing 

    Returns
    -------
    float
        function returns the utility of a household.
        
    """
	return c**(1-par.sigma_int)/(1-par.sigma_int) + par.j *h**(1-par.sigma_int)/(1-par.sigma_int) - (n**(1+par.eta))/(1+par.eta)
	#return torch.log(c) + par.j * torch.log(h) - n**par.eta/par.eta 
	#return torch.log(c) + par.j * torch.log(h) + par.vphi * ((1-n)**(1-par.nu))/(1-par.nu)

def marg_util_c(c: float) -> float:
    return 1/c

def marg_util_h(h: float, par: tuple) -> float:
    return par.j * 1/h

def marg_util_n(n: float, par: tuple) -> float:
    return -n**(par.varphi -1)

def inverse_marg_util(u: float) -> float:
    return 1/u

#%% Outcome
def outcomes(model, states, actions, t0 = 0, t=None):
    """
    

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    states : Tensor
        tensor of states.
    actions : Tensor
        tensor of actions.
    t0 : float, optional
        initial time. The default is 0.
    t : TYPE, optional
        when solving for deepvpd or deepfoc set t=one. The default is None.

    Returns
    -------
    Tensor
        outcome of actions.

    """
    
    # a. unpack model parameters and model settings for training
    par = model.par
    train = model.train

    # b. get state observations
    w = states[...,0]  # inflation at time t
    m = states[...,1]   # real money holdings at time t
    d = states[...,2]  # debt accumulated to time t
    q  = states[...,3]  # real house prices at time t
    pi  = states[...,4]  # inflation at time t
    R_b  = states[...,5]  # nominal interest rate at time t
    R_e  = states[...,6]  # return on equities at time t

    
    # c. shares of labor, debt and portfolio allocation from actions
    n_act   = actions[...,0]  # labor hours share
    alpha_d = actions[...,1]  # debt share
    alpha_e = actions[...,2]  # equity share
    alpha_b = actions[...,3]  # bond share
    alpha_h = actions[...,4]  # house share
    
    ## normalize portfolio allocation to 1
    total_alpha = alpha_e + alpha_b + alpha_h + 1e-8
    alpha_e_norm = alpha_e/total_alpha
    alpha_b_norm = alpha_b/total_alpha
    alpha_h_norm = alpha_h/total_alpha
    
    # d. calculate outcomes 
    # compute funds before and after retirement
    if t is None: #when solving for DeepVPD or DeepFOC
        T,N = states.shape[-1]
        kappa = par.kappa.reshape(par.T,1,1).repeat(1,N,train.Nquad)/(1+pi)
        
        if par.T_retired == par.T: #condition if retirement year is equal to full time period
            x = m + kappa[:par.T-1]*w*n_act + alpha_d*par.vartheta*w*n_act
        else:
            x_before = m + kappa[:par.T_retired]*w*n_act + alpha_d*par.vartheta*w*n_act
            x_after = m + kappa[par.T_retired] * torch.ones_like(w[par.T_retired:])
            x = torch.cat((x_before, x_after), dim=0)
    else: #when simulating
         real_kappa = par.kappa[t] / (1+pi)
         if t < par.T_retired:
             real_kappa = real_kappa / (1+pi)
             x = m + real_kappa*w*n_act + alpha_d*par.vartheta*w*n_act
         else: 
             x = m + real_kappa
   # print(f"income at time t {income}")
    #print(f"income on avg {round(torch.mean(income).item())}")
    
    ## housing
    h_n = alpha_h_norm*(1-alpha_e_norm)*(1-alpha_b_norm)*x #/q
    
    ## consumption
    c = (1-alpha_h_norm)*(1-alpha_e_norm)*(1-alpha_b_norm)*x
    
    #print(f"funds on avg {round(torch.mean(x).item())}")
    #print(f"housing price on avg. {round(torch.mean(q).item())}")
    #print(f"housing on avg. {round(torch.mean(h).item())}")
    #print(f"consumption on avg. {round(torch.mean(c).item())}")
    #print(f"housing on avg. {round(torch.mean(h).item())}")
        
    return torch.stack((c, h_n, n_act, x),dim=-1)
        
#%% Reward

def reward(model,states,actions,outcomes,t0=0,t=None):
    par = model.par
    train = model.train
    
    # a. 
    c = outcomes[...,0]
    h = outcomes[...,1]
    n = outcomes[...,2]
    
    # b. utility
    u = utility(c, h, n, par)
    #print(f"print avg utility: {u}")
    #print(f"utility on avg: {round(torch.mean(u).item(), 5)}")
    #print(f"house holding on avg: {round(torch.mean(h).item(), 5)}")
    # c. finalize
    return u 

#%% Transition
def state_trans_pd(model,states,actions,outcomes,t0=0,t=None):
    """
    

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    states : Tensor
        tensor of states.
    actions : Tensor
        tensor of actions.
    t0 : float, optional
        initial time. The default is 0.
    t : TYPE, optional
        when solving for deepvpd or deepfoc set t=one. The default is None.

    Returns
    -------
    Tensor
        post decision.

    """
    
    # a. unpack model parameters and model settings for training
    par = model.par
    train = model.train
    
    # b. get outcomes
    c = outcomes[...,0]
    h = outcomes[...,1]
    n = outcomes[...,2]
    x = outcomes[...,3]
    
    #print(f"consumption on avg: {round(torch.mean(c).item(), 5)}")
    
    # c. get state observations
    w = states[...,0]  # inflation at time t
    m = states[...,1]   # real money holdings at time t
    d = states[...,2]  # debt accumulated to time t
    q  = states[...,3]  # real house prices at time t
    pi  = states[...,4]  # inflation at time t
    R_b  = states[...,5]  # nominal interest rate at time t
    R_e  = states[...,6]  # return on equities at time t
    
    # d. get actions
    n_act   = actions[...,0]  # labor hours share
    alpha_d = actions[...,1]  # debt share
    alpha_e = actions[...,2]  # equity share
    alpha_b = actions[...,3]  # bond share
    alpha_h = actions[...,4]  # house share
    
    ## normalize portfolio allocation to 1
    total_alpha = alpha_e + alpha_b + alpha_h + 1e-8
    alpha_e_norm = alpha_e/total_alpha
    alpha_b_norm = alpha_b/total_alpha
    alpha_h_norm = alpha_h/total_alpha
    
    # e. post-decision 
    if t < par.T_retired:
        d_n = alpha_d*par.vartheta*w*n_act
    else: 
        d_n = torch.zeros((w.shape))
        
    b = alpha_b_norm * x
    e = alpha_e_norm*(1-alpha_b_norm) * x
    
    #print(f"bonds on avg: {round(torch.mean(b).item(), 5)}")
    #if torch.mean(x).item() > 0:
    #print(f"labor supply on avg: {round(torch.mean(n_act).item(), 5)}")
    #print(f"w on avg: {round(torch.mean(w).item(), 5)}")
    #print(f"funds avaliable on avg: {round(torch.mean(x).item(), 5)}")
    #print(f"money holdings on avg: {round(torch.mean(m).item(), 5)}")
    #return torch.stack((b, e, h, w, q, pi),dim=-1)
    return torch.stack((b, e, h, w, q, d, d_n, pi, R_b),dim=-1)



def state_trans(model,state_trans_pd,shocks,t=None):

    par = model.par
    train = model.train
    
    # b. unpack outcomes
    b_pd = state_trans_pd[...,0] 
    e_pd = state_trans_pd[...,1] 
    h_pd = state_trans_pd[...,2]
    w_pd = state_trans_pd[...,3]
    q_pd = state_trans_pd[...,4]
    d_pd = state_trans_pd[...,5]
    d_n_pd = state_trans_pd[...,6]
    pi_pd = state_trans_pd[...,7]
    R_b_pd = state_trans_pd[...,8]
    
    # c. unpack shocks
    psi_plus = shocks[:,0]
    epsn_R_plus = shocks[:,1]
    epsn_pi_plus = shocks[:,2]
    epsn_e_plus = shocks[:,3]
    epsn_h_plus = shocks[:,4]
    
	# b. adjust shape and scale quadrature nodes (when solving)
    if t is None:
        b_pd = expand_to_quad(b_pd, train.Nquad)
        e_pd  = expand_to_quad(e_pd, train.Nquad)
        h_pd  = expand_to_quad(h_pd, train.Nquad)
        w_pd  = expand_to_quad(w_pd, train.Nquad)
        q_pd  = expand_to_quad(q_pd, train.Nquad)
        d_pd  = expand_to_quad(d_pd, train.Nquad)
        d_n_pd  = expand_to_quad(d_n_pd, train.Nquad)
        pi_pd  = expand_to_quad(pi_pd, train.Nquad)
        R_b_pd  = expand_to_quad(R_b_pd, train.Nquad)

        
        psi_plus     = expand_to_states(psi_plus,state_trans_pd)
        epsn_R_plus  = expand_to_states(epsn_R_plus,state_trans_pd)
        epsn_pi_plus = expand_to_states(epsn_pi_plus,state_trans_pd)
        epsn_e_plus  = expand_to_states(epsn_e_plus,state_trans_pd)
        epsn_h_plus  = expand_to_states(epsn_h_plus,state_trans_pd)
	
    # c. calculate inflation, interest rate, return on investment

    pi_plus = (1-par.rhopi)*par.pi_star + par.rhopi*pi_pd - par.Rpi*R_b_pd + epsn_pi_plus 

    R_plus = epsn_R_plus * ((1+par.R_star)*((1+pi_plus)/(1+par.pi_star))**((par.A*(par.R_star+1))/(par.R_star-1)) +1) #next period nominal interest rate adjusted by centralbank

    R_e_plus = pi_plus - R_plus + epsn_e_plus #next period equity returns
  
    q_plus = (par.q_h*(R_e_plus - R_plus) + q_pd + epsn_h_plus)/(1+pi_plus) #next period real house prices
    
    w_plus = (par.gamma * psi_plus * w_pd) #next period wage
	
    R_q_plus = (q_plus - q_pd)/q_pd 

    R_d_plus = R_plus + par.eps_rp
    
    #pi_plus = pi_plus/ 100
    #print(f"Inflation on avg: {round(torch.mean(pi_plus).item(), 5)}")
    #R_plus = R_plus/ 100
    #print(f"Nominal interest on avg: {round(torch.mean(R_plus).item(), 5)}")
    #R_e_plus = R_e_plus / 100 #percentage
    #print(f"Stock retun on avg: {round(torch.mean(R_e_plus).item(), 5)}")
    #R_q_plus =  R_q_plus / 100 #percentage
    #print(f"House return on avg: {round(torch.mean(R_q_plus).item(), 5)}")
    
    # d. calculate money holdings next period and rolling debt
    #m_plus = (1+R_plus)* b_pd/(1+pi_plus) + (1+R_e_plus)*e_pd/(1+pi_plus) + (1+R_q_plus)*h_pd/(1+pi_plus)
    m_plus = (1+R_plus*b_pd)/(1+pi_plus) + (1+R_e_plus)*e_pd/(1+pi_plus) + (1+R_q_plus)*h_pd - (par.lbda + R_d_plus)*(d_pd+d_n_pd)/(1+pi_plus) -(h_pd/(1+pi_plus))*abs(q_plus - q_pd)
    #d_plus = d_n_pd/(1+pi_plus) + (1-par.lbda)*d_pd/(1+pi_plus)    
    d_plus = (1-par.lbda)*(d_pd+d_n_pd)/(1+pi_plus)
    #print(f"Money t+1 on avg: {round(torch.mean(m_plus).item(), 5)}")
    # e. finalize
    #states_plus = torch.stack((w_plus,m_plus,q_plus,pi_plus,R_plus,R_e_plus),dim=-1)
    states_plus = torch.stack((w_plus,m_plus,d_plus,q_plus,pi_plus,R_plus,R_e_plus),dim=-1)
    return states_plus

#%% Equations for DeepFOC

# def eval_equations_FOC(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
    
#     par = model.par
    
#     sq_equations = eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus)
    
#     return sq_equations

# def eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
    
#     par = model.par
#     train = model.train
#     beta = auxilliary.discount_factor(model,states)
    
#     # a. states at time t
#     pi = states[...,0]  # inflation at time t
#     R = states[...,1]   # nominal interest rate t-1
#     eq = states[...,2]  # return on equity t-1
#     w  = states[...,3]  # wage at time t
#     e  = states[...,4]  # equity holdings t-1
#     b  = states[...,5]  # bond holdings t-1
#     m  = states[...,6]  # money holdings t-1
#     d  = states[...,7]  # debt t-1
#     q  = states[...,8]  # house prices at time t
#     h  = states[...,9]  # house holdings t-1
    
#     # b. states at time t+1
#     pi_plus = states_plus[...,0]  # inflation at time t
#     R_plus = states_plus[...,1]   # nominal interest rate t-1
#     eq_plus = states_plus[...,2]  # return on equity t-1
#     w_plus  = states_plus[...,3]  # wage at time t
#     e_plus  = states_plus[...,4]  # equity holdings t-1
#     b_plus  = states_plus[...,5]  # bond holdings t-1
#     m_plus  = states_plus[...,6]  # money holdings t-1
#     d_plus  = states_plus[...,7]  # debt t-1
#     q_plus  = states_plus[...,8]  # house prices at time t
#     h_plus  = states_plus[...,9]  # house holdings t-1
    
#     # c. outcomes at time t
#     c = outcomes[...,0]
#     m = outcomes[...,1]
#     h = outcomes[...,2]
#     n = outcomes[...,3]
    
#     # d. get actions
#     c_act = actions[...,0] # Consumption
#     h_act = actions[...,1]  # Housing investment
#     n_act = actions[...,2] # Hours worked
#     e_act = actions[...,3]  # Equity investment
#     b_act = actions[...,4]  # Bond investment
#     d_act = actions[...,5]  # New debt
    
#     # e. outcomes at time t
#     c_plus = outcomes_plus[...,0]
#     m_plus = outcomes_plus[...,1]
#     h_plus = outcomes_plus[...,2]
#     n_plus = outcomes_plus[...,3]
    
#     # f. multiplier at time t
#     mu_t = actions[...,6]
    
#     # g. compute marginal utility at time t
#     marg_util_c_t = marg_util_c(c)
#     marg_util_m_t = marg_util_c(m)
#     marg_util_h_t = marg_util_c(h)
#     marg_util_n_t = marg_util_c(n)
    
#     # h. compute marginal utility at time t+1
#     marg_util_c_plus = marg_util_c(c_plus)
#     marg_util_m_plus = marg_util_c(m_plus)
#     marg_util_h_plus = marg_util_c(h_plus)
#     marg_util_n_plus = marg_util_c(n_plus)
    
#     # i. first order conditions
#     ## 1. compute expected marginal utility at time t+1
#     exp_marg_util_plus = torch.sum(train.quad_w[None,None,:]*marg_util_c_plus,dim=-1)
    
#     ## 2. euler equation
#     FOC_c_ = inverse_marg_util(beta[:-1]*(1+eq_plus)*exp_marg_util_plus) * c - 1
#     FOC_c_terminal = torch.zeros_like(FOC_c_[-1])
#     FOC_c = torch.cat((FOC_c_,FOC_c_terminal[None,...]),dim=0)
    
#     ## 3. money demand
#     FOC_m_ = (c_act - 1/(mu_t*par.eta))*par.chi - m
#     FOC_m_terminal = torch.zeros_like(FOC_m_[-1])
#     FOC_m = torch.cat((FOC_m_,FOC_m_terminal[None,...]),dim=0)
    
#     ## 4. housing demand
#     FOC_h_ = (c_act/q - 1/(mu_t*par.eta*q))*par.j - h_act
#     FOC_h_terminal = torch.zeros_like(FOC_h_[-1])
#     FOC_h = torch.cat((FOC_h_,FOC_h_terminal[None,...]),dim=0)
    
#     ## 5. labor supply
#     FOC_n_ = 1/(1/c_act - mu_t*par.vartheta)*n**(par.varphi-1) - w
#     FOC_n_terminal = torch.zeros_like(FOC_n_[-1])
#     FOC_n = torch.cat((FOC_n_,FOC_n_terminal[None,...]),dim=0)
    
#     # j. borrowing constraint
#     constraint = par.eta*(q*h_act + e_act + m + b_act) + par.vartheta - (1+par.eta)*d_act
#     slackness_ = constraint[-1] * mu_t[:-1]
#     slackness_terminal = constraint[-1]
#     slackness = torch.cat((slackness_,slackness_terminal[None,...]),dim=0)
    
#     # k. combine equations
#     eq = torch.stack((FOC_c**2,FOC_m**2,FOC_h**2, FOC_n**2,slackness),dim=-1)
    
    # return eq
    