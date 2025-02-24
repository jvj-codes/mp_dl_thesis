# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:13:54 2025

@author: JVJ
"""

import torch
from rl_project.EconDLSolvers import auxilliary 


#%% Utility

def util(c: float, h: float , L: float , m: float , par: tuple) -> float:
	"""
    utility function 
    
    Parameters
    ----------
    c : float
        consumption.
    h : float
        the holdings of housing.
    L : float
        labor hours of work.
    m : float
        money balance divided by price level.
    par : tuple
        that consits of parameters,
        eta elasticity of intertemporal substition for hours of work
        xhi an importance parameter for the real money balance
        j an importance parameter for the holdings of housing 

    Returns
    -------
    float
        function returns the utility of a household.
        
    """

	return torch.log(c) + par.xhi * torch.log(m) + par.j * torch.log(h) - L**par.eta/par.eta 

def marg_util_c(c: float) -> float:
    return 1/c

def marg_util_m(m: float, par: tuple) -> float:
    return par.chi * 1/m

def marg_util_h(h: float, par: tuple) -> float:
    return par.j * 1/h

def marg_util_n(n: float, par: tuple) -> float:
    return -n**(par.varphi -1)

def inverse_marg_util(u: float) -> float:
    return 1/u

def util_evans_hon(c: float, m: float , n: float , par: tuple) -> float:
	"""
    utility function 
    
    Parameters
    ----------
    c : float
        consumption.
    m : float
        real money balance
    n : float
        labor hours of work.
    par : tuple
        that consits of parameters,
        sigma elasticity of intertemporal substition for consumption/real money balance
        chi elasticity of intertemporal substition for labor hours

    Returns
    -------
    float
        function returns the utility of a household.
        
    """

	return (c**(1 - par.sigma) / (1 - par.sigma)) + (par.chi * m**(1 - par.sigma) / (1 - par.sigma)) - (n**(1 + par.phi) / (1 + par.phi))

#%% Outcome
def outcomes(model, states: torch.Tensor, actions: torch.Tensor, t0: float = 0, t=None) -> torch.Tensor:
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
    
    # b. get consumption, bond saving, hours worked from actions
    c_act = actions[...,0]  # Consumption
    h_act = actions[...,1]  # Housing investment
    n_act = actions[...,2]  # Hours worked
    e_act = actions[...,3]  # Equity investment
    b_act = actions[...,4]  # Bond investment
    d_act = actions[...,5]  # New debt

    # c. get state observations
    pi = states[...,0]  # inflation at time t
    R = states[...,1]   # nominal interest rate t-1
    eq = states[...,2]  # return on equity t-1
    w  = states[...,3]  # wage at time t
    e  = states[...,4]  # equity holdings t-1
    b  = states[...,5]  # bond holdings t-1
    m  = states[...,6]  # money holdings t-1
    d  = states[...,7]  # debt t-1
    q  = states[...,8]  # house prices at time t
    h  = states[...,9]  # house holdings t-1
    
    # d. get income/wealth 
    if t is None: # when solving with DeepVPD or DeepFOC
        if len(states.shape) == 9: # when solving
            T,N = states.shape[:-1]
            y = par.y.reshape(par.T,1).repeat(1,N) # life time/wealth income
        else:
            T,N = states.shape
            y = par.y.reshape(par.T,1,1).repeat(1,N,train.Nquad)
        
        wealth = y[t0:T+t0] * n_act*w + (1+eq)*e + R*b + q*h + m 
    
    else:
        wealth = par.y[t] * n_act*w + (1+eq)*e + R*b + q*h + m 
        
    # the money holdings transition
    m = wealth - c_act - q*h_act - e_act - b_act - d*(R-1+par.rp) - par.lbda * d + d_act - (1-par.lbda) * d
    m = m/(1+pi) #real money holdings
    
    return torch.stack((c_act, m, h_act, n_act),dim=-1)
        
#%% Reward

def reward(model, outcomes: torch.Tensor, t0: int = 0 , t=None) -> float:
    par = model.par
    
    # a. 
    c = outcomes[...,0]
    m = outcomes[...,1]
    h = outcomes[...,2]
    n = outcomes[...,3]
    
    # b. utility
    u = util_evans_hon(c, m, h, n, par)
    
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
    #c = outcomes[...,0]
    m = outcomes[...,1]
    #h = outcomes[...,2]
    #n = outcomes[...,3]
    
    # c. get state observations
    pi = states[...,0]  # inflation at time t
    #R = states[...,1]   # nominal interest rate t-1
    #eq = states[...,2]  # return on equity t-1
    w  = states[...,3]  # wage at time t
    #e  = states[...,4]  # equity holdings t-1
    #b  = states[...,5]  # bond holdings t-1
    #m  = states[...,6]  # money holdings t-1
    #d  = states[...,7]  # debt t-1
    q  = states[...,8]  # house prices at time t
    #h  = states[...,9]  # house holdings t-1
    
    # d. get actions
    #c_act = actions[...,0]  # Consumption
    h_act = actions[...,1]  # Housing investment
    #n_act = actions[...,2]  # Hours worked
    e_act = actions[...,3]  # Equity investment
    b_act = actions[...,4]  # Bond investment
    d_act = actions[...,5]  # New debt

    # e. post-decision    
    return torch.stack((pi, w, e_act, b_act, m, d_act, q, h_act),dim=-1)



def state_trans(model,state_trans_pd,shocks,t=None):

    par = model.par
    train = model.train
    
    # b. unpack outcomes
    pi_pd = state_trans_pd[...,0] 
    w_pd = state_trans_pd[...,1] 
    e_pd = state_trans_pd[...,2]
    b_pd = state_trans_pd[...,3]
    m_pd = state_trans_pd[...,4]
    d_pd = state_trans_pd[...,5]
    q_pd = state_trans_pd[...,6]
    h_pd = state_trans_pd[...,7]
    
    # c. unpack shocks
    psi_plus = shocks[:,0]
    epsn_R_plus = shocks[:,1]
    epsn_pi_plus = shocks[:,2]
    epsn_Q_plus = shocks[:,3] 
    epsn_h_plus = shocks[:,4]
    
	# b. adjust shape and scale quadrature nodes (when solving)
    if t is None:
        pi_pd = auxilliary.expand_to_quad(pi_pd, train.Nquad)
        w_pd  = auxilliary.expand_to_quad(w_pd, train.Nquad)
        e_pd  = auxilliary.expand_to_quad(e_pd, train.Nquad)
        b_pd  = auxilliary.expand_to_quad(b_pd, train.Nquad)
        m_pd  = auxilliary.expand_to_quad(m_pd, train.Nquad)
        d_pd  = auxilliary.expand_to_quad(d_pd, train.Nquad)
        q_pd  = auxilliary.expand_to_quad(q_pd, train.Nquad)
        h_pd  = auxilliary.expand_to_quad(h_pd, train.Nquad)

        
        psi_plus     = auxilliary.expand_to_states(psi_plus,state_trans_pd)
        epsn_R_plus  = auxilliary.expand_to_states(epsn_R_plus,state_trans_pd)
        epsn_pi_plus = auxilliary.expand_to_states(epsn_pi_plus,state_trans_pd)
        epsn_Q_plus  = auxilliary.expand_to_states(epsn_Q_plus,state_trans_pd)
        epsn_h_plus  = auxilliary.expand_to_states(epsn_h_plus,state_trans_pd)
	
    # c. next period 
    w_plus = par.gamma * psi_plus * w_pd #next period wage
    
    pi_plus = (1-par.rhopi)*par.pi_star + par.rhopi*pi_pd + epsn_pi_plus #next period inflation
    
    R_plus = epsn_R_plus * (par.R_star -1)*(pi_plus/par.pi_star)**((par.A*par.R_star)/(par.R_star -1)) + 1 #next period nominal interest rate adjusted by centralbank
    
    eq_plus = pi_plus - R_plus + epsn_Q_plus #next period equity returns
    
    q_plus = par.q0 + par.q_h*(eq_plus - R_plus) + q_pd + epsn_h_plus #next period house prices
	
    # d. finalize
    states_plus = torch.stack((pi_plus, R_plus, eq_plus, w_plus, e_pd, b_pd, m_pd, d_pd, q_plus, h_pd),dim=-1)
    return states_plus

#%% Equations for DeepFOC

def eval_equations_FOC(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
    
    par = model.par
    
    sq_equations = eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus)
    
    return sq_equations

def eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
    
    par = model.par
    train = model.train
    beta = auxilliary.discount_factor(model,states)
    
    # a. states at time t
    pi = states[...,0]  # inflation at time t
    R = states[...,1]   # nominal interest rate t-1
    eq = states[...,2]  # return on equity t-1
    w  = states[...,3]  # wage at time t
    e  = states[...,4]  # equity holdings t-1
    b  = states[...,5]  # bond holdings t-1
    m  = states[...,6]  # money holdings t-1
    d  = states[...,7]  # debt t-1
    q  = states[...,8]  # house prices at time t
    h  = states[...,9]  # house holdings t-1
    
    # b. states at time t+1
    pi_plus = states_plus[...,0]  # inflation at time t
    R_plus = states_plus[...,1]   # nominal interest rate t-1
    eq_plus = states_plus[...,2]  # return on equity t-1
    w_plus  = states_plus[...,3]  # wage at time t
    e_plus  = states_plus[...,4]  # equity holdings t-1
    b_plus  = states_plus[...,5]  # bond holdings t-1
    m_plus  = states_plus[...,6]  # money holdings t-1
    d_plus  = states_plus[...,7]  # debt t-1
    q_plus  = states_plus[...,8]  # house prices at time t
    h_plus  = states_plus[...,9]  # house holdings t-1
    
    # c. outcomes at time t
    c = outcomes[...,0]
    m = outcomes[...,1]
    h = outcomes[...,2]
    n = outcomes[...,3]
    
    # d. get actions
    c_act = actions[...,0] # Consumption
    h_act = actions[...,1]  # Housing investment
    n_act = actions[...,2] # Hours worked
    e_act = actions[...,3]  # Equity investment
    b_act = actions[...,4]  # Bond investment
    d_act = actions[...,5]  # New debt
    
    # e. outcomes at time t
    c_plus = outcomes_plus[...,0]
    m_plus = outcomes_plus[...,1]
    h_plus = outcomes_plus[...,2]
    n_plus = outcomes_plus[...,3]
    
    # f. multiplier at time t
    mu_t = actions[...,6]
    
    # g. compute marginal utility at time t
    marg_util_c_t = marg_util_c(c)
    marg_util_m_t = marg_util_c(m)
    marg_util_h_t = marg_util_c(h)
    marg_util_n_t = marg_util_c(n)
    
    # h. compute marginal utility at time t+1
    marg_util_c_plus = marg_util_c(c_plus)
    marg_util_m_plus = marg_util_c(m_plus)
    marg_util_h_plus = marg_util_c(h_plus)
    marg_util_n_plus = marg_util_c(n_plus)
    
    # i. first order conditions
    ## 1. compute expected marginal utility at time t+1
    exp_marg_util_plus = torch.sum(train.quad_w[None,None,:]*marg_util_c_plus,dim=-1)
    
    ## 2. euler equation
    FOC_c_ = inverse_marg_util(beta[:-1]*(1+eq_plus)*exp_marg_util_plus) * c - 1
    FOC_c_terminal = torch.zeros_like(FOC_c_[-1])
    FOC_c = torch.cat((FOC_c_,FOC_c_terminal[None,...]),dim=0)
    
    ## 3. money demand
    FOC_m_ = (c_act - 1/(mu_t*par.eta))*par.chi - m
    FOC_m_terminal = torch.zeros_like(FOC_m_[-1])
    FOC_m = torch.cat((FOC_m_,FOC_m_terminal[None,...]),dim=0)
    
    ## 4. housing demand
    FOC_h_ = (c_act/q - 1/(mu_t*par.eta*q))*par.j - h_act
    FOC_h_terminal = torch.zeros_like(FOC_h_[-1])
    FOC_h = torch.cat((FOC_h_,FOC_h_terminal[None,...]),dim=0)
    
    ## 5. labor supply
    FOC_n_ = 1/(1/c_act - mu_t*par.vartheta)*n**(par.varphi-1) - w
    FOC_n_terminal = torch.zeros_like(FOC_n_[-1])
    FOC_n = torch.cat((FOC_n_,FOC_n_terminal[None,...]),dim=0)
    
    # j. borrowing constraint
    constraint = par.eta*(q*h_act + e_act + m + b_act) + par.vartheta - (1+par.eta)*d_act
    slackness_ = constraint[-1] * mu_t[:-1]
    slackness_terminal = constraint[-1]
    slackness = torch.cat((slackness_,slackness_terminal[None,...]),dim=0)
    
    # k. combine equations
    eq = torch.stack((FOC_c**2,FOC_m**2,FOC_h**2, FOC_n**2,slackness),dim=-1)
    
    return eq
    