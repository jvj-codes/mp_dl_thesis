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
    return c**(-par.sigma_int)

def marg_util_h(h: float, par: tuple) -> float:
    return par.j * h**(-par.sigma_int)

def marg_util_n(n: float, par: tuple) -> float:
    return -n**(par.eta)

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
    total_alpha = alpha_e + alpha_b + alpha_h 
    alpha_e_norm = alpha_e /total_alpha
    alpha_b_norm = alpha_b /total_alpha
    alpha_h_norm = alpha_h /total_alpha
    
    # d. calculate outcomes 
    # compute funds before and after retirement
    if t is None: #when solving for DeepVPD or DeepFOC
        T = states.shape[0] 
        N = states.shape[1]
        kappa = par.kappa[:T].reshape(-1,1,1).repeat(1,N,train.Nquad)/(1+pi)
        
        if par.T_retired == par.T: #condition if retirement year is equal to full time period
            x = m + kappa[:par.T]*w*n_act + alpha_d*par.vartheta*w*n_act
        else:
            x_before = m[:par.T_retired] + kappa[:par.T_retired]*w[:par.T_retired]*n_act[:par.T_retired] + alpha_d[:par.T_retired]*par.vartheta*w[:par.T_retired]*n_act[:par.T_retired]
            x_after = m[par.T_retired:] + kappa[par.T_retired:] * torch.ones_like(w[par.T_retired:])
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

    ## new debt 
    if t is None:  # solving phase (DeepVPD or DeepFOC)
        T = w.shape[0]  # time dimension
        mask = (torch.arange(T, device=w.device) < par.T_retired).float().reshape(-1, 1, 1)
        d_n = alpha_d * par.vartheta * w * n_act * mask  # zero out after retirement
        
    else: #DeepSimulate
        if t < par.T_retired:
            d_n = alpha_d*par.vartheta*w*n_act
        else: 
            d_n = torch.zeros(w.shape, device=w.device)

    # bonds    
    b = alpha_b_norm * x

    # equity
    e = alpha_e_norm*(1-alpha_b_norm) * x
    
    #print(f"funds on avg {round(torch.mean(x).item())}")
    #print(f"housing price on avg. {round(torch.mean(q).item())}")
    #print(f"housing on avg. {round(torch.mean(h).item())}")
    #print(f"consumption on avg. {round(torch.mean(c).item())}")
    #print(f"allocation on avg. {round(torch.mean((1-alpha_h_norm)*(1-alpha_e_norm)*(1-alpha_b_norm)).item())}")
        
    return torch.stack((c, h_n, n_act, x, b, e, d_n),dim=-1)
        
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
    device = train.device

    
    # b. get outcomes
    c = outcomes[...,0]
    h = outcomes[...,1]
    n = outcomes[...,2]
    x = outcomes[...,3]
    b = outcomes[...,4]
    e = outcomes[...,5]
    d_n = outcomes[...,6]
    
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
    #total_alpha = alpha_e + alpha_b
    #alpha_e_norm = alpha_e/total_alpha
    #alpha_b_norm = alpha_b/total_alpha
    #alpha_h_norm = alpha_h/total_alpha
    
    # e. post-decision 
   #if t is None:  # solving phase (DeepVPD or DeepFOC)
   #    T = w.shape[0]  # time dimension
   #    mask = (torch.arange(T, device=w.device) < par.T_retired).float().reshape(-1, 1, 1)
   #    d_n = alpha_d * par.vartheta * w * n_act * mask  # zero out after retirement
   #    
   #else: #DeepSimulate
   #    if t < par.T_retired:
   #        d_n = alpha_d*par.vartheta*w*n_act
   #    else: 
   #        d_n = torch.zeros(w.shape, device=device)
   #    
   #b = alpha_b_norm * x
   #e = alpha_e_norm*(1-alpha_b_norm) * x
    
    #print(f"debt avg: {round(torch.mean(d_n).item(), 5)}")
    #if torch.mean(x).item() > 0:
    #print(f"labor supply on avg: {round(torch.mean(n_act).item(), 5)}")
    #print(f"w on avg: {round(torch.mean(w).item(), 5)}")
    #print(f"funds avaliable on avg: {round(torch.mean(x).item(), 5)}")
    #print(f"money holdings on avg: {round(torch.mean(m).item(), 5)}")
    #return torch.stack((b, e, h, w, q, pi),dim=-1)
    return torch.stack([b.to(device),e.to(device),h.to(device),w.to(device),q.to(device),d.to(device),d_n.to(device),pi.to(device),R_b.to(device)], dim=-1)

    #return torch.stack((b, e, h, w, q, d, d_n, pi, R_b),dim=-1)



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
    
    #R_plus_taylor = par.R_star + pi_plus + 0.5 * (pi_plus - par.pi_star)
    R_plus = epsn_R_plus * (par.R_star+1) * ((1+pi_plus)/(1+par.pi_star))**((par.A*(par.R_star+1))/(par.R_star)) #next period nominal interest rate adjusted by centralbank

    R_e_plus = pi_plus - R_plus + epsn_e_plus #next period equity returns
  
    q_plus = (par.q_h*(R_e_plus - R_plus) + q_pd + epsn_h_plus)/(1+pi_plus) #next period real house prices
    
    w_plus = ((par.gamma - par.theta*(R_b_pd - par.R_star)) * psi_plus * w_pd) #next period wage
	
    #print(f"wage on avg {round(torch.mean(w_plus).item())}")
    R_q_plus = (q_plus - q_pd)/q_pd 

    R_d_plus = R_plus + par.eps_rp
    
    #pi_plus = pi_plus/ 100
    #print(f"Inflation on avg: {round(torch.mean(pi_plus).item(), 5)}")
    #R_plus = R_plus/ 100
    #rint(f"Nominal interest on avg: {round(torch.mean(R_plus).item(), 5)}")
    #print(f"Taylor rule on avg: {round(torch.mean(R_plus_taylor).item(), 5)}")
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

def eval_equations_FOC(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
    
    par = model.par
    
    sq_equations = eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus)
    
    return sq_equations

def eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
    
    par = model.par
    train = model.train
    beta = discount_factor(model,states)
    
    # a. states at time t
    w = states[...,0]  # inflation at time t
    m = states[...,1]   # real money holdings at time t
    d = states[...,2]  # debt accumulated to time t
    q  = states[...,3]  # real house prices at time t
    pi  = states[...,4]  # inflation at time t
    R_b  = states[...,5]  # nominal interest rate at time t
    R_e  = states[...,6]  # return on equities at time t
    
    # b. states at time t+1
    w_plus = states_plus[...,0]  # inflation at time t
    m_plus = states_plus[...,1]   # real money holdings at time t
    d_plus = states_plus[...,2]  # debt accumulated to time t
    q_plus  = states_plus[...,3]  # real house prices at time t
    pi_plus  = states_plus[...,4]  # inflation at time t
    R_b_plus  = states_plus[...,5]  # nominal interest rate at time t
    R_e_plus  = states_plus[...,6]  # return on equities at time t

    R_q_plus = (q_plus - q)/q 
    R_d_plus = R_b_plus + par.eps_rp
        
    # c. outcomes at time t
    c = outcomes[...,0]
    h = outcomes[...,1]
    n = outcomes[...,2]
    x = outcomes[...,3]
    b = outcomes[...,4]
    e = outcomes[...,5]
    d_n = outcomes[...,6]
    
    # d. get actions
    n_act   = actions[...,0]  # labor hours share
    alpha_d = actions[...,1]  # debt share
    alpha_e = actions[...,2]  # equity share
    alpha_b = actions[...,3]  # bond share
    alpha_h = actions[...,4]  # house share

    total_alpha = alpha_e + alpha_b + alpha_h 
    alpha_e_norm = alpha_e /total_alpha
    alpha_b_norm = alpha_b /total_alpha
    alpha_h_norm = alpha_h /total_alpha
    
    # e. outcomes at time t
    c_plus = outcomes_plus[...,0]
    h_plus = outcomes_plus[...,1]
    n_plus = outcomes_plus[...,2]
    x_plus = outcomes_plus[...,3]
    b_plus = outcomes_plus[...,4]
    e_plus = outcomes_plus[...,5]
    d_n_plus = outcomes_plus[...,6]
    
    # f. multiplier at time t
    lbda_t = actions[...,6]
    mu_t = actions[...,7]
    
    # g. compute marginal utility at time t
    marg_util_c_t = marg_util_c(c)
    marg_util_h_t = marg_util_c(h)
    marg_util_n_t = marg_util_c(n)
    
    # h. compute marginal utility at time t+1
    marg_util_c_plus = marg_util_c(c_plus)
    marg_util_h_plus = marg_util_h(h_plus)
    marg_util_n_plus = marg_util_n(n_plus)
    
    # i. first order conditions
    ## 1. compute expected marginal utility at time t+1
    exp_marg_util_plus = torch.sum(train.quad_w[None,None,:]*marg_util_c_plus,dim=-1)
    
    ## 2. euler equation bond allocation
    alpha_b_marg = marg_util_c_t * (alpha_e_norm + alpha_h_norm - alpha_h_norm*alpha_e_norm-1)*x + marg_util_h_t * (alpha_e_norm * alpha_h_norm-alpha_h_norm)*x
    FOC_alpha_b = alpha_b_marg + beta[:-1]*exp_marg_util_plus*((1+R_b_plus)*x - (1+R_e_plus)*alpha_e_norm*x + (1+R_q_plus)*x*(alpha_h_norm*alpha_e_norm - alpha_h_norm))
    FOC_alpha_b_terminal = torch.zeros_like(FOC_b[-1])
    FOC_alpha_b = torch.cat((FOC_b,FOC_b_terminal[None,...]),dim=0)

    ## 3. euler equation for equity share
    alpha_e_marg = marg_util_c_t * (alpha_b_norm + alpha_e_norm - alpha_e_norm*alpha_b_norm-1)*x + marg_util_h_t * (1-alpha_b_norm-alpha_e_norm+alpha_e_norm*alpha_h_norm)*x
    FOC_alpha_e = alpha_e_marg + beta[:-1]*exp_marg_util_plus*((1+R_b_plus)*(1-alpha_b_norm)*x + (1+R_q_plus)*x*(alpha_h_norm*alpha_b_norm - alpha_h_norm))
    FOC_alpha_e_terminal = torch.zeros_like(FOC_b[-1])
    FOC_alpha_e = torch.cat((FOC_e,FOC_e_terminal[None,...]),dim=0)

    ## 4. euler equation for housing share
    alpha_h_marg = marg_util_c_t * (alpha_b_norm + alpha_h_norm - alpha_h_norm*alpha_b_norm-1)*x + marg_util_h_t * (-alpha_h_norm+alpha_h_norm*alpha_b_norm)*x
    FOC_alpha_h = alpha_h_marg + beta[:-1]*exp_marg_util_plus*((1+R_q_plus)*x*(1-alpha_e_norm)*(1-alpha_b_norm)-(1-alpha_e_norm)*(1-alpha_b_norm)*abs(q_plus - q))
    FOC_alpha_h_terminal = torch.zeros_like(FOC_h[-1])
    FOC_alpha_h = torch.cat((FOC_h,FOC_h_terminal[None,...]),dim=0)

    ## 5. euler equation for debt share
    alpha_d_marg = marg_util_c_t * (1-alpha_h_norm)*(1-alpha_e_norm)*(1-alpha_b_norm)*par.vartheta*w*n_act + marg_util_h_t * alpha_h_norm*(1-alpha_e_norm)*(1-alpha_b_norm)*par.vartheta*w*n_act
    FOC_alpha_d_debt = beta[:-1]*exp_marg_util_plus*(par.lbda + R_b_plus)*par.vartheta*w*n_act
    FOC_alpha_d_money = beta[:-1]*exp_marg_util_plus*((1+R_b_plus)*alpha_b_norm + (1+R_e_plus)*alpha_e_norm*(1-alpha_b_norm) + (1+R_q_plus)*alpha_h_norm*(1-alpha_e_norm)*(1-alpha_b_norm))*par.vartheta*w*n_act 
    FOC_alpha_d = alpha_d_marg - FOC_alpha_d_debt + FOC_alpha_d_money
    FOC_alpha_d_terminal = torch.zeros_like(FOC_d_[-1])
    FOC_alpha_d = torch.cat((FOC_d,FOC_d_terminal[None,...]),dim=0)

    ## 6. euler equation for labor share
    n_marg = marg_util_c_t * (1-alpha_h_norm)*(1-alpha_e_norm)*(1-alpha_b_norm)*(x-d_n-m)/n_act + marg_util_h_t * alpha_h_norm*(1-alpha_e_norm)*(1-alpha_b_norm)*(x-d_n-m)/n_act
    FOC_n = n_marg +  beta[:-1]*exp_marg_util_plus*((1+R_b_plus)*alpha_b_norm + (1+R_e_plus)*alpha_e_norm*(1-alpha_b_norm) + (1+R_q_plus)*alpha_h_norm*(1-alpha_e_norm)*(1-alpha_b_norm))*(x-d_n-m)/n_act
    FOC_n_terminal = torch.zeros_like(FOC_n[-1])
    FOC_n = torch.cat((FOC_n_,FOC_n_terminal[None,...]),dim=0)

    
    # j. borrowing constraint (debt constraint)
    constraint1 = par.vartheta * w * n_act - d_n
    slackness1_ = constraint1[:-1] * lbda_t[:-1]
    slackness1_terminal = constraint1[-1]
    slackness1 = torch.cat((slackness1_, slackness1_terminal[None,...]), dim=0)

    # j.2: fallback constraint (in case d_n is not used — optional, context-dependent)
    constraint2 = par.vartheta * w  # possibly for upper bound on debt
    slackness2_ = constraint2[:-1] * lbda_t[:-1]
    slackness2_terminal = constraint2[-1]
    slackness2 = torch.cat((slackness2_ , slackness2_terminal[None,...]), dim=0)

    # j.3: portfolio allocation budget constraint
    constraint3 = alpha_b_norm + alpha_e_norm + alpha_h_norm - 1.0
    slackness3_ = constraint3[:-1] * mu_t[:-1]
    slackness3_terminal = constraint3[-1]
    slackness3 = torch.cat((slackness3_, slackness3_terminal[None,...]), dim=0)

    # k. combine equations
    eq = torch.stack((
        FOC_alpha_b**2,
        FOC_alpha_e**2,
        FOC_alpha_h**2,
        FOC_alpha_d**2,
        FOC_n**2,
        slackness1**2,
        slackness2**2,
        slackness3**2
    ), dim=-1)

    
    return eq
    