# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:13:54 2025

@author: JVJ
"""

import torch
import math
from EconDLSolvers import expand_to_quad, expand_to_states, exploration #, terminal_reward_pd
from misc import expand_dim

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
	return c**(1-par.sigma_int)/(1-par.sigma_int) + par.j *h**(1-par.sigma_int)/(1-par.sigma_int) - par.nu*(n**(1+par.eta))/(1+par.eta)
	#return torch.log(c) + par.j * torch.log(h) - n**par.eta/par.eta 
	#return torch.log(c) + par.j * torch.log(h) + par.vphi * ((1-n)**(1-par.nu))/(1-par.nu)

def marg_util_c(c, par) -> float:
    return c**(-par.sigma_int)

def marg_util_h(h, par) -> float:
    return par.j * h**(-par.sigma_int)

def marg_util_n(n, par) -> float:
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
    R_q  = states[...,7]  # return on house at time t
    
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
        n_retired = n_act[par.T_retired - 1] 
        
        if pi.ndim == 3:  # (T, N, Nquad) → solving mode
            kappa = par.kappa[:T].reshape(-1,1,1).repeat(1,N,train.Nquad) 
        else:  # (T, N) → simulation mode
            kappa = par.kappa[:T].reshape(-1,1).repeat(1,N)

        if par.T_retired == par.T: #condition if retirement year is equal to full time period
            x = m + kappa[:par.T]*w*n_act + alpha_d*par.vartheta*w*n_act
            d_n = alpha_d*par.vartheta*w*n_act
        else:
            d_n_before = alpha_d[:par.T_retired]*par.vartheta*w[:par.T_retired]*n_act[:par.T_retired]
            d_n_after = torch.zeros_like(w[par.T_retired:])
            n_act_before = n_act[:par.T_retired]
            n_act_after = torch.zeros_like(w[par.T_retired:])
            x_before = m[:par.T_retired] + kappa[:par.T_retired]*w[:par.T_retired]*n_act[:par.T_retired] + d_n_before
            x_after = m[par.T_retired:] + kappa[par.T_retired:] * torch.ones_like(w[par.T_retired:])*w[par.T_retired:]*n_retired + d_n_after
            x = torch.cat((x_before, x_after), dim=0)
            d_n = torch.cat((d_n_before, d_n_after), dim=0)
            n_act = torch.cat((n_act_before, n_act_after), dim=0)
    else: #when simulating
         real_kappa = par.kappa[t] # / (1+pi)
         if t == par.T_retired - 1:
             par.n_retired_fixed = (actions[:,0]).detach().clone()
             #print(f" w retired: {par.w_retired_fixed}")
         if t < par.T_retired:
             d_n = alpha_d * par.vartheta * w * n_act
             x = m + real_kappa*w*n_act + d_n
         else:   
             #real_kappa = par.kappa[t] # / (1+pi)
             d_n = torch.zeros_like(w)
             n_act = torch.zeros_like(w)
             alpha_d = torch.zeros_like(w)
             x = m + real_kappa * w * par.n_retired_fixed
    
    ## housing
    h_n = alpha_h_norm*(1-alpha_e_norm)*(1-alpha_b_norm)*x #/q
    h_n = torch.clamp(h_n, min=1e-6)
    
    ## consumption
    c = (1-alpha_h_norm)*(1-alpha_e_norm)*(1-alpha_b_norm)*x
    c = torch.clamp(c, min=1e-6)

    # bonds    
    b = alpha_b_norm * x
    b = torch.clamp(b, min=1e-6)
    
    # equity with clamping
    e = alpha_e_norm*(1-alpha_b_norm) * x
    e = torch.clamp(e, min=1e-6)
    
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
    
    # c. get state observations
    w = states[...,0]  # inflation at time t
    m = states[...,1]   # real money holdings at time t
    d = states[...,2]  # debt accumulated to time t
    q  = states[...,3]  # real house prices at time t
    pi  = states[...,4]  # inflation at time t
    R_b  = states[...,5]  # nominal interest rate at time t
    R_e  = states[...,6]  # return on equities at time t
    R_q  = states[...,7]  # return on house at time t
    
    # d. get actions
    n_act   = actions[...,0]  # labor hours share
    alpha_d = actions[...,1]  # debt share
    alpha_e = actions[...,2]  # equity share
    alpha_b = actions[...,3]  # bond share
    alpha_h = actions[...,4]  # house share
    return torch.stack([b,e,h,w,q,d,d_n,pi,R_b], dim=-1)



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
    
    R_plus = epsn_R_plus * (par.R_star+1) * ((1+pi_plus)/(1+par.pi_star))**((par.A*(par.R_star+1))/(par.R_star)) #next period nominal interest rate adjusted by centralbank

    R_e_plus = pi_plus - R_plus + epsn_e_plus #next period equity returns
  
    q_plus = (par.q_h*(R_e_plus - R_plus) + q_pd + epsn_h_plus)/(1+pi_plus) #next period real house prices

    if t is None:
        w_plus_before = ((par.gamma - par.theta*(R_b_pd[:par.T_retired] - par.R_star)) * psi_plus[:par.T_retired] * w_pd[:par.T_retired]) * torch.ones_like(w_pd[:par.T_retired]) / (1 + pi_plus[:par.T_retired]) 
        w_plus_after = w_pd[par.T_retired:] / (1 + pi_plus[par.T_retired:])
        w_plus = torch.cat((w_plus_before, w_plus_after), dim=0)
    else:
        if t < par.T_retired:
            w_plus = ((par.gamma - par.theta*(R_b_pd - par.R_star)) * psi_plus * w_pd) / (1 + pi_plus)
        else:
            w_plus = w_pd / (1+pi_plus)

    R_q_plus = (q_plus - q_pd)/q_pd 

    R_d_plus = R_plus + par.eps_rp
    

    m_plus = (1+R_plus*b_pd)/(1+pi_plus) + (1+R_e_plus)*e_pd/(1+pi_plus) + (1+R_q_plus)*h_pd - (par.lbda + R_d_plus)*(d_pd+d_n_pd)/(1+pi_plus) -(h_pd/(1+pi_plus))*abs(q_plus - q_pd)
    d_plus = (1-par.lbda)*(d_pd+d_n_pd)/(1+pi_plus)
    # e. finalize
    states_plus = torch.stack((w_plus,m_plus,d_plus,q_plus,pi_plus,R_plus,R_e_plus,R_q_plus),dim=-1)
    return states_plus


def terminal_actions(model,states):
	""" terminal actions """

	# Case I: states.shape = (1,...,Nstates)
	# Case II: states.shape = (N,Nstates)
	
	par = model.par
	train = model.train
	dtype = train.dtype
	device = train.device
	
	actions = torch.ones((*states.shape[:-1],5),dtype=dtype,device=device)

	if train.algoname == 'DeepFOC':
		multipliers = torch.zeros((*states.shape[:-1],2),dtype=dtype,device=device)
		actions = torch.cat((actions,multipliers),dim=-1)

	return actions 

def terminal_reward_pd(model, states_pd):
	""" terminal reward """

	# Case I: states_pd.shape = (1,...,Nstates_pd)
	# Case II: states_pd.shape = (N,Nstates_pd)

	train = model.train
	par = model.par
	dtype = train.dtype
	device = train.device
	#m = states[...,1]   # real money holdings at time t
	b_pd = states_pd[..., 0]  # bond holdings
	e_pd = states_pd[..., 1]  # equity holdings
	h_pd = states_pd[..., 2]  # housing holdings
	d_pd = states_pd[..., 5]  # outstanding debt

	# Compute terminal utility
	if par.bequest == 0:
		u = torch.zeros_like(b_pd)
	else:
		# avoid log(0) or negative values
		net_wealth = torch.clamp(b_pd + e_pd + h_pd - d_pd, min=1e-6)
		u = par.bequest * (net_wealth ** (1 - par.sigma_int)) / (1 - par.sigma_int)
		#u = par.bequest * torch.log(m - d_pd)

	value_pd = u.unsqueeze(-1)
	return value_pd

#%% Discount Factor
#def discount_factor(model, states, t0=0, t=None):
#    """
#    Returns discount factors for each agent.
#    - Supports fixed scalar beta: par.beta = 0.98
#    - Supports heterogeneous beta: par.beta = [0.95, 0.99]
#    """
#    par = model.par
#    device = states.device
#
#    N = states.shape[-2]  # assumes states is [T, N, ...] or [N, ...]
#
#    if isinstance(par.beta, float):
#        return torch.full((states.shape[:-1]), par.beta, device=device)
#
#    elif isinstance(par.beta, (list, tuple)) and len(par.beta) == 2:
#        beta_low, beta_high = par.beta
#
#        # If beta_vec is not already generated, cache it
#        if not hasattr(par, "beta_vec") or par.beta_vec.shape[0] != N:
#            par.beta_vec = torch.distributions.Uniform(beta_low, beta_high).sample((N,)).to(device)
#
#        if states.ndim == 3:
#            return par.beta_vec.unsqueeze(0).expand(states.shape[0], -1)  # [T, N]
#        elif states.ndim == 2:
#            return par.beta_vec  # [N]
#        else:
#            raise RuntimeError(f"Unexpected state shape {states.shape}")
#
#    else:
#        raise ValueError("par.beta must be a float or a list/tuple with two values.")

def discount_factor(model, states, t0=0, t=None):
    """ Discount factor with optional heterogeneity. """
    par = model.par

    if isinstance(par.beta, float):
        beta = par.beta * torch.ones_like(states[..., 0], device=states.device)

    elif isinstance(par.beta, (list, tuple)):
        beta_low, beta_high = par.beta
        shape = states[..., 0].shape
        dist = torch.distributions.Uniform(beta_low, beta_high)
        beta = dist.sample(shape).to(states.device)

    else:
        raise ValueError("par.beta must be either a float or a list/tuple [low, high]")

    return beta




#%% Equations for DeepFOC

def eval_equations_FOC(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
    
    par = model.par
    
    sq_equations = eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus)
    
    return sq_equations

def eval_KKT(model, states, states_plus, actions, actions_plus, outcomes, outcomes_plus):
    par = model.par
    train = model.train
    beta = discount_factor(model, states)

    # --- a. Extract current states ---
    w, m, d, q, pi, R_b, R_e, R_q = [states[..., i] for i in range(8)]

    # --- b. Extract future states ---
    w_plus, m_plus, d_plus, q_plus, pi_plus, R_b_plus, R_e_plus, R_q_plus = [states_plus[..., i] for i in range(8)]
    R_d_plus = R_b_plus + par.eps_rp

    # --- c. Extract outcomes ---
    c, h, n, x, b, e, d_n = [outcomes[..., i] for i in range(7)]
    c_plus, h_plus, n_plus, x_plus, b_plus, e_plus, d_n_plus = [outcomes_plus[..., i] for i in range(7)]

    # --- d. Extract actions ---
    n_act, alpha_d, alpha_e, alpha_b, alpha_h = [actions[..., i] for i in range(5)]
    lbda_t, mu_t = actions[..., 5], actions[..., 6]

    # Normalize portfolio shares
    total_alpha = alpha_e + alpha_b + alpha_h
    alpha_e_norm = alpha_e / total_alpha
    alpha_b_norm = alpha_b / total_alpha
    alpha_h_norm = alpha_h / total_alpha

    # --- e. Marginal utilities ---
    marg_util_c_t = marg_util_c(c, par)
    marg_util_h_t = marg_util_h(h, par)
    marg_util_n_t = marg_util_n(n, par)

    marg_util_c_plus = marg_util_c(c_plus, par)
    marg_util_h_plus = marg_util_h(h_plus, par)
    marg_util_n_plus = marg_util_n(n_plus, par)

    # --- f. Expand relevant variables for time t ---
    expand_vars = [x, alpha_b_norm, alpha_e_norm, alpha_h_norm, m, n_act, w, d_n]
    x_exp, alpha_b_norm_exp, alpha_e_norm_exp, alpha_h_norm_exp, m_exp, n_act_exp, w_exp, d_n_exp = [expand_dim(v, R_b_plus) for v in expand_vars]

    #expand_vars = [marg_util_c_t, marg_util_h_t, marg_util_n_t]
    #marg_util_c_t, marg_util_h_t, marg_util_n_t = [expand_dim(v, R_b_plus) for v in expand_vars]
    # --- g. Euler equations ---
    # Expected marginal utility of consumption
    E_muc_plus = torch.sum(train.quad_w[None, None, :] * marg_util_c_plus, dim=-1)
    
    # Bond share FOC
    term_b = (1 + R_b_plus) * x_exp - (1 + R_e_plus) * alpha_e_norm_exp * x_exp + (1 + R_q_plus) * x_exp * (alpha_h_norm_exp * alpha_e_norm_exp - alpha_h_norm_exp)
    lhs_b = marg_util_c_t * (alpha_e_norm + alpha_h_norm - alpha_h_norm * alpha_e_norm - 1) * x
    lhs_b += marg_util_h_t * (alpha_e_norm * alpha_h_norm - alpha_h_norm) * x
    FOC_b = lhs_b[:-1] +beta[:-1] * torch.sum(train.quad_w[None, None, :] * term_b * marg_util_c_plus, dim=-1)

 

    # Equity share FOC
    term_e = (1 + R_b_plus) * (1 - alpha_b_norm_exp) * x_exp + (1 + R_q_plus) * x_exp * (alpha_h_norm_exp * alpha_b_norm_exp - alpha_h_norm_exp)
    lhs_e = marg_util_c_t * (alpha_b_norm + alpha_e_norm - alpha_e_norm * alpha_b_norm - 1) * x
    lhs_e += marg_util_h_t * (1 - alpha_b_norm - alpha_e_norm + alpha_e_norm * alpha_h_norm) * x
    FOC_e = lhs_e[:-1] +beta[:-1]  * torch.sum(train.quad_w[None, None, :] * term_e * marg_util_c_plus, dim=-1)

    # Housing share FOC
    term_h = (1 + R_q_plus) * x_exp * (1 - alpha_e_norm_exp) * (1 - alpha_b_norm_exp) - (1 - alpha_e_norm_exp) * (1 - alpha_b_norm_exp) * abs(q_plus - q.unsqueeze(-1)[:-1])
    lhs_h = marg_util_c_t * (alpha_b_norm + alpha_h_norm - alpha_h_norm * alpha_b_norm - 1) * x
    lhs_h += marg_util_h_t * (-alpha_h_norm + alpha_h_norm * alpha_b_norm) * x
    FOC_h = lhs_h[:-1] +beta[:-1]  * torch.sum(train.quad_w[None, None, :] * term_h * marg_util_c_plus, dim=-1)

    # Debt share FOC
    income_term = par.vartheta * w * n_act
    lhs_d = marg_util_c_t * (1 - alpha_h_norm) * (1 - alpha_e_norm) * (1 - alpha_b_norm) * income_term
    lhs_d += marg_util_h_t * alpha_h_norm * (1 - alpha_e_norm) * (1 - alpha_b_norm) * income_term
    income_term_exp = expand_dim(income_term, R_b_plus)
    term_d1 = (par.lbda + R_b_plus) * income_term_exp
    term_d2 = ((1 + R_b_plus) * alpha_b_norm_exp + (1 + R_e_plus) * alpha_e_norm_exp * (1 - alpha_b_norm_exp) + (1 + R_q_plus) * alpha_h_norm_exp * (1 - alpha_e_norm_exp) * (1 - alpha_b_norm_exp)) * income_term_exp
    FOC_d = lhs_d[:-1] -beta[:-1]  * torch.sum(train.quad_w[None, None, :] * term_d1 * marg_util_c_plus, dim=-1)
    FOC_d += beta[:-1] * torch.sum(train.quad_w[None, None, :] * term_d2 * marg_util_c_plus, dim=-1)

    # Labor share FOC
    labor_term = (x - d_n - m) / n_act
    lhs_n = marg_util_c_t * (1 - alpha_h_norm) * (1 - alpha_e_norm) * (1 - alpha_b_norm) * labor_term
    lhs_n += marg_util_h_t * alpha_h_norm * (1 - alpha_e_norm) * (1 - alpha_b_norm) * labor_term
    labor_term_exp = expand_dim(labor_term, R_b_plus)
    gain_term = ((1 + R_b_plus) * alpha_b_norm_exp + (1 + R_e_plus) * alpha_e_norm_exp * (1 - alpha_b_norm_exp) + (1 + R_q_plus) * alpha_h_norm_exp * (1 - alpha_e_norm_exp) * (1 - alpha_b_norm_exp)) * labor_term_exp
    FOC_n = lhs_n[:-1] +beta[:-1]  * torch.sum(train.quad_w[None, None, :] * gain_term * marg_util_c_plus, dim=-1)

    # --- h. Constraints ---
    constraint1 = par.vartheta * w * n_act - d_n
    slackness1 = constraint1 * lbda_t

    constraint2 = par.vartheta * w
    slackness2 = constraint2 * lbda_t

    constraint3 = alpha_b_norm + alpha_e_norm + alpha_h_norm - 1.0
    slackness3 = constraint3 * mu_t

    # --- i. Final loss (squared KKT residuals) ---
    eq = torch.stack((
        FOC_b**2,
        FOC_e**2,
        FOC_h**2,
        FOC_d**2,
        FOC_n**2,
        slackness1[:-1]**2,
        slackness2[:-1]**2,
        slackness3[:-1]**2
    ), dim=-1)

    return eq

#def eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
#    
#    par = model.par
#    train = model.train
#    beta = discount_factor(model,states)
#    
#    # a. states at time t
#    w = states[...,0]  # inflation at time t
#    m = states[...,1]   # real money holdings at time t
#    d = states[...,2]  # debt accumulated to time t
#    q  = states[...,3]  # real house prices at time t
#    pi  = states[...,4]  # inflation at time t
#    R_b  = states[...,5]  # nominal interest rate at time t
#    R_e  = states[...,6]  # return on equities at time t
#    R_q  = states[...,7]  # return on house at time t
#    
#    # b. states at time t+1
#    w_plus = states_plus[...,0]  # inflation at time t
#    m_plus = states_plus[...,1]   # real money holdings at time t
#    d_plus = states_plus[...,2]  # debt accumulated to time t
#    q_plus  = states_plus[...,3]  # real house prices at time t
#    pi_plus  = states_plus[...,4]  # inflation at time t
#    R_b_plus  = states_plus[...,5]  # nominal interest rate at time t
#    R_e_plus  = states_plus[...,6]  # return on equities at time t
#    R_q_plus  = states_plus[...,7]  # return on house at time t
#    
#
#    R_d_plus = R_b_plus + par.eps_rp
#        
#    # c. outcomes at time t
#    c = outcomes[...,0]
#    h = outcomes[...,1]
#    n = outcomes[...,2]
#    x = outcomes[...,3]
#    b = outcomes[...,4]
#    e = outcomes[...,5]
#    d_n = outcomes[...,6]
#    
#    # d. get actions
#    n_act   = actions[...,0]  # labor hours share
#    alpha_d = actions[...,1]  # debt share
#    alpha_e = actions[...,2]  # equity share
#    alpha_b = actions[...,3]  # bond share
#    alpha_h = actions[...,4]  # house share
#
#    total_alpha = alpha_e + alpha_b + alpha_h 
#    alpha_e_norm = alpha_e /total_alpha
#    alpha_b_norm = alpha_b /total_alpha
#    alpha_h_norm = alpha_h /total_alpha
#    
#    # e. outcomes at time t
#    c_plus = outcomes_plus[...,0]
#    h_plus = outcomes_plus[...,1]
#    n_plus = outcomes_plus[...,2]
#    x_plus = outcomes_plus[...,3]
#    b_plus = outcomes_plus[...,4]
#    e_plus = outcomes_plus[...,5]
#    d_n_plus = outcomes_plus[...,6]
#    
#    # f. multiplier at time t
#    lbda_t = actions[...,5]
#    mu_t = actions[...,6]
#    
#    # g. compute marginal utility at time t
#    marg_util_c_t = marg_util_c(c,par)
#    marg_util_h_t = marg_util_c(h,par)
#    marg_util_n_t = marg_util_c(n,par)
#    
#    # h. compute marginal utility at time t+1
#    marg_util_c_plus = marg_util_c(c_plus,par)
#    marg_util_h_plus = marg_util_h(h_plus,par)
#    marg_util_n_plus = marg_util_n(n_plus,par)
#
#    #expand dimensions
#    # determine target shape from quadrature-based states
#
#    # variables to expand
#    vars_to_expand = [x[:-1], alpha_b_norm[:-1], alpha_e_norm[:-1], alpha_h_norm[:-1], m[:-1], n_act[:-1], w[:-1]]
#
#    # unpack expanded variables
#    x, alpha_b_norm, alpha_e_norm, alpha_h_norm, m, n_act, w = [
#    expand_dim(v[:-1], R_b_plus) for v in vars_to_expand]
#    
#    # i. first order conditions
#    ## 1. compute expected marginal utility at time t+1
#    exp_marg_util_plus = torch.sum(train.quad_w[None,None,:]*marg_util_c_plus,dim=-1)
#    
#    ## 2. euler equation bond allocation
#    alpha_b_marg = marg_util_c_t * (alpha_e_norm + alpha_h_norm - alpha_h_norm*alpha_e_norm-1)*x + marg_util_h_t * (alpha_e_norm * alpha_h_norm-alpha_h_norm)*x
#    FOC_alpha_b = alpha_b_marg + beta[:-1]*exp_marg_util_plus*((1+R_b_plus)*x - (1+R_e_plus)*alpha_e_norm*x + (1+R_q_plus)*x*(alpha_h_norm*alpha_e_norm - alpha_h_norm))
#    FOC_alpha_b_terminal = torch.zeros_like(FOC_b[-1])
#    FOC_alpha_b = torch.cat((FOC_b,FOC_b_terminal[None,...]),dim=0)
#
#    ## 3. euler equation for equity share
#    alpha_e_marg = marg_util_c_t * (alpha_b_norm + alpha_e_norm - alpha_e_norm*alpha_b_norm-1)*x + marg_util_h_t * (1-alpha_b_norm-alpha_e_norm+alpha_e_norm*alpha_h_norm)*x
#    FOC_alpha_e = alpha_e_marg + beta[:-1]*exp_marg_util_plus*((1+R_b_plus)*(1-alpha_b_norm)*x + (1+R_q_plus)*x*(alpha_h_norm*alpha_b_norm - alpha_h_norm))
#    FOC_alpha_e_terminal = torch.zeros_like(FOC_b[-1])
#    FOC_alpha_e = torch.cat((FOC_e,FOC_e_terminal[None,...]),dim=0)
#
#    ## 4. euler equation for housing share
#    alpha_h_marg = marg_util_c_t * (alpha_b_norm + alpha_h_norm - alpha_h_norm*alpha_b_norm-1)*x + marg_util_h_t * (-alpha_h_norm+alpha_h_norm*alpha_b_norm)*x
#    FOC_alpha_h = alpha_h_marg + beta[:-1]*exp_marg_util_plus*((1+R_q_plus)*x*(1-alpha_e_norm)*(1-alpha_b_norm)-(1-alpha_e_norm)*(1-alpha_b_norm)*abs(q_plus - q))
#    FOC_alpha_h_terminal = torch.zeros_like(FOC_h[-1])
#    FOC_alpha_h = torch.cat((FOC_h,FOC_h_terminal[None,...]),dim=0)
#
#    ## 5. euler equation for debt share
#    alpha_d_marg = marg_util_c_t * (1-alpha_h_norm)*(1-alpha_e_norm)*(1-alpha_b_norm)*par.vartheta*w*n_act + marg_util_h_t * alpha_h_norm*(1-alpha_e_norm)*(1-alpha_b_norm)*par.vartheta*w*n_act
#    FOC_alpha_d_debt = beta[:-1]*exp_marg_util_plus*(par.lbda + R_b_plus)*par.vartheta*w*n_act
#    FOC_alpha_d_money = beta[:-1]*exp_marg_util_plus*((1+R_b_plus)*alpha_b_norm + (1+R_e_plus)*alpha_e_norm*(1-alpha_b_norm) + (1+R_q_plus)*alpha_h_norm*(1-alpha_e_norm)*(1-alpha_b_norm))*par.vartheta*w*n_act 
#    FOC_alpha_d = alpha_d_marg - FOC_alpha_d_debt + FOC_alpha_d_money
#    FOC_alpha_d_terminal = torch.zeros_like(FOC_d_[-1])
#    FOC_alpha_d = torch.cat((FOC_d,FOC_d_terminal[None,...]),dim=0)
#
#    ## 6. euler equation for labor share
#    n_marg = marg_util_c_t * (1-alpha_h_norm)*(1-alpha_e_norm)*(1-alpha_b_norm)*(x-d_n-m)/n_act + marg_util_h_t * alpha_h_norm*(1-alpha_e_norm)*(1-alpha_b_norm)*(x-d_n-m)/n_act
#    FOC_n = n_marg +  beta[:-1]*exp_marg_util_plus*((1+R_b_plus)*alpha_b_norm + (1+R_e_plus)*alpha_e_norm*(1-alpha_b_norm) + (1+R_q_plus)*alpha_h_norm*(1-alpha_e_norm)*(1-alpha_b_norm))*(x-d_n-m)/n_act
#    FOC_n_terminal = torch.zeros_like(FOC_n[-1])
#    FOC_n = torch.cat((FOC_n_,FOC_n_terminal[None,...]),dim=0)
#
#    
#    # j. borrowing constraint (debt constraint)
#    constraint1 = par.vartheta * w * n_act - d_n
#    slackness1_ = constraint1[:-1] * lbda_t[:-1]
#    slackness1_terminal = constraint1[-1]
#    slackness1 = torch.cat((slackness1_, slackness1_terminal[None,...]), dim=0)
#
#    # j.2: fallback constraint (in case d_n is not used — optional, context-dependent)
#    constraint2 = par.vartheta * w  # possibly for upper bound on debt
#    slackness2_ = constraint2[:-1] * lbda_t[:-1]
#    slackness2_terminal = constraint2[-1]
#    slackness2 = torch.cat((slackness2_ , slackness2_terminal[None,...]), dim=0)
#
#    # j.3: portfolio allocation budget constraint
#    constraint3 = alpha_b_norm + alpha_e_norm + alpha_h_norm - 1.0
#    slackness3_ = constraint3[:-1] * mu_t[:-1]
#    slackness3_terminal = constraint3[-1]
#    slackness3 = torch.cat((slackness3_, slackness3_terminal[None,...]), dim=0)
#
#    # k. combine equations
#    eq = torch.stack((
#        FOC_alpha_b**2,
#        FOC_alpha_e**2,
#        FOC_alpha_h**2,
#        FOC_alpha_d**2,
#        FOC_n**2,
#        slackness1**2,
#        slackness2**2,
#        slackness3**2
#    ), dim=-1)
#
#    
#    return eq
    