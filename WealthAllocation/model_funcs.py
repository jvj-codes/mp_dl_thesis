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
    h_prev = states[...,7] # house holdings at time t, for t-1
    
    # c. shares of labor, debt and portfolio allocation from actions
    n_act   = actions[...,0]  # labor hours share
    alpha_d = actions[...,1]  # debt share
    alpha_e = actions[...,2]  # equity share
    alpha_b = actions[...,3]  # bond share
    alpha_h = actions[...,4]  # house share

    
    # d. calculate outcomes 
    # compute funds before and after retirement
    if t is None: #when solving for DeepVPD or DeepFOC
        T = states.shape[0] 
        N = states.shape[1]
        
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
            d_n = torch.cat((d_n_before, d_n_after), dim=0)
            
            n_act_before = n_act[:par.T_retired]
            n_act_after = torch.ones_like(w[par.T_retired:])*par.n_retired
            n_act = torch.cat((n_act_before, n_act_after), dim=0)
            
            x_before = m[:par.T_retired] + kappa[:par.T_retired]*w[:par.T_retired]*n_act_before + d_n_before
            x_after = m[par.T_retired:] + kappa[par.T_retired:]*w[par.T_retired:]*n_act_after
            x = torch.cat((x_before, x_after), dim=0)
            
        
    else: #when simulating
        
         if t < par.T_retired:
             d_n = alpha_d * par.vartheta * w * n_act
             x = m + par.kappa[t]*w*n_act + d_n
         else:   
             d_n = torch.zeros_like(w)
             n_act = par.n_retired*torch.ones_like(w)
             x = m + par.kappa[t]*w*n_act
      
    ## ressources
    x = torch.clamp(x, min=1e-3)
    
    x_h = alpha_h*(1-alpha_e)*(1-alpha_b)*x 
    x_h = torch.clamp(x_h, min=1e-6)
    
    h = x_h / q + h_prev
    
    ## consumption
    c = (1-alpha_h)*(1-alpha_e)*(1-alpha_b)*x
    c = torch.clamp(c, min=1e-3)

    # bonds    
    b = alpha_b * x
    b = torch.clamp(b, min=1e-3)
    
    # equity with clamping
    e = alpha_e*(1-alpha_b) * x
    e = torch.clamp(e, min=1e-3)
    
    return torch.stack((c, h, n_act, x, b, e, d_n),dim=-1)
        
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

    u = torch.clamp(u, min=-1e6, max=1e6)

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
    w = states[...,0]     # inflation at time t
    m = states[...,1]     # real money holdings at time t
    d = states[...,2]     # debt accumulated to time t
    q  = states[...,3]    # real house prices at time t
    pi  = states[...,4]   # inflation at time t
    R_b  = states[...,5]  # nominal interest rate at time t
    R_e  = states[...,6]  # return on equities at time t
    h_prev = states[...,7]
    
    # d. get actions
    n_act   = actions[...,0]  # labor hours share
    alpha_d = actions[...,1]  # debt share
    alpha_e = actions[...,2]  # equity share
    alpha_b = actions[...,3]  # bond share
    alpha_h = actions[...,4]  # house share

    states_pd = torch.stack([b,e,h,w,q,d,d_n,pi,R_b,h_prev], dim=-1)

    if par.Nstates_fixed > 0:
        fixed_states = states[..., -par.Nstates_fixed:]
        states_pd = torch.cat((states_pd, fixed_states), dim=-1)
        
    return states_pd



def state_trans(model,state_trans_pd,shocks,t=None):

    par = model.par
    train = model.train
    
    # b. unpack transition post-decision
    b_pd = state_trans_pd[...,0] 
    e_pd = state_trans_pd[...,1] 
    h_pd = state_trans_pd[...,2]
    w_pd = state_trans_pd[...,3]
    q_pd = state_trans_pd[...,4]
    d_pd = state_trans_pd[...,5]
    d_n_pd = state_trans_pd[...,6]
    pi_pd = state_trans_pd[...,7]
    R_b_pd = state_trans_pd[...,8]
    h_pd_prev = state_trans_pd[...,9]

    if par.Nstates_fixed > 0:
        beta = state_trans_pd[...,9]
        
        if t is None:
            #beta = expand_to_quad(beta, train.Nquad)    
            beta = beta.unsqueeze(-1)  
            beta = beta.unsqueeze(2).expand(-1, -1, train.Nquad, -1)  # (T, N, Nquad, 1)
            
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
    
    #q_plus = (par.q0  + par.rho_q  * q_pd - par.q_h * q_pd * (R_b_pd - par.R_star) + epsn_h_plus) / (1 + pi_plus)  # real price next period
    q_tilde = (1-par.rho_q)*par.q_bar + par.rho_q * torch.log(q_pd) - par.q_h*(R_b_pd - par.R_star) - pi_plus + epsn_h_plus
    q_plus = torch.exp(q_tilde)
    
    if t is None:
        w_plus_before = ((1 + pi_pd[:par.T_retired] + par.gamma) * w_pd[:par.T_retired]*par.rho_w - par.theta*q_pd[:par.T_retired]*(R_b_pd[:par.T_retired] - par.R_star) + psi_plus[:par.T_retired])/ (1 + pi_plus[:par.T_retired]) 
        w_plus_after = w_pd[par.T_retired:] / (1 + pi_plus[par.T_retired:])
        w_plus = torch.cat((w_plus_before, w_plus_after), dim=0)
    else:
        if t < par.T_retired:
            #w_plus = (w_pd*(1+pi_pd + par.gamma)*par.rho_w - par.theta*q_pd*(R_b_pd - par.R_star) + psi_plus) / (1 + pi_plus)
            print(w_pd.mean())
            w_tilde = (1-par.rho_w)*par.w_bar + par.rho_w*torch.log(w_pd) - par.theta * (R_b_pd - par.R_star) + pi_pd - pi_plus + psi_plus
            w_plus = torch.exp(w_tilde)
        else:
            w_tilde = torch.log(w_pd) - pi_plus
            w_plus = torch.exp(w_tilde)
            
    q_pd_adj = q_pd*(1+pi_pd)                             #nominel price on house previously
    R_q_plus =  (q_plus*(1+pi_plus) - q_pd_adj)/q_pd_adj  #nominel return on house
    
    rb_plus = (1 + R_plus)/(1 + pi_plus)
    re_plus = (1 + R_e_plus)/(1 + pi_plus)
    rq_plus = (1 + R_q_plus)/(1 + pi_plus)
    rd_plus = (1 + R_plus + par.eps_rp)/(1 + pi_plus)
    

    #m_plus = rb_plus*b_pd + re_plus*e_pd + rq_plus*h_pd - (par.lbda + rd_plus -1 )*(d_pd+d_n_pd) -par.h_cost * h_pd* torch.sqrt(abs(q_plus - q_pd))
    m_plus = rb_plus*b_pd + re_plus*e_pd + rq_plus*h_pd*q_pd - (par.lbda + rd_plus -1 )*(d_pd+d_n_pd) -par.tau * (h_pd - h_pd_prev)**2 * q_pd
    d_plus = (1-par.lbda)*(d_pd+d_n_pd)/(1+pi_plus)
    
    # e. finalize
    states_plus = torch.stack((w_plus,m_plus,d_plus,q_plus,pi_plus,R_plus,R_e_plus, h_pd),dim=-1)

    
    #print('states_plus shape:', states_plus.shape)
    #print('beta shape:', beta.shape)

    if par.Nstates_fixed > 0:
        # Ensure beta has the correct shape
        if beta.ndim == 1:  # shape (N,)
            beta = beta.unsqueeze(-1)  # → shape (N,1)
        elif beta.ndim == 3 and beta.shape[-1] != 1:  # shape (T,N,Nquad)
            beta = beta.unsqueeze(-1)  # → shape (T,N,Nquad,1)

        states_plus = torch.cat((states_plus, beta), dim=-1)

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
		#u = par.bequest * net_wealth 
		#u = par.bequest * torch.log(net_wealth)
		#u = par.bequest * torch.log(m - d_pd)

	value_pd = u.unsqueeze(-1)
	return value_pd

#%% Discount Factor
def discount_factor(model, states, t0=0, t=None):
    par = model.par
    if par.Nstates_fixed > 0:
        beta = states[..., -1]  # assume beta is last
    else:
        beta = par.beta * torch.ones_like(states[..., 0])
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
    #total_alpha = alpha_e + alpha_b + alpha_h
    alpha_e_norm = alpha_e #/ total_alpha
    alpha_b_norm = alpha_b #/ total_alpha
    alpha_h_norm = alpha_h #/ total_alpha

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
    