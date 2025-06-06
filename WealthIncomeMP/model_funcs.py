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

def utility(c, n, par):
	"""
    utility function 
    
    Parameters
    ----------
    c : float
        consumption.
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
	return c**(1-par.sigma_int)/(1-par.sigma_int) - par.nu*(n**(1+par.eta))/(1+par.eta)

def marg_util_c(c, par) -> float:
    return c**(-par.sigma_int)

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
    a = states[...,0]  # illiquid assets carried over from t-1
    w = states[...,1]  # real wage at time t
    m = states[...,2]   # real money holdings at time t
    pi  = states[...,3]  # inflation at time t
    R_b  = states[...,4]  # nominal interest rate at time t
    R_e  = states[...,5]  # return on illiquid assets at time t
    d = states[...,6]  # debt carried over from t-1
    
    # c. shares of labor, debt and portfolio allocation from actions
    n_act   = actions[...,0]  # labor hours share
    alpha_e = actions[...,1]  # equity share
    alpha_b = actions[...,2]  # bond share
    alpha_d = actions[...,3]  # bond share

    
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
            
            s = m + kappa[:par.T]*w*n_act + par.phi*a*alpha_d
            
        else:
            n_act_before = n_act[:par.T_retired]
            n_act_after = torch.ones_like(w[par.T_retired:])*par.n_retired
            n_act = torch.cat((n_act_before, n_act_after), dim=0)
            
            d_before = par.phi*a[:par.T_retired]*alpha_d[:par.T_retired]
            d_after = torch.zeros_like(w[par.T_retired:])
            d_n = torch.cat((d_before, d_after), dim=0)
            
            s_before = m[:par.T_retired] + kappa[:par.T_retired]*w[:par.T_retired]*n_act_before + d_before
            s_after = m[par.T_retired:] + kappa[par.T_retired:]*w[par.T_retired:]*n_act_after + d_after
            s = torch.cat((s_before, s_after), dim=0)
            
        
    else: #when simulating
        
         if t < par.T_retired:
             d_n = alpha_d * a * par.phi
             s = m + par.kappa[t]*w*n_act + d_n
         else:
             d_n = torch.zeros_like(w)
             n_act = par.n_retired*torch.ones_like(w)
             s = m + par.kappa[t]*w*n_act
      
    ## ressources
    #s = torch.clamp(s, min=1e-3)
    
    ## consumption
    c = (1-alpha_e)*(1-alpha_b)*s
    #c = torch.clamp(c, min=1e-3)

    # bonds    
    b = alpha_b * s
    #b = torch.clamp(b, min=1e-3)
    
    # illiquid assets with clamping
    a_bar = alpha_e*(1-alpha_b) * s
    #a_bar = torch.clamp(a_bar, min=1e-3)
    
    a_t = a_bar + a
        
    return torch.stack((c, n_act, s, b, a_t,d_n),dim=-1)
        
#%% Reward

def reward(model,states,actions,outcomes,t0=0,t=None):
    par = model.par
    train = model.train
    
    # a.
    c = outcomes[...,0]
    n = outcomes[...,1]
    
    # b. utility
    u = utility(c, n, par)

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
    c_pd = outcomes[...,0]
    n_pd = outcomes[...,1]
    s_pd = outcomes[...,2]
    b_pd = outcomes[...,3]
    a_pd = outcomes[...,4]
    d_pd = outcomes[...,5]
    
    # c. get state observations
    a_prev = states[...,0]  # illiquid assets carried over from t-1
    w_pd = states[...,1]  # real wage at time t
    m_pd = states[...,2]   # real money holdings at time t
    pi_pd  = states[...,3]  # inflation at time t
    R_b_pd  = states[...,4]  # nominal interest rate at time t
    R_e_pd  = states[...,5]  # return on illiquid assets at time t
    d_prev = states[...,6]  # debt carried over from t-1
    
    # d. get actions
    n_act   = actions[...,0]  # labor hours share
    alpha_e = actions[...,1]  # equity share
    alpha_b = actions[...,2]  # bond share
    alpha_d = actions[...,3]

    states_pd = torch.stack([a_pd,  a_prev, b_pd, w_pd, pi_pd, R_b_pd, R_e_pd,d_prev,d_pd], dim=-1)

    if par.Nstates_fixed > 0:
        fixed_states = states[..., -par.Nstates_fixed:]
        states_pd = torch.cat((states_pd, fixed_states), dim=-1)
        
    return states_pd



def state_trans(model,state_trans_pd,shocks,t=None):

    par = model.par
    train = model.train
    
    # b. unpack transition post-decision
    a_pd = state_trans_pd[...,0] 
    a_prev = state_trans_pd[...,1] 
    b_pd = state_trans_pd[...,2] 
    w_pd = state_trans_pd[...,3]
    pi_pd = state_trans_pd[...,4]
    R_b_pd = state_trans_pd[...,5]
    R_e_pd = state_trans_pd[...,6]
    d_prev = state_trans_pd[...,7]
    d_pd = state_trans_pd[...,8]

    if par.Nstates_fixed > 0:
        beta = state_trans_pd[..., -par.Nstates_fixed:]
        
        if t is None:
            #beta = expand_to_quad(beta, train.Nquad)    
            #beta = beta.unsqueeze(-1)  
            beta = beta.unsqueeze(2).expand(-1, -1, train.Nquad, -1)  # (T, N, Nquad, 1)
            
    # c. unpack shocks
    psi_plus = shocks[:,0]
    epsn_R_plus = shocks[:,1]
    epsn_pi_plus = shocks[:,2]
    epsn_e_plus = shocks[:,3]
    
	# b. adjust shape and scale quadrature nodes (when solving)
    if t is None:
        a_pd = expand_to_quad(a_pd, train.Nquad)
        b_pd  = expand_to_quad(b_pd, train.Nquad)
        w_pd  = expand_to_quad(w_pd, train.Nquad)
        pi_pd  = expand_to_quad(pi_pd, train.Nquad)
        R_b_pd  = expand_to_quad(R_b_pd, train.Nquad)
        R_e_pd  = expand_to_quad(R_e_pd, train.Nquad)
        d_prev = expand_to_quad(d_prev, train.Nquad)
        d_pd = expand_to_quad(d_pd, train.Nquad)
        a_prev = expand_to_quad(a_prev, train.Nquad)
        
        psi_plus     = expand_to_states(psi_plus,state_trans_pd)
        epsn_R_plus  = expand_to_states(epsn_R_plus,state_trans_pd)
        epsn_pi_plus = expand_to_states(epsn_pi_plus,state_trans_pd)
        epsn_e_plus  = expand_to_states(epsn_e_plus,state_trans_pd)
	
    # c. calculate inflation, interest rate, return on investment
    
    # 1. Inflation, Philips-Curve
    pi_plus = ( par.pi_bar
               + par.rho_pi * (pi_pd - par.pi_bar)
               - par.phi_R * (R_b_pd - par.R_bar)
               + epsn_pi_plus)
    
    # 2. Nominal policy rate, Taylor rule
    log_R_plus = ( torch.log(torch.tensor(par.R_bar))
                  + par.rho_R * (torch.log(R_b_pd) - torch.log(torch.tensor(par.R_bar)))
                  + (1 - par.rho_R) * par.phi_pi * torch.log(pi_plus)
                  + epsn_R_plus)
    
    R_plus = torch.exp(log_R_plus)
    
    # 3. Nominal Illiquid Return
    log_Ra_plus = ( torch.log(torch.tensor(par.Ra_bar))
                   + par.rho_a * (torch.log(R_e_pd) - torch.log(torch.tensor(par.Ra_bar)))
                   - par.phi_Ra * (torch.log(R_b_pd) - torch.log(torch.tensor(par.R_bar)))
                   + epsn_e_plus )
    
    Ra_plus = torch.exp(log_Ra_plus)
    
    # 4. Wages
    if t is None:
        w_tilde_before = (1-par.rho_w)*torch.log(torch.tensor(par.w_bar)) + par.rho_w*torch.log(w_pd[:par.T_retired]) - par.theta * (torch.log(R_b_pd[:par.T_retired]) - torch.log(torch.tensor(par.R_bar))) + torch.log(pi_pd[:par.T_retired]) + psi_plus[:par.T_retired]
        w_tilde_after = torch.log(w_pd[par.T_retired:])
        
        w_plus_before = torch.exp(w_tilde_before)
        w_plus_after = torch.exp(w_tilde_after)
        w_plus = torch.cat((w_plus_before, w_plus_after), dim=0)
    else:
        if t < par.T_retired:
            w_tilde = (1-par.rho_w)* torch.log(torch.tensor(par.w_bar)) + par.rho_w*torch.log(w_pd) - par.theta * (torch.log(R_b_pd) - torch.log(torch.tensor(par.R_bar))) + torch.log(pi_pd) + psi_plus
            w_plus = torch.exp(w_tilde)
        else:
            w_tilde = torch.log(w_pd) 
            w_plus = torch.exp(w_tilde)
    
    # 5. Debt
    d_plus = (1-par.lbda)*(d_prev + d_pd)

    # 6. Illiquid Assets
    a_plus = a_pd 
    
    # 7. Cash on Hand
    m_plus =  (R_b_pd / pi_plus) * b_pd + (Ra_plus/pi_plus -1) * a_pd - par.gamma * (a_pd - a_prev)**2 - (par.lbda+par.eps_rp + (R_b_pd/pi_plus -1))*(d_prev+d_pd)
    
    # e. finalize
    states_plus = torch.stack((a_plus,w_plus,m_plus,pi_plus, R_plus, Ra_plus, d_plus) ,dim=-1)

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
	actions = torch.ones((*states.shape[:-1],4),dtype=dtype,device=device)

	if train.algoname == 'DeepFOC':
		multipliers = torch.zeros((*states.shape[:-1],1),dtype=dtype,device=device)
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

	a_pd = states_pd[...,0]
	a_prev = states_pd[...,1]
	b_pd = states_pd[...,2]
	w_pd = states_pd[...,3]
	pi_pd = states_pd[...,4]
	R_b_pd = states_pd[...,5]
	R_e_pd = states_pd[...,6]
	d_prev = states_pd[...,7]
	d_pd = states_pd[...,8]

	# Compute terminal utility
	if par.bequest == 0:
		u = torch.zeros_like(b_pd)
	else:
		net_wealth = torch.clamp(a_pd - d_prev, min=1e-6)
		u = par.bequest * torch.sqrt(net_wealth)
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
    a_prev, w, m, pi, R_b, R_e, d_prev = [states[..., i] for i in range(7)]


    # --- b. Extract future states ---
    a, w_plus, m_plus, pi_plus, R_b_plus, R_e_plus, d = [states_plus[..., i] for i in range(7)]

    # --- c. Extract outcomes ---
    c, n, s, b, a_pd, d_pd  = [outcomes[..., i] for i in range(6)]
    
    # --- d. Extract future outcomes ---
    c_plus, n_plus, s_plus, b_plus, a_pd_plus, d_pd_plus = [outcomes_plus[..., i] for i in range(6)]

    # --- e. Extract actions ---
    n_act, alpha_e, alpha_b, alpha_d, lambda_t = [actions[..., i] for i in range(5)]
    
    # --- f. Clamping ---
    s = torch.clamp(s, min=1e-3)
    c = torch.clamp(c, min=1e-3)
    b = torch.clamp(b, min=1e-3)
    a_pd = torch.clamp(a_pd, min=1e-3)
    
    s_plus = torch.clamp(s_plus, min=1e-3)
    c_plus = torch.clamp(c_plus, min=1e-3)
    b_plus = torch.clamp(b_plus, min=1e-3)
    a_pd_plus = torch.clamp(a_pd_plus, min=1e-3)
    

    # --- e. Marginal utilities ---
    # f. marginal utility current and next period
    marg_util_c_t = marg_util_c(c, par)
    marg_util_n_t = marg_util_n(n, par)

    marg_util_c_plus = marg_util_c(c_plus, par)

    # --- f. Expand relevant variables for time t ---
    kappa = (s - m) / (n_act * w )
    expand_vars = [a_pd, a_prev, R_b, w, alpha_e, alpha_b]
    a_pd_exp, a_prev_exp, R_b_exp, alpha_e_exp, alpha_b_exp, kappa_exp = [expand_dim(v, R_b_plus) for v in expand_vars]
    
    # --- g. Derive FOCS ---
    #  Illiquid alpha_t^e share FOC
    term_e = (R_e_plus/pi_plus-1) - 2*par.gamma*(a_pd_exp - a_prev_exp) + 2*par.gamma*(a_pd_plus - a)
    rhs_e = torch.sum(train.quad_w[None, None, :]*term_e* marg_util_c_plus, dim=-1)
    lhs_e = marg_util_c_t
    FOC_ilq_ = beta[:-1] * rhs_e - lhs_e[:-1]
    FOC_ilq_terminal = lhs_e[-1]
    FOC_ilq = torch.cat((FOC_ilq_,FOC_ilq_terminal[None,...]),dim=0)

    # Liquid alpha_t^b share FOC
    term_b = R_b_plus / pi_plus - (R_e_plus/pi_plus -1)*alpha_e_exp + alpha_e_exp * 2*par.gamma*(a_pd_exp - a_prev_exp) + 2*par.gamma*(a_pd_plus - a)
    rhs_b = torch.sum(train.quad_w[None, None, :]*term_b* marg_util_c_plus, dim=-1)
    lhs_b = (1-alpha_e) * marg_util_c_t
    FOC_bond_ = beta[:-1] * rhs_b - lhs_b[:-1]
    FOC_bond_terminal = lhs_b[-1]
    FOC_bond = torch.cat((FOC_bond_,FOC_bond_terminal[None,...]),dim=0)
    
    # Labor supply n_t share FOC
    term_n = alpha_b_exp * (R_b_plus) / pi_plus + (1-alpha_b_exp)*alpha_e_exp*((R_e_plus/pi_plus-1) - 2*par.gamma*(a_pd_exp - a_prev_exp) + 2*par.gamma*(a_pd_plus - a))
    rhs_n = torch.sum(train.quad_w[None, None, :]*term_n* marg_util_c_plus, dim=-1)
    lhs_n = marg_util_n_t/w*kappa + marg_util_c_t * (1-alpha_e)*(1-alpha_b)
    FOC_labor_ =  beta[:-1] * rhs_n - lhs_n[:-1]
    FOC_labor_terminal = lhs_n[-1]
    FOC_labor = torch.cat((FOC_labor_,FOC_labor_terminal[None,...]),dim=0)
    
    # Debt allocation alpha_t^d
    term_d = alpha_b_exp * (R_b_plus) / pi_plus + (1-alpha_b_exp)*alpha_e_exp*(R_e_plus/pi_plus-1) - (par.lbda + par.eps_rp+ (R_b_plus/pi_plus  -1))
    rhs_d = torch.sum(train.quad_w[None, None, :]*term_d*marg_util_c_plus, dim=-1)
    lhs_d = marg_util_c_t * (1-alpha_e)*(1-alpha_b)
    FOC_debt_ = beta[:-1] * rhs_d - lhs_d[:-1]
    FOC_debt_terminal = lhs_d[-1]
    FOC_debt = torch.cat((FOC_debt_,FOC_debt_terminal[None,...]),dim=0)
    
    # --- h. Constraints ---
    constraint_1 = par.phi*a_prev - d_pd
    slackness_1_ = constraint_1[:-1] * lambda_t[:-1]
    slackness_terminal_1 = constraint_1[-1]
    slackness_1 = torch.cat((slackness_1_,slackness_terminal_1[None,...]),dim=0)

    # --- i. Final loss (squared KKT residuals) ---
    eq = torch.stack((
        FOC_ilq**2,
        FOC_bond**2,
        FOC_labor**2,
        FOC_debt**2,
        slackness_1**2
    ), dim=-1)

    return eq
    