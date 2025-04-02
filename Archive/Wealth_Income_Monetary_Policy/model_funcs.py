# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:13:54 2025

@author: JVJ
"""

import torch
import math
from EconDLSolvers import expand_to_quad, expand_to_states, terminal_reward_pd, discount_factor, exploration
from misc import expand_dim

#%% Utility

def utility(c, h, par):
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
	return c**(1-par.sigma_int)/(1-par.sigma_int) + par.j *h**(1-par.sigma_int)/(1-par.sigma_int) 
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
    m = states[...,0]   # real money holdings at time t
    p = states[...,1]   #permanent income
    d = states[...,2]  # debt accumulated to time t
    q  = states[...,3]  # real house prices at time t
    pi  = states[...,4]  # inflation at time t
    R_b  = states[...,5]  # nominal interest rate at time t
    R_e  = states[...,6]  # return on equities at time t
    R_q  = states[...,7]  # return on house at time t
    
    # c. shares of labor, debt and portfolio allocation from actions
    alpha_d = actions[...,0]  # debt share
    alpha_e = actions[...,1]  # equity share
    alpha_b = actions[...,2]  # bond share
    alpha_h = actions[...,3]  # house share
    
    ## normalize portfolio allocation to 1
    total_alpha = alpha_e + alpha_b + alpha_h 
    alpha_e_norm = alpha_e /total_alpha
    alpha_b_norm = alpha_b /total_alpha
    alpha_h_norm = alpha_h /total_alpha

    
    # d. calculate outcomes 
    # compute funds before and after retirement
    
    if t is None:
        T = m.shape[0]
    
        if m.ndim == 2:  # [T, N] → normal mode
            mask = (torch.arange(T, device=m.device) < par.T_retired).float().reshape(-1, 1)
        elif m.ndim == 3:  # [T, N, Nquad] → quadrature mode
            mask = (torch.arange(T, device=m.device) < par.T_retired).float().reshape(-1, 1, 1)
        else:
            raise RuntimeError("Unexpected shape for `w`")
    
        d_n = alpha_d * par.vartheta * m * mask
        
        x = m + d_n
    
    else:
        if t < par.T_retired:
            d_n = alpha_d * par.vartheta * m 
            x = m + d_n
        else:
            x = m
            d_n = torch.zeros_like(m)
    
    ## housing
    h_n = alpha_h_norm*(1-alpha_e_norm)*(1-alpha_b_norm)*x #/q
    
    ## consumption
    c = (1-alpha_h_norm)*(1-alpha_e_norm)*(1-alpha_b_norm)*x

    # bonds    
    b = alpha_b_norm * x

    # equity
    e = alpha_e_norm*(1-alpha_b_norm) * x
    
    return torch.stack((c, h_n, x, b, e, d_n),dim=-1)
        
#%% Reward

def reward(model,states,actions,outcomes,t0=0,t=None):
    par = model.par
    train = model.train
    
    # a. 
    c = outcomes[...,0]
    h = outcomes[...,1]
    
    # b. utility
    u = utility(c, h, par)
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
    x = outcomes[...,2]
    b = outcomes[...,3]
    e = outcomes[...,4]
    d_n = outcomes[...,5]
    
    # c. get state observations
    m = states[...,0]   # real money holdings at time t
    p = states[...,1]   #permanent income
    d = states[...,2]  # debt accumulated to time t
    q  = states[...,3]  # real house prices at time t
    pi  = states[...,4]  # inflation at time t
    R_b  = states[...,5]  # nominal interest rate at time t
    R_e  = states[...,6]  # return on equities at time t
    R_q  = states[...,7]  # return on house at time t
    
    p_pd = p
    b_pd = b
    e_pd = e
    h_pd = h
    q_pd = q
    d_pd = d
    d_n_pd = d_n
    pi_pd = pi
    R_b_pd = R_b
    
    states_pd = torch.stack((p_pd,b_pd,e_pd,h_pd,q_pd,d_pd,d_n_pd,pi_pd,R_b_pd),dim=-1) 
    if par.Nstates_fixed > 0: 
        states_pd = torch.cat((states_pd,states[...,7:]),dim=-1)    
    return states_pd



def state_trans(model,state_trans_pd,shocks,t=None):

    par = model.par
    train = model.train
    
    # b. unpack post decision states
    p_pd = state_trans_pd[...,0]
    b_pd = state_trans_pd[...,1]
    e_pd = state_trans_pd[...,2]
    h_pd = state_trans_pd[...,3]
    q_pd = state_trans_pd[...,4]
    d_pd = state_trans_pd[...,5]
    d_n_pd = state_trans_pd[...,6]
    pi_pd = state_trans_pd[...,7]
    R_b_pd = state_trans_pd[...,8]
    
    # c. unpack shocks
    xi  = shocks[:,0]
    psi = shocks[:,1]
    epsn_R_plus = shocks[:,2]
    epsn_pi_plus = shocks[:,3]
    epsn_e_plus = shocks[:,4]
    epsn_h_plus = shocks[:,5]
    
    if par.Nstates_fixed == 0:
        xi = torch.ones_like(p_pd)*par.sigma_xi_base
        psi = torch.ones_like(p_pd)*par.sigma_psi_base
        rho_p = torch.ones_like(p_pd)*par.rho_p_base
    elif par.Nstates_fixed == 1:
        sigma_xi = state_trans_pd[...,9]
        sigma_psi = torch.ones_like(p_pd)*par.sigma_psi_base
        rho_p = torch.ones_like(p_pd)*par.rho_p_base
    elif par.Nstates_fixed == 2:
        sigma_xi = state_trans_pd[...,9]
        sigma_psi = state_trans_pd[...,10]
        rho_p = torch.ones_like(p_pd)*par.rho_p_base
    else:
        sigma_xi = state_trans_pd[...,9]
        sigma_psi = state_trans_pd[...,10]
        rho_p = state_trans_pd[...,11]
        
    xi = torch.exp(sigma_xi*xi-0.5*sigma_xi**2)
    psi = torch.exp(sigma_psi*psi-0.5*sigma_psi**2)
    
	# b. adjust shape and scale quadrature nodes (when solving)
    if t is None:
        T,N = state_trans_pd.shape[:-1]
        
        b_pd = expand_to_quad(b_pd, train.Nquad)
        e_pd  = expand_to_quad(e_pd, train.Nquad)
        h_pd  = expand_to_quad(h_pd, train.Nquad)
        q_pd  = expand_to_quad(q_pd, train.Nquad)
        d_pd  = expand_to_quad(d_pd, train.Nquad)
        d_n_pd  = expand_to_quad(d_n_pd, train.Nquad)
        pi_pd  = expand_to_quad(pi_pd, train.Nquad)
        R_b_pd  = expand_to_quad(R_b_pd, train.Nquad)
        
        sigma_xi = expand_to_quad(sigma_xi,train.Nquad)
        sigma_psi = expand_to_quad(sigma_psi,train.Nquad)
        rho_p = expand_to_quad(rho_p,train.Nquad)
        
        xi = expand_to_states(xi,state_trans_pd)
        psi = expand_to_states(psi,state_trans_pd)

        
        #psi_plus     = expand_to_states(psi_plus,state_trans_pd)
        epsn_R_plus  = expand_to_states(epsn_R_plus,state_trans_pd)
        epsn_pi_plus = expand_to_states(epsn_pi_plus,state_trans_pd)
        epsn_e_plus  = expand_to_states(epsn_e_plus,state_trans_pd)
        epsn_h_plus  = expand_to_states(epsn_h_plus,state_trans_pd)
    


    # c. next period

    # i. persistent income
    p_plus = p_pd**rho_p*xi
    
    # ii. actual income
    if t is None: # when solving

        kappa = par.kappa.reshape(par.T,1,1).repeat(1,N,train.Nquad)

        if par.T_retired == par.T:
            y_plus = kappa[:par.T-1]*p_plus*psi
        else:
            y_plus_before = kappa[:par.T_retired]*p_plus[:par.T_retired]*psi[:par.T_retired]
            y_plus_after = kappa[par.T_retired] * torch.ones_like(p_plus[par.T_retired:])
            y_plus = torch.cat((y_plus_before,y_plus_after),dim=0)
    else: # when simulating

        if t < par.T_retired:
            y_plus = par.kappa[t]*p_plus*psi
        else:
            y_plus = par.kappa[t]
    # c. calculate inflation, interest rate, return on investment

    pi_plus = (1-par.rhopi)*par.pi_star + par.rhopi*pi_pd - par.Rpi*R_b_pd + epsn_pi_plus 
    
    #R_plus_taylor = par.R_star + pi_plus + 0.5 * (pi_plus - par.pi_star)
    R_plus = epsn_R_plus * (par.R_star+1) * ((1+pi_plus)/(1+par.pi_star))**((par.A*(par.R_star+1))/(par.R_star)) #next period nominal interest rate adjusted by centralbank

    R_e_plus = pi_plus - R_plus + epsn_e_plus #next period equity returns
  
    q_plus = (par.q_h*(R_e_plus - R_plus) + q_pd + epsn_h_plus)/(1+pi_plus) #next period real house prices

    #print(f"wage on avg {round(torch.mean(w_plus).item())}")
    R_q_plus = (q_plus - q_pd)/q_pd 

    R_d_plus = R_plus + par.eps_rp
    
    # d. calculate money holdings next period and rolling debt
    m_plus = y_plus/(1+pi_plus) +(1+R_plus*b_pd)/(1+pi_plus) + (1+R_e_plus)*e_pd/(1+pi_plus) + (1+R_q_plus)*h_pd - (par.lbda + R_d_plus)*(d_pd+d_n_pd)/(1+pi_plus) -(h_pd/(1+pi_plus))*abs(q_plus - q_pd)
    d_plus = (1-par.lbda)*(d_pd+d_n_pd)/(1+pi_plus)
    
    # iv. fixed states
    if par.Nstates_fixed == 0:
        fixed_states_tuple = ()
    elif par.Nstates_fixed == 1:
        fixed_states_tuple = (sigma_xi,)
    elif par.Nstates_fixed == 2:
        fixed_states_tuple = (sigma_xi,sigma_psi)
    else:
        fixed_states_tuple = (sigma_xi,sigma_psi,rho_p)
    # e. finalize
    states_plus = torch.stack((m_plus, p_plus, d_plus,q_plus,pi_plus,R_plus,R_e_plus,R_q_plus)+ fixed_states_tuple,dim=-1)
    return states_plus

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
    x, alpha_b_norm, alpha_e_norm, alpha_h_norm, m, n_act, w, d_n = [expand_dim(v, R_b_plus) for v in expand_vars]

    expand_vars = [marg_util_c_t, marg_util_h_t, marg_util_n_t]
    marg_util_c_t, marg_util_h_t, marg_util_n_t = [expand_dim(v, R_b_plus) for v in expand_vars]
    # --- g. Euler equations ---
    # Expected marginal utility of consumption
    E_muc_plus = torch.sum(train.quad_w[None, None, :] * marg_util_c_plus, dim=-1)

    # Bond share FOC
    term_b = (1 + R_b_plus) * x - (1 + R_e_plus) * alpha_e_norm * x + (1 + R_q_plus) * x * (alpha_h_norm * alpha_e_norm - alpha_h_norm)
    lhs_b = marg_util_c_t * (alpha_e_norm + alpha_h_norm - alpha_h_norm * alpha_e_norm - 1) * x
    lhs_b += marg_util_h_t * (alpha_e_norm * alpha_h_norm - alpha_h_norm) * x
    beta_exp = beta[:-1].view(-1, 1).expand(-1, lhs_b.shape[1])
    FOC_b = lhs_b +beta_exp * torch.sum(train.quad_w[None, None, :] * term_b * marg_util_c_plus, dim=-1)

    # Equity share FOC
    term_e = (1 + R_b_plus) * (1 - alpha_b_norm) * x + (1 + R_q_plus) * x * (alpha_h_norm * alpha_b_norm - alpha_h_norm)
    lhs_e = marg_util_c_t * (alpha_b_norm + alpha_e_norm - alpha_e_norm * alpha_b_norm - 1) * x
    lhs_e += marg_util_h_t * (1 - alpha_b_norm - alpha_e_norm + alpha_e_norm * alpha_h_norm) * x
    FOC_e = lhs_e +beta_exp  * torch.sum(train.quad_w[None, None, :] * term_e * marg_util_c_plus, dim=-1)

    # Housing share FOC
    term_h = (1 + R_q_plus) * x * (1 - alpha_e_norm) * (1 - alpha_b_norm) - (1 - alpha_e_norm) * (1 - alpha_b_norm) * abs(q_plus - q.unsqueeze(-1))
    lhs_h = marg_util_c_t * (alpha_b_norm + alpha_h_norm - alpha_h_norm * alpha_b_norm - 1) * x
    lhs_h += marg_util_h_t * (-alpha_h_norm + alpha_h_norm * alpha_b_norm) * x
    FOC_h = lhs_h +beta_exp  * torch.sum(train.quad_w[None, None, :] * term_h * marg_util_c_plus, dim=-1)

    # Debt share FOC
    income_term = par.vartheta * w * n_act
    lhs_d = marg_util_c_t * (1 - alpha_h_norm) * (1 - alpha_e_norm) * (1 - alpha_b_norm) * income_term
    lhs_d += marg_util_h_t * alpha_h_norm * (1 - alpha_e_norm) * (1 - alpha_b_norm) * income_term
    term_d1 = (par.lbda + R_b_plus) * income_term
    term_d2 = ((1 + R_b_plus) * alpha_b_norm + (1 + R_e_plus) * alpha_e_norm * (1 - alpha_b_norm) + (1 + R_q_plus) * alpha_h_norm * (1 - alpha_e_norm) * (1 - alpha_b_norm)) * income_term
    FOC_d = lhs_d -beta_exp  * torch.sum(train.quad_w[None, None, :] * term_d1 * marg_util_c_plus, dim=-1)
    FOC_d += beta * torch.sum(train.quad_w[None, None, :] * term_d2 * marg_util_c_plus, dim=-1)

    # Labor share FOC
    labor_term = (x - d_n - m) / n_act
    lhs_n = marg_util_c_t * (1 - alpha_h_norm) * (1 - alpha_e_norm) * (1 - alpha_b_norm) * labor_term
    lhs_n += marg_util_h_t * alpha_h_norm * (1 - alpha_e_norm) * (1 - alpha_b_norm) * labor_term
    gain_term = ((1 + R_b_plus) * alpha_b_norm + (1 + R_e_plus) * alpha_e_norm * (1 - alpha_b_norm) + (1 + R_q_plus) * alpha_h_norm * (1 - alpha_e_norm) * (1 - alpha_b_norm)) * labor_term
    FOC_n = lhs_n +beta_exp  * torch.sum(train.quad_w[None, None, :] * gain_term * marg_util_c_plus, dim=-1)

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
        slackness1**2,
        slackness2**2,
        slackness3**2
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
    