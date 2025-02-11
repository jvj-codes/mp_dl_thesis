# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:13:54 2025

@author: JVJ
"""

import numpy as np
import torch

# Now import the required functions
from rl_project.EconDLSolvers import auxilliary 


#%% Utility

def util_iacoviello(c: float, h: float , L: float , m: float , par: tuple) -> float:
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

	return torch.log(c) + par.j * torch.log(h) - L**par.eta/par.eta + par.xhi * torch.log(m)

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
        post decision .

    """
    
    # a. unpack model parameters and model settings for training
    par = model.par
    train = model.train
    
    # b. get consumption, bond saving, hours worked from actions
    c_act = actions[...,0]
    b_act = actions[...,1]
    n_act = actions[...,2]
    
    # c. get state observations
    # m_prev = states[...,0] #real money balance t-1
    # b_prev = states[...,1] #real bond holdings t-1
    # p_prev = states[...,2] #inflation t-1
    # c_prev = states[...,3] #consumption t-1
    # n_prev = states[...,4] #hours worked t-1
    # epsn_t = states[...,5] #exogenous fiscal policy shock
    # epsn_R = states[...,6] #exogenous monetary policy shock
    # epsn_R_prev = states[...,7] #exogenous monetary policy shock prev
    # epsn_y = states[...,8] #exogenous technology shock
    
#     if t is None: # when solving with DeepVPD or DeepFOC
#         if len(states.shape) == 9: # when solving
#             T,N = states.shape[:-1]
#             y = par.y.reshape(par.T,1).repeat(1,N)

#        	else:
# 		 	T = states.shape[0]
# 		 	N = states.shape[1]
# 		 	y = par.y.reshape(par.T,1,1).repeat(1,N,train.Nquad)

        #actions lead to changes in the envoirment
    
    #wages and production set
    # y = epsn_y*n_act**(1-par.eta) #equation 8
    
    #p, c, b form actions and markets clear
    # p = c_act/y #equation 21, markets clear and is set
    # c = c_act/p #equation 22
    # b = b_act/p #equation 23
    
    # #policy realisations equation 16 (taylor rule) and equation 15 goverment raises taxes
    # R = epsn_R * (par.R_star -1) * (p / par.p_star) ** (par.A*par.R_star/ (par.R_star-1))
    # R_prev = epsn_R_prev * (par.R_star -1) * (p_prev / par.p_star) ** (par.A*par.R_star/ (par.R_star-1))
    # tau = par.gamma0 + par.gamma * b_prev + epsn_t
    
    # m = m_prev/p +  R_prev + b_prev/p - b - tau
    # n = y**(1/(1-par.eta)) * 1/(epsn_y)

    return torch.stack((c_act, b_act, n_act),dim=-1)
        
#%% Reward

def reward(model, outcomes: torch.Tensor, t0: int = 0 , t=None) -> float:
    par = model.par
    
    # a. consumption
    c = outcomes[...,3]
    m = outcomes[...,0]
    n = outcomes[...,4]
    
    # b. utility
    u = util_evans_hon(c, m, n, par)
    
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
        post decision .

    """
    
    # a. unpack model parameters and model settings for training
    par = model.par
    train = model.train
    
    # b. get consumption, bond saving, hours worked from actions
    c_act = actions[...,0]
    b_act = actions[...,1]
    n_act = actions[...,2]
    
    # c. get state observations
    m_prev = states[...,0] #real money balance t-1
    b_prev = states[...,1] #real bond holdings t-1
    p_prev = states[...,2] #inflation t-1
    c_prev = states[...,3] #consumption t-1
    n_prev = states[...,4] #hours worked t-1
    epsn_t = states[...,5] #exogenous fiscal policy shock
    epsn_R = states[...,6] #exogenous monetary policy shock
    #epsn_R_prev = states[...,7] #exogenous monetary policy shock prev
    epsn_y = states[...,7] #exogenous technology shock
    
#     if t is None: # when solving with DeepVPD or DeepFOC
#         if len(states.shape) == 9: # when solving
#             T,N = states.shape[:-1]
#             y = par.y.reshape(par.T,1).repeat(1,N)

#       	else:
# 		 	T = states.shape[0]
# 		 	N = states.shape[1]
# 		 	y = par.y.reshape(par.T,1,1).repeat(1,N,train.Nquad)

        #actions lead to changes in the envoirment
    
    #wages and production set
    y = epsn_y*n_act**(1-par.eta) #equation 8
    
    #p, c, b form actions and markets clear
    p = c_act/y #equation 21, markets clear and is set
    c = c_act/p #equation 22
    b = b_act/p #equation 23
    
    #policy realisations equation 16 (taylor rule) and equation 15 goverment raises taxes
    R = epsn_R * (par.R_star -1) * (p / par.p_star) ** (par.A*par.R_star/ (par.R_star-1))
    R_prev = par.epsn_R_prev * (par.R_star -1) * (p_prev / par.p_star) ** (par.A*par.R_star/ (par.R_star-1))
    tau = par.gamma0 + par.gamma * b_prev + epsn_t
    
    m = m_prev/p +  R_prev + b_prev/p - b - tau
    n = y**(1/(1-par.eta)) * 1/(epsn_y)
    
    par.epsn_R_prev = R
    
    if t is None:
        
        m = auxilliary.expand_to_quad(m, train.Nquad)
        b = auxilliary.expand_to_quad(b, train.Nquad)
        p = auxilliary.expand_to_quad(p, train.Nquad)
        c = auxilliary.expand_to_quad(c, train.Nquad)
        n = auxilliary.expand_to_quad(n, train.Nquad)
        
    return torch.stack((m, b, p, c, n),dim=-1)



def state_trans(model,state_trans_pd,shocks,t=None):

    par = model.par
    train = model.train
    
    # b. unpack outcomes
    m_pd = state_trans_pd[...,0] 
    b_pd = state_trans_pd[...,1]
    p_pd = state_trans_pd[...,2]
    c_pd = state_trans_pd[...,3]
    n_pd = state_trans_pd[...,4]
    
    # c. unpack shocks
    epsn_t_plus = shocks[:,0]
    epsn_R_plus = shocks[:,1]
    epsn_R = shocks[:,2]
    epsn_y_plus = shocks[:,3] 

	# b. adjust shape and scale quadrature nodes (when solving)
    if t is None:
        
        m_pd = auxilliary.expand_to_quad(m_pd, train.Nquad)
        b_pd = auxilliary.expand_to_quad(b_pd, train.Nquad)
        p_pd = auxilliary.expand_to_quad(p_pd, train.Nquad)
        c_pd = auxilliary.expand_to_quad(c_pd, train.Nquad)
        n_pd = auxilliary.expand_to_quad(n_pd, train.Nquad)
        

        epsn_t_plus = auxilliary.expand_to_states(epsn_t_plus,outcomes)
        epsn_R_plus = auxilliary.expand_to_states(epsn_R_plus,outcomes)
        epsn_R = auxilliary.expand_to_states(epsn_R,outcomes)
        epsn_y_plus = auxilliary.expand_to_states(epsn_y_plus,outcomes)
	
	# d. finalize
    states_plus = torch.stack((m_pd, b_pd, p_pd, c_pd, n_pd, epsn_t_plus, epsn_R_plus, epsn_R, epsn_y_plus),dim=-1)
    return states_plus




