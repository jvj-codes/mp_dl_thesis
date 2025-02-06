# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:13:54 2025

@author: JVJ
"""

import numpy as np
import torch

from EconDLSolvers import expand_to_quad, expand_to_states, discount_factor, terminal_reward_pd, exploration

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
def outcomes(model, states, actions, t0=0, t=None):
    
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
    epsn_y = states[...,7] #exogenous technology shock
    
    if t is None: # when solving with DeepVPD or DeepFOC
        if len(states.shape) == 8: # when solving
            T,N = states.shape[:-1]
            r.y.reshape(par.T,1).repeat(1,N)
    
  		else:
  			T = states.shape[0]
  			N = states.shape[1]
  			y = par.y.reshape(par.T,1,1).repeat(1,N,train.Nquad)

        #actions lead to changes in the envoirment
        y = epsn_y*n_act**(1-par.eta)
        p = c_act/y
        c = c_act/p
        b = b_act/p
        n = y**(1/(1-par.eta)) * 1/(epsn_y)

        return torch.stack((c,m,m),dim=-1)
        
#%% Reward
def reward(model,states,actions,outcomes,t0=0,t=None):
	""" reward """

	par = model.par

	# a. consumption
	c = outcomes[...,0]
	m = outcomes[...,1]
    n = outcomes[...,3]

	# b. utility
	u = utilevans_hon(c, m, c, par)

	# c. finalize
	return u 

def reward(model, action: np.array, outcomes: np.array, util = "evans") -> float:
    """

    Parameters
    ----------
    model : TYPE
        models DeepV, DeepVPD etc.
    action : np.array
        the action of the household.
    outcomes : np.array
        the outcome of the action.
    util : TYPE, optional
        DESCRIPTION. The default is "iacoviello".

    Returns
    -------
    float
        returns the reward for the actions taken.

    """
    
    return NotImplementedError

import numpy as np
from EconDLSolvers import DeepVPD

# Utility function
def utility(c, m, n, sigma, phi, chi):
    return (c**(1 - sigma) / (1 - sigma)) + (chi * m**(1 - sigma) / (1 - sigma)) - (n**(1 + phi) / (1 + phi))

# Reward function
def reward(state, action, params):
    c, m, n = action
    sigma, phi, chi = params['sigma'], params['phi'], params['chi']
    return utility(c, m, n, sigma, phi, chi)

# State transition function
def transition(state, action, params):
    m_prev, b_prev, π_prev = state
    c, m_new, n = action
    β, R, τ, π_target = params['beta'], params['R'], params['tau'], params['pi_target']
    π = c / π_prev  # Inflation from market clearing
    b_new = (1 + R / π_prev) * b_prev + m_prev - m_new - τ
    return np.array([m_new, b_new, π])

# Solve using DeepVPD
def solve_drl(params):
    solver = DeepVPD(
        state_dim=3,  # State: (money holdings, bonds, inflation)
        control_dim=3,  # Actions: (consumption, new money holdings, labor)
        reward=lambda s, a: reward(s, a, params),
        transition=lambda s, a: transition(s, a, params),
        beta=params['beta'],  # Discount factor
        state_bounds=[[0, 10], [-10, 10], [0.5, 2]],  # Bounds for states
        control_bounds=[[0.1, 5], [0, 10], [0, 2]],  # Bounds for actions
        max_steps=1000,  # Training steps
        T=30,  # Time horizon
        network_layers=[64, 64],  # Neural network layers
        name="EconomicModel"
    )
    solver.train()
    return solver

# Parameters for the model
params = {
    "beta": 0.99,  # Discount factor
    "sigma": 3.0,  # Risk aversion
    "phi": 1.0,  # Frisch elasticity inverse
    "chi": 0.1,  # Weight of money in utility
    "R": 1.01,  # Nominal interest rate
    "tau": 0.02,  # Tax rate
    "pi_target": 1.01  # Inflation target
}

# Train the model
solver = solve_drl(params)

# Test results
print("Policy function at test states:")
test_states = np.array([[1, 1, 1], [2, -1, 1.02], [1, 2, 0.98]])
print(solver.get_policy_function()(test_states))


