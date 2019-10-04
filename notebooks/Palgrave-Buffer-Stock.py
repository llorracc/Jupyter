# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# Setup
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.utilities import plotFuncsDer, plotFuncs
from time import clock
import matplotlib.pyplot as plt
import numpy as np
mystr = lambda number : "{:.4f}".format(number)

from copy import copy

from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

# %% {"code_folding": []}
# Create two consumers, a perfect foresight one and one with shocks to income

# Standard parameters
IdiosyncDict={
    # Parameters shared with the perfect foresight model
    "CRRA": 2.0,                           # Coefficient of relative risk aversion
    "Rfree": 1.03,                         # Interest factor on assets
    "DiscFac": 0.96,                       # Intertemporal discount factor
    "LivPrb" : [0.98],                     # Survival probability
    "PermGroFac" :[1.0],                  # Permanent income growth factor
    
    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd" : [0.0],                  # Standard deviation of log permanent shocks to income
    "PermShkCount" : 7,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [0.8],                  # Standard deviation of log transitory shocks to income
    "TranShkCount" : 7,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.0,                     # Probability of unemployment while working
    "IncUnemp" : 0.0,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.0,                      # Flat income tax rate (legacy parameter, will be removed in future)
    
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin" : 0.001,                    # Minimum end-of-period "assets above minimum" value
    "aXtraMax" : 20,                       # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 48,                     # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac" : 3,                    # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra" : [None],                 # Additional values to add to aXtraGrid
    
    # A few other paramaters
    "BoroCnstArt" : -500,                  # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool" : True,                    # Whether to calculate the value function during solution   
    "CubicBool" : False,                   # Preference shocks currently only compatible with linear cFunc
    "T_cycle" : 1,                         # Number of periods in the cycle for this agent type        
    
    # Parameters only used in simulation
    "AgentCount" : 10000,                  # Number of agents of this type
    "T_sim" : 120,                         # Number of periods to simulate
    "aNrmInitMean" : -6.0,                 # Mean of log initial assets
    "aNrmInitStd"  : 1.0,                  # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
}

# Parameters without uncertainty
PFDict = copy(IdiosyncDict)
PFDict["PermShkStd"] = [0]
PFDict["TranShkStd"] = [0]


# %%
# Solve consumers
IndShockConsumer = IndShockConsumerType(**IdiosyncDict)
IndShockConsumer.cycles = 0 # Make this type have an infinite horizon
IndShockConsumer.solve()

PFConsumer = IndShockConsumerType(**PFDict)
PFConsumer.cycles = 0
PFConsumer.solve()

# %%
# Figure 1
def uP(agent, c):
    return( c**(-agent.CRRA) )
    
def approxOmegaP(agent, a_min, a_max):
    
    # Find the m values generating a_min and a_max
    aux = lambda m: m - agent.solution[0].cFunc(m)
    
    # Find the level of resourses that would generate a_min and a_max as
    # as optimal responses
    m_min = root_scalar(lambda m: a_min - aux(m), x0 = a_min, x1 = a_max).root
    m_max = root_scalar(lambda m: a_max - aux(m), x0 = a_min, x1 = a_max).root
    
    # Create grids
    m_grid = np.linspace(min(m_min, m_max)-1, max(m_min, m_max)+1, 1000)
    c_grid = agent.solution[0].cFunc(m_grid)
    a_grid = m_grid - c_grid
    
    # Omega prime is U' at the optimal C.
    omega_grid = uP(agent, c_grid)
    
    # Construct interpolating function
    omegaP = interp1d(a_grid, omega_grid, kind='cubic')
    
    return(omegaP)

m = 10
m_min = 4
a_grid = np.linspace(m_min, m*0.85, 50)

omegaP_uncert = approxOmegaP(IndShockConsumer, a_grid[0], a_grid[-1])
omegaP_PF = approxOmegaP(PFConsumer,  a_grid[0], a_grid[-1])


 
lab1 = '$\omega_t\'(a) = R \\beta E_t [v_{t+1}\'(aR + \\tilde{y}_{t+1})]$'
lab2 = '$R \\beta v_{t+1}\'(aR + E_t[\\tilde{y}_{t+1}])$'
lab3 = '$u\'(m_t-a)$'
plt.figure()
# Omega uncertainty
plt.plot(a_grid, omegaP_uncert(a_grid), label = lab1)
# Omega Perfect foresight
plt.plot(a_grid, omegaP_PF(a_grid), label = lab2)
# Marginal utility
plt.plot(a_grid, uP(IndShockConsumer, m - a_grid), label = lab3)

plt.xlabel('a')
plt.legend()



# %%
# Figure 2

# Define a function for the delta(m)=0 locus
m0_locus = lambda m: m - (m-1)/(IdiosyncDict["Rfree"]/IdiosyncDict["PermGroFac"][0])

# Define grid of market resources
m_max = 50
m_grid = np.linspace(IndShockConsumer.solution[0].mNrmMin, m_max, 500)

plt.figure()
# Uncertainty solution
plt.plot(m_grid, IndShockConsumer.solution[0].cFunc(m_grid), label = '$c(m)$')
# Perfect foresight solution
plt.plot(m_grid, PFConsumer.solution[0].cFunc(m_grid), label = 'Perf. Foresight $c(m)$')
# Stable resource line
plt.plot(m_grid, m0_locus(m_grid), label = 'Perm. Inc')
# Target
targ = (IndShockConsumer.solution[0].mNrmSS,
        IndShockConsumer.solution[0].cFunc(IndShockConsumer.solution[0].mNrmSS))
plt.plot(targ[0], targ[1], '*')

# Annotations
plt.xlabel('m')
plt.ylabel('c')
plt.annotate('Target',
             xy = targ,
             xytext = (targ[0]+5, targ[1]-0.5),
             arrowprops=dict(facecolor='black', shrink=0.05,
                             headwidth = 3, width = 0.5)
            )
plt.legend()

# %%
