# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.1'
#       jupytext_version: 0.8.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.7.1
# ---

# %% [markdown]
# # Numerical Solution of the Ramsey/Cass-Koopmans model
#
# ## by [Mateo Velásquez-Giraldo](https://github.com/Mv77)
#
# This notebook implements a class representing Ramsey's growth model. Current utilities include:
# - Numerically finding the consumption rule through 'time elimination', as originally implemented by Alexander Tabarrok and updated by Christopher D. Carroll in this [Wolfram Mathematica notebook](www.econ2.jhu.edu/people/ccarroll/public/LectureNotes/Growth/RamseyNumericSolve.zip)
# - Drawing the phase diagram of the model.
# - Simulating optimal capital dynamics from a given starting point.
#
# A formal treatment of the exact version of the model implemented in this notebook can be found in [Christopher D. Carroll's graduate macroeconomics lecture notes](http://www.econ2.jhu.edu/people/ccarroll/public/LectureNotes/Growth/RamseyCassKoopmans/).
#

# %% {"code_folding": [0]}
# Setup
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy import interpolate

from numpy import linalg as LA

# %% {"code_folding": [0]}
# Class implementation

class RCKmod:
    """
    A class representing Ramsey/Cass-Koopmans growth models.
    """
    
    def __init__(self,rho,alpha,theta,xi,delta,phi):
        """
        Inputs:
        - rho:   relative risk aversion coefficient for CRRA utility.
        - alpha: capital's share of production in Cobb-Douglas output function.
        - theta: time preference/discount rate.
        - xi:    population growth rate.
        - delta: capital depreciation rate.
        - phi:   labor productivity growth rate.        
        """
        # Assign parameter values
        self.rho = rho
        self.alpha = alpha
        self.theta = theta
        self.xi = xi
        self.delta = delta
        self.phi = phi
        
        # Create empty consumption function
        self.cFunc = None
        
        # Maximum capital
        self.kmax = (1/(self.phi + self.xi + self.delta))**(1/(1-self.alpha))
        
        # Steady state capital
        self.kss = (alpha/(theta + xi + delta + rho*phi))**(1/(1-alpha))
        # Steady state consumption
        self.css = self.kss**alpha - (xi + delta + phi)*self.kss
        
        # Solve the model to create its consumption function
        self.solve()
    
    def output(self,k):
        """
        Cobb-Douglas normalized production function
        """
        return(k**self.alpha)
    
    def dcdt(self,c,k):
        """
        Consumption differential equation
        """
        dc = c/self.rho*(self.alpha*k**(self.alpha - 1) - self.theta -\
                         (self.xi + self.delta) -self.rho*self.phi)
        return(dc)
    
    def dkdt(self,c,k):
        """
        Capital differential equation
        """
        dk = self.output(k) - c - (self.phi + self.xi + self.delta)*k
        return(dk)
        
    def dcdk(self,c,k):
        """
        Differential equation for the time elimination method.
        This corresponds to dc/dk = (dc/dt)/(dk/dt)
        """
        return(self.dcdt(c,k)/self.dkdt(c,k))
    
    def solve(self, eps = 10**(-8), npoints = 400, lin_approx = True):
        """
        Solves for the model's consumption rule through the time elimination
        method.
        
        Parameters:
        - eps:     disturbance used to prevent dc/dk from becoming 0/0 at
                   the steady state value of capital.
        - npoints: number of points used on each side of the steady
                   state capital for solving the dc/dk equation.
        """
        # K ranges, avoiding kss through a small disturbance
        k_below = np.linspace(self.kss-eps,0.0001,npoints)
        k_above = np.linspace(self.kss+eps,self.kmax,npoints)
        k = np.concatenate((k_below,k_above)).flatten()
        
        # Solve for c on each side of the steady state capital,
        if lin_approx:
            # Using the slope of the saddle path to approximate initial
            # conditions
            c_below = odeint(self.dcdk,
                             self.css - eps*self.slope_ss(), k_below)
            c_above = odeint(self.dcdk,
                             self.css + eps*self.slope_ss(), k_above)
            
        else:
            # Assuming a slope of 1 to approximate initial conditions
            c_below = odeint(self.dcdk, self.css - eps, k_below)
            c_above = odeint(self.dcdk, self.css + eps, k_above)
        
        c = np.concatenate((c_below,c_above)).flatten()
        
        # Create consumption function as an interpolation of the
        # numerical solutions.
        self.cFunc = interpolate.interp1d(k,c)
    
    
    def dkdt_opt(self,k,t):
        """
        Differential equation for k assuming optimal c decisions.
        """
        return(self.dkdt(self.cFunc(k),k))
    
    
    def k_dynamics(self,k0,t):
        """
        Simulates optimal capital dynamics from a given starting point.
        Parameters:
        - t : vector of time points at which to solve for capital
        - k0: value of capital at t[0]
        """
        k = odeint(self.dkdt_opt, k0, t)
        return(k)
    
    
    def k0locus(self,k):
        """
        Returns the consumption value that leaves a given ammount of
        effective capital unchanged.
        """
        return(self.output(k) - (self.phi + self.xi + self.delta)*k)
        
    def phase_diagram(self, npoints = 200, arrows = False, n_arrows = 5):
        """
        Plots the model's phase diagram.
        - npoints:  number of ticks in the k axis.
        - arrows:   boolean to indicate whether or not to draw arrow
                    grid.
        - n_arrows: controls the number of arrows in the grid
        """
        
        k = np.linspace(0.01,self.kmax,npoints)
        
        # Plot k0 locus
        plt.plot(k,self.k0locus(k),label = '$\\dot{k}=0$ locus')
        # Plot c0 locus
        plt.axvline(x = self.kss,linestyle = '--',
                    label = '$\\dot{c}=0$ locus')
        # Plot saddle path
        plt.plot(k,self.cFunc(k), label = 'Saddle path')
        # Plot steady state
        plt.plot(self.kss,self.css,'*r', label = 'Steady state')
        
        # Add arrows ilustrating behavior in different parts of
        # the diagram.
        # Taken from:
        # http://systems-sciences.uni-graz.at/etextbook/sw2/phpl_python.html
        if arrows:
            x = np.linspace(k[0],k[-1],n_arrows)
            y = np.linspace(self.cFunc(k[0]),self.cFunc(k[-1]),n_arrows)
            
            X, Y = np.meshgrid(x,y)
            dc = self.dcdt(Y,X)
            dk = self.dkdt(Y,X)
            
            M = (np.hypot(dk, dc))
            M[ M == 0] = 1.
            dk /= M
            dc /= M
            plt.quiver(X, Y, dk, dc, M, pivot='mid', alpha = 0.3)
        
        # Labels
        plt.title('Phase diagram and consumption rule\n(normalized by efficiency units)')
        plt.xlabel('k')
        plt.ylabel('c')
        plt.legend()
        plt.show()
        
    
    def J_matrix(self,c,k):
        """
        Returns the matrix of first derivatives of the solution's dynamic system
        evaluated at the point (c,k).
        This matrix is used for linear approximations of the system around point
        (c,k).
        """
        
        J = np.array([[1/self.rho*(self.alpha*k**(self.alpha - 1)-\
                                   self.theta-self.xi-self.delta-self.phi),\
                       c/self.rho*\
                       self.alpha*(self.alpha - 1)*k**(self.alpha - 2)],
                      [-1,
                       self.alpha*k**(self.alpha-1) -\
                       (self.phi + self.xi +self.delta)]])
        
        return(J)
    
    def slope_ss(self):
        """
        Finds the slope of the saddle path at the steady state.
        """
        J = self.J_matrix(self.css,self.kss)
        
        # Find eigenvalues and eigenvectors
        w, v = LA.eig(J)
        # Find position of smallest eigenvalue
        min_eig = np.argsort(w)[0]
        
        # The slope of the saddle path is that
        # generated by the eigenvector of the
        # negative eigenvalue.
        slope = v[0,min_eig]/v[1,min_eig]
        
        return(slope)

# %% [markdown]
# ## _Example_
#
# This is a quick example of how the class is used.
#
# An instance of the model is first created by assigning the required parameter values.
#
# The model needs to be solved in order to find the consumption rule or 'saddle path'.

# %% {"code_folding": [0]}
# Create and solve model
RCKmodExample = RCKmod(rho = 2,alpha = 0.3,theta = 0.02,xi = 0.01,
                       delta = 0.08,phi = 0.03)
RCKmodExample.solve()

# Test the consumption rule
print('Consumption at k = %1.2f is c = %1.2f'\
      % (RCKmodExample.kss/2, RCKmodExample.cFunc(RCKmodExample.kss/2)))

# %% [markdown]
# The model's phase diagram can then be generated.

# %%
RCKmodExample.phase_diagram(arrows= True, n_arrows = 12)

# %% [markdown]
# The class can also be used to simulate the dynamics of capital given a starting point.

# %% {"code_folding": [0]}
# Create grid of time points
t = np.linspace(0,100,100)

# Find capital dynamics at the desired time points and with
# a given starting capital
k0 = 4
k = RCKmodExample.k_dynamics(k0,t)

# Plot
plt.plot(t,k)
plt.axhline(y = RCKmodExample.kss,linestyle = '--',color = 'k', label = '$\\bar{k}$')
plt.title('Capital')
plt.xlabel('Time')
plt.legend()
plt.show()

# %% [markdown]
# With capital, the consumption rule can be used to find the dynamics of consumption.

# %% {"code_folding": [0]}
# Find consumption
c = RCKmodExample.cFunc(k)

# Plot
plt.plot(t,c)
plt.axhline(y = RCKmodExample.css,linestyle = '--',color = 'k', label = '$\\bar{c}$')
plt.title('Consumption')
plt.xlabel('Time')
plt.legend()
plt.show()

# %% [markdown]
# # Why use the saddle path slope?
#
# The following example shows an instance in which the solution method with the default disturbance size succeeds when using the saddle path slope, and fails when it is not used.

# %% {"code_folding": [0]}
# We create a model with a high value for rho
RCKmodExample2 = RCKmod(rho = 12,alpha = 0.3,theta = 0.02,xi = 0.01,
                        delta = 0.08,phi = 0.03)

# %% {"code_folding": [0]}
# Solving with the saddle path slope approximation generates the
# usual phase diagram
RCKmodExample2.solve(lin_approx = True)
RCKmodExample2.phase_diagram(arrows= True, n_arrows = 12)

# %% {"code_folding": [0]}
# However, not using the approximation generates a downward-sloping
# consumption rule.
RCKmodExample2.solve(lin_approx = False)
RCKmodExample2.phase_diagram(arrows= True, n_arrows = 12)

# %% [markdown]
# # Appendix: finding the slope of the saddle path at the steady state
#
# From the solution of the model, we know that the system of differential equations that describes the dynamics of $c$ and $k$ is 
#
# \begin{align}
# \begin{bmatrix}
# \dot{c_t}\\
# \dot{k_t}
# \end{bmatrix}
# =
# \begin{bmatrix}
# f(c_t,k_t)\\
# g(c_t,k_t)
# \end{bmatrix}
# =
# \begin{bmatrix}
# \frac{c_t}{\rho}(\alpha k_t^{\alpha - 1} - \theta - \xi - \delta) - \phi\\
# k_t^\alpha - c_t - (\phi + \xi + \delta)*k_t
# \end{bmatrix}
# \end{align}
#
# We seek to approximate this system around the steady state $(\bar{c},\bar{k})$ through
#
# \begin{align}
# \begin{bmatrix}
# \dot{c_t}\\
# \dot{k_t}
# \end{bmatrix}
# \approx
# \begin{bmatrix}
# f(\bar{c},\bar{k}) + f_c(\bar{c},\bar{k})(c_t - \bar{c}) + f_k(\bar{c},\bar{k})(k_t - \bar{k})\\
# g(\bar{c},\bar{k}) + g_c(\bar{c},\bar{k})(c_t - \bar{c}) + g_k(\bar{c},\bar{k})(k_t - \bar{k})
# \end{bmatrix}
# \end{align}
#
# For this we find the system's matrix of first derivatives
#
# \begin{align}
# J(c,k) =
# \begin{bmatrix}
# f_c(c,k) & f_k(c,k)\\
# g_c(c,k) & g_k(c,k)
# \end{bmatrix} = 
# \begin{bmatrix}
# \frac{1}{\rho}(\alpha k^{\alpha - 1} - \theta - \xi - \delta) - \phi & \frac{c}{\rho}\alpha (\alpha-1) k^{\alpha - 2}\\
# -1 & \alpha k^{\alpha - 1} - (\phi + \xi + \delta)
# \end{bmatrix}
# \end{align}
#
# Given the saddle-path stability of the system, $J(c_{ss},k_{ss})$ will have a positive and a negative eigenvalue. The slope of the saddle path at the steady state capital is given by the eigenvector associated with the negative eigenvalue.
