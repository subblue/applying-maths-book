#!/usr/bin/env python
# coding: utf-8

# ## Questions 13 - 16

# ### Q13 Accurate numerical algorithms
# This question illustrates the importance of using accurate numerical algorithms. Compare the Euler, modified Euler, and Runge - Kutta method to integrate the coupled equations (32) from $0 \to 20$ with initial values, $x_0 = 2,\; y_0 = 1$ at $t = 0$ and integration $500$ points. Plot $y$ vs time for the three methods. The exact solution for $y$ to the equation, which can be obtained by the methods of Chapter 10, is 
# 
# $$y = \sin(t) + 2e^{-t} - 1$$
# 
# and thus at small times the function should decay exponentially then become sinusoidal.
# 
# ### Q14 Euler algorithm
# Change the Euler Algorithm to include the modified Euler or Runge - Kutta equations and then solve the system of equations
# 
# $$\displaystyle \frac{dy}{dt} = -(k_f + k_2)y + k_1x,\qquad \frac{dx}{dt} = -(k_1 + k_e)x + k_2y$$
# 
# over the range $0 \le t \le 1$ with initial conditions $y_0 = 1,\; x_0 = 0$ and with constants $k_f = 10,\; k_1 =2.5,\; k_2 = 2, k_e = 5$. These equations describe two coupled species as shown in the reaction scheme and are similar to the kinetics for excimers described in Q 10.30.
# 
# $$\displaystyle\begin{array}{cccc}
# y &\overset{k_2} { \underset{k_1}{ \overset{\longrightarrow} \longleftarrow } } & x \\
# \quad\downarrow k_f & & \quad \downarrow k_e
# \end{array}$$
# 
# ### Q15 Michaelis - Menten scheme
# The Michaelis - Menten scheme is the simplest description of an enzyme catalysed reaction. The enzyme E and substrate S come into equilibrium with an intermediate complex ES that breaks up into reactants or produces product P and the enzyme is returned to its functioning state having acted as a catalyst by converting S into P. The rate equations are derived from the scheme
# 
# $$\displaystyle E+S \overset{k_1} {  \underset{k_{-1}}{ \overset{\longrightarrow} \longleftarrow } }   ES \overset{k_2}{\longrightarrow} P+E $$
# 
# However, these equations cannot be solved analytically but either have to be solved at steady state, by setting the rate of change of the intermediate ES to zero (see Chapter 10), or have to be solved numerically.
# 
# (a) Using the Euler method, write down and solve the rate equations and plot each species vs
# time up to $10$ seconds using a time step of milliseconds. The initial concentrations are
# 
# $S(0) = 5 \cdot 10^{-3}\,\mathrm{  mol\, dm^{-3}}, E(0) = 1.5 \cdot 10^{-3} \,\mathrm{  mol\, dm^{-3}}, ES(0) = 0 \,\mathrm{  mol\, dm^{-3}}$ 
# 
# and the rate constants are 
# 
# $k_1 = 1000\,\mathrm{dm^3\, mol^{-1}\, s^{-1}}, k_{-1} = 0.05\,\mathrm{ s^{-1}}, k_2 = 1.0\,\mathrm{ s^{-1}}$.
# 
# (b) Explain the shape of the curves produced and identify where the steady state is likely to be valid.
# 
# **Strategy:** The Euler method algorithm has to be changed slightly to add new species instead of the two used in most examples so far; for example, the product is calculated inside the 'for' loop with a term such as
# 
# $$\displaystyle\mathtt{P = P + h*dpdt(x,y,z)}$$
# 
# The rate equations have to defined first, in a lambda function such as 
# 
# $$\mathtt{dpdt=  lambda\; S,E,ES :  k2*ES }$$
# 
# as must the initial concentrations and the arrays to hold the concentrations of the four species.
# 
# ### Q16 Solve eqn.
# Solve the equation 
# 
# $$\displaystyle \frac{d^3y}{dx^3}+x\frac{dy}{dx}+y-x+1=0  \tag{36}$$ 
# 
# from $x  = 0 \to 10$ with the initial conditions $y_0 = 5,\; dy/dx\big|_0 = 1$, and $d^2y/dx^2\big|_0 = 2$. Note that there are three initial conditions, each evaluated at $x = 0$.
# 
# **Strategy:** Define two new variables to represent the derivatives and so split the equation into three and solve numerically. This equation can be solved directly by SymPy, but only by producing integrals that have to be solved numerically.

# In[ ]:




