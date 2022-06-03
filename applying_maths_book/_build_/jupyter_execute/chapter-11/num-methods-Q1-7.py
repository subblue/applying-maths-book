#!/usr/bin/env python
# coding: utf-8

# ## Questions 1 - 7

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy.integrate import quad,odeint
from scipy.optimize import fsolve
init_printing()                      # allows printing of SymPy results in typeset maths format
plt.rcParams.update({'font.size': 16})  # set font size for plots


# ### Q1 Logistic example
# The equation $x_{n+1} = cx_n(1 - x_n)$ maps $cx_n(1 - x_n)$ onto $x$, 
# 
# see Fig. 2, but many other expressions can do this. Try $x_{n+1} = c \sin(-x_n)$ and $x_{n+1} = ce^{-x_n}$ and see what happens. You may need to change the range of $c$.
# 
# ### Q2 Numerical integration
# Calculate the integral of $\cos(x^2)$ over the range $0 \to 10$, by the trapezoidal rule and by Simpson's rule. Choose the number of points sufficient to guarantee accuracy to three decimal places in each case. This is a highly oscillating function, varying between $\pm 1$, so many terms will be needed for an accurate calculation.
# 
# ### Q3 Numerical integration
# Numerically calculate the integral $\displaystyle \int_0^3\frac{x^6}{1-x}dx$.
# 
# **Strategy:** As this function will become infinite at $x = 1$, the mid-point rule should be used. Choose $100$ points to begin with and then alter this to see how the accuracy varies. The reason for the large number of points is that towards the asymptote $y \to \infty$ at $x = 1$ the function has to be accurately evaluated and values either side of this point have opposite sign and similar value and so almost cancel each other out, but do not always do so exactly.
# 
# ### Q4 Energy of crystalline solid
# The equation $\displaystyle E =\frac{9N_0k_BT}{x_m^3}\int_0^{x_m}\frac{x^3}{e^x -1}dx$ 
# 
# describes the total energy per mole of a crystalline solid and is used in the Debye theory of the heat capacity of solids. This theory assumes that a solid has a range of vibrational frequencies from zero up to a maximum $\nu_m$, each of which contributes to the total energy. The integral cannot be evaluated algebraically other than at limits of large and small $x$. The variable $x$ is defined as $x = h\nu/k_BT$ and the upper limit $x_m$ is defined similarly but with a maximum frequency $\nu_m$ which, for example, in solid Li is $8.06 \cdot 10^{12} s^{-1}$.
# 
# (a) Evaluate the integral by three numerical methods at $1000$ K using a $50$ point integration and quote the answer to six figures.
# 
# (b) Calculate the low and high temperature limits to the integral algebraically, and calculate the heat capacity CV and compare with the numerical results. Show that at low temperature the heat capacity varies as the cube of the temperature.
# 
# **Strategy:** The trapezoidal and Simpson's rule will fail because at exactly $x = 0$ the denominator is infinity, and the limit making the function zero at $x = 0$ cannot be evaluated by the algorithm. This can be overcome by making the lower limit slightly larger than zero, for example $10^{-6}$, which can be done in this instance only because the function,s value is almost zero when $x$ is close to zero, and will not appreciably affect the integral.
# 
# ### Q5 Slowly converging integral
# As discussed in the text, the integral $\displaystyle \int_0^\infty \frac{1}{1+x^2}dx$  
# 
# converges slowly as $x$ increases and when integrated numerically, the result gets worse as the integration upper limit increases. Use the transformation $x = e^{uz} - 1$ to evaluate the integral numerically.
# 
# **Strategy:** The algebra to do the calculation is shown in the text. The choice of $u$ is not crucial, it could be $1$ or $10$, but the numerical limits have to be inspected to ensure the entire curve is included in the summation.
# 
# ### Q6 Tunnelling
# Quantum mechanical tunnelling is a commonly observed phenomenon. It happens when alpha particles leave the nuclei of heavy radioactive nuclei, in the scanning tunnelling microscope, where electrons tunnel from the tip to the substrate, and in molecules it is observed when H and D atoms pass through a potential barrier. The probability of tunnelling through a finite width barrier at energy $E$ is given by $G$ where 
# 
# $$\displaystyle \ln(G_E)=-\frac{\sqrt{2m}}{\hbar}\int_{x_1}^{x_2}\sqrt{V(x)-E}\;dx$$
# 
# and $V(x)$ is the barrier shape and $E$ the energy.
# 
# Numerically integrate and obtain the permeability at several values of the energy $0 \le E \le E_0$ for the Eckardt barrier 
# 
# $$\displaystyle V(x) = E_0\mathrm{ sech}^2(ax)$$
# 
# which is symmetrical and has a width $L$. For simplicity, assume $m/\hbar$ and $E_0$ are unity and $a = 0.5$. Repeat the calculation by doubling $m/\hbar$ and $a$ and observe what happens.
# 
# **Strategy:** Before the integration can be performed, the limits have to be determined. As the barrier is symmetrical, the limits are also. At an energy $V(x) = E$, the value of $x$ is 
# 
# $$\displaystyle x_E=\frac{1}{a}\mathrm{sech}^{-1}\sqrt{E/E_0}$$
# 
# and the limits are therefore $\pm x_E$.
# 
# ### Q7 Scattering angle
# Calculate the scattering angle $\chi$ vs impact parameter $b$ for a hard sphere potential of radius $d$ with initial energy $E_0$. The hard sphere has a potential of zero at distances $r \gt d$ and is otherwise infinite. The result is shown in Fig. 5.
