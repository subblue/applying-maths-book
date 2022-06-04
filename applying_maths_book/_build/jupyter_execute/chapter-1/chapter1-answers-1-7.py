#!/usr/bin/env python
# coding: utf-8

# ## Solutions Q1 - 7

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
init_printing()                      # allows printing of SymPy results in typeset maths format
plt.rcParams.update({'font.size': 16})  # set font size for plots


# ### Q1 answer
# The energy gaps are given in wavenumbers, which are not SI units. To convert the units to frequency, multiply by the speed of light $c$ in cm s$^{-1}$, then by Planck's constant to produce the energy in joules. The smaller energy gap is typical of the rotational motion of a molecule and the larger one vibrational motion.
# 
# Using Python the results can be calculated as

# In[2]:


h = 6.626e-34  # J s
c = 2.9970e10  # cm/s 
kB= 1.3806e-23 # J/mol/K

n1_n0 = lambda deltaE, T: np.exp(-deltaE*h*c/(kB*T) )    # make a function
deltaE= 1.0
T     = 30.0
print('{:8.3f}    {:8.3f} {:8.5g}'.format( deltaE, T, n1_n0(deltaE,T)) )


# The table shows results for different energy and temperatures. The values should be rounded to four figures as only four figures are used in the constants. The effect of lowering the temperature is very dramatic for the larger energy gap but hardly noticeable for the small one.
# 
# $$\displaystyle \begin{array}{lll}
# \Delta E/\mathrm{cm}^{-1} &  T/\mathrm{K}   &  n_1/n_0 \\ 
#    1.000  &   30.000&  0.95319\\
#    1.000  &  300.000&  0.99522\\
# 1500.000  &   30.000& 5.8373\cdot 10^{-32}\\
# 1500.000  &  300.000& 0.0007527
# \end{array}$$
# 
# 
# ### Q2 answer
# $$\displaystyle \begin{array}{l|llll}
# (a) \log_{10}(100) = 2, \text{ as } 10^2 = 100 & (b) \log_{10}(10) = 1, \text{ as } 10^1 = 10. \\
# (c) \log_e(e) = 1. & (d) \log_b(b) = 1 \\
# (e) \log_b(bx) = x & (f) \log_2(1/8) = -\log_2(8) = -3 \text{ as } 2^3 = 8\\
# (g) \log_{10}(n)=-4,; n=10^{-4} & (h)  \log_n(1/16) = -2 \text{ as } \log_n(1/16) = -\log_n(16),
# \; n^2 =16\\
# (i) \log_5(125)=n,; 5^n =125
# \end{array}$$
# 
# ### Q3 answer
# $$\displaystyle \begin{array}{l|lll}
# (a) \ln(3^x) = x \ln(3) &    (b) \ln(a^2/b)=2\ln(a)-\ln(b)\\
# (c) \ln(a^{1/2}/b^3) = (1/2)\ln(a) - 3\ln(b) & (d) \ln(3^{1/(x+6)}) = (x + 6)^{-1}\ln(3)\\
# (e)  \displaystyle \ln\left( \frac{x^{1/2} \sin(x) }{\ln(x^n)}  \right) = \frac{1}{2} \ln(x) + \ln(\sin(x)) - \ln\left(n \ln(x)\right)
# \end{array}$$
# 
# ### Q4 answer
# $\displaystyle \ln\left(1 + s/a)^r\right) = r \ln(1 + s/a)$ which follows because $\displaystyle \ln(x^n) = n\ln(x)$. Converting to exponential form gives $\displaystyle x^n = e^{n\ln(x)}$ and therefore $\displaystyle (1 + s/a)^r = e^{r\ln(1+s/a)} $. 
# 
# More formally let $\displaystyle y=n^x$, taking logs produces $\displaystyle \ln(y) = x \ln(n)$ then 'exponentiating', $\displaystyle y = e^{x\ln(n)}$  produces the general form: $\displaystyle n^x = e^{x\ln(n)}$. Now  substitute $n \to 1 + s/a$  and $x \to r$, the result is $\displaystyle (1 + s/a)^r = e^{r\ln(1+s/a)} $.
# 
# ### Q5 answer
# $$\displaystyle \begin{array}{l|ll}
# (a) \log_{10}(10^6)=6 &  (b) \log_e(e^5)=5 \\
# (c) \ln(e^2)^3 = [2 \ln(e)]^3 = 2^3 = 8 & (d)\displaystyle \log_3(n^6) = \frac{6 \ln(n) }{\ln(3)}
# \end{array}$$
# 
# In (d)  Equation 14 was used to convert to log base e and equation 12 for the $n^2$ term. 
# 
# If we wanted to use base $10$ then, $\log_3(n^6) = 6\log_3(10)\log_{10}(n)$ and then use eqn. 14 again to remove the $\log_3$ term. This produces 
# 
# $$\displaystyle \log_3(n^6) = 6 \log_{10}(10)\log_{10}(n)\;/\;\log_{10}(3) = 6 \log_{10}(n)\;/\;\log_{10}(3)$$
# 
# 
# ### Q6 answer
# He removes $\displaystyle b=V\left(1-\frac{1}{n}\right)$ of the beer on the first mouthful and then the volume is restored. The second mouthful removes the same volume, but removes only $\displaystyle bV\left(1-\frac{1}{n}\right)=V\left(1-\frac{1}{n}\right)^2$ of the beer because it has been diluted and only $V(1-1/n)$ is there initially.
# 
# By induction after $n$ mouthfuls the amount remaining is $\displaystyle V\left(1-\frac{1}{n}\right)^N$. As $n \to \infty$, this product is $V/e$, or $0.368V$, so surprisingly perhaps, he never drinks all his beer and $209$ of the $568$ ml remain.
# 
# ### Q7 answer
# The light transmitted after $n$ layers is to be found. If $I_0$ light is incident on a leaf, after 1 layer this transmits $\displaystyle I_1=I_0\left( 1-\frac{\alpha A}{n} \right)$ and after two layers $\displaystyle I_2=I_0\left( 1-\frac{\alpha A}{n} \right)^2$ and after $n$ layers $\displaystyle I_n=I_0\left( 1-\frac{\alpha A}{n} \right)^n$. 
# 
# As $n \to \infty$ , the limit produces the exponential $\displaystyle I_n = I_0e^{−\alpha A}$ which is
# the familiar expression for the absorption of light by molecules and is called the Beer - Lambert law. This is
# often expressed as 
# 
# $$\displaystyle I_n = I_0e^{-\epsilon [C]L }$$
# 
# where $\epsilon$  is the extinction coefficient ($\mathrm{dm^3 \,mole^{-1}\, cm^{-1}}$) of the molecule at the wavelength used, $[C]$ its concentration mole dm$^{-3}$ and absorption $L$ the path length (cm). The product $\epsilon [C]L$ is also called the optical density. 
# 
# (The opacity of wood, paper, ceramic tea cups, or plates, and so forth, is not mainly caused by absorption, but by the scattering of the light due to the inhomogeneous nature of these materials. This inhomogeneity causes frequent changes in the refractive index of the material and hence scattering. You may have noticed that frosted glass becomes more transparent when wet, because the refractive index of water is more similar to that of glass than air.)

# In[ ]:




