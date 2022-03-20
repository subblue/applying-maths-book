#!/usr/bin/env python
# coding: utf-8

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


# **Q13 answer**  Using the algorithms in the text the following figure can be produced from which it is clear that both the Euler methods fail badly in this instance and even the Runge - Kutta fails when $t$ is large. If the number of integration points is increased to $1000$, the Euler methods improve slightly but still fail, however, the Runge - Kutta is essentially identical to the exact solution but only up to about $t=15$ when it starts to fail badly. This illustrates how difficult, and time consuming numerical calculations can be. Time consuming since very small steps may be needed to ensure accuracy.
# 
# <img src='num-methods-fig33.png' alt='Drawing' style='width:350px;'/>
# 
# Figure 33. Comparison of the Euler methods with the Runge - Kutta and the exact solution, red line.
# ____
# 
# **Q14 answer**  The code in the 'for' loop has to be changed to use the modified Euler equation (33). The derivatives are defined as

# In[2]:


dydt = lambda t,x,y : -(kf+k2)*y+k1*x
dxdt = lambda t,x,y : -(k1+ke)*x+k2*y


# and using the values given in the question the plot produced is 
# 
# <img src='num-methods-fig34.png' alt='Drawing' style='width:350px;'/>
# 
# The plot shows that initially $x$ increases from zeros, because it is produced from $y$, then decays away with rate constant $k_e$. Equilibrium is beginning to be is set up between $x$ and $y$ but this does not last as both species decay with rate constants $k_f$ and $k_e$.
# 
# **Q15 answer** Four rate equations are needed and are,
# 
# $$\begin{array}{ll}
# \displaystyle \frac{d[S]}{dt}=-k_1[E][S] +k_{-1}[ES] & \displaystyle\frac{d[E]}{dt}=-k_1[E][S] +(k_{-1}+k_2)[ES] \\
# \displaystyle\frac{d[ES]}{dt}=k_1[E][S] -(k_{-1}+k_2)[ES] & \displaystyle\frac{d[P]}{dt}=k_2[ES] 
# \end{array}$$
# 
# The functions in the algorithm can be written as 

# In[3]:


dSdt=  lambda S,E,ES : -k1*S*E + km1*ES
dEdt=  lambda S,E,ES : -k1*S*E + (km1+k2)*ES
dESdt= lambda S,E,ES :  k1*S*E - (km1+k2)*ES
dpdt=  lambda S,E,ES :  k2*ES


# and in the 'for' loop as 

# In[4]:


#for i in range(1,n):                # commented out here only to prevent error as this is a stub.
#    S =  S  + h*dSdt( S,E,ES)
#    E =  E  + h*dEdt( S,E,ES)
#    ES = ES + h*dESdt(S,E,ES)
#    p =  p  + h*dpdt( S,E,ES)
#    t = t + h
#    # etc


# The results are shown in Fig 35.
# 
# <img src='num-methods-fig35.png' alt='Drawing' style='width:400px;'/>
# 
# Figure 35 Michaelis - Menten concentration profiles. The intermediate complex ES is represented by the red line.
# _______
# In the figure we see that the product concentration $[P]$ rises to reach the same concentration as the substrate concentration $[S_0]$ and that the enzyme E after initially reacting returns to its initial concentration $ [E_0]$. The substrate concentration falls rapidly to begin with, then more slowly. This is due to establishing the equilibrium between S + E and ES and therefore ES initially rises rapidly and reached a maximum and falls because it is slowly lost to product. The steady state condition is  $d[ES]/dt = 0$ is only approximately satisfied with these rate constants after ES has reached its maximum and extends only to about two seconds. The calculation clearly shows the approximate nature of the steady state approximation, we assume that the gradient is zero but have to be satisfied that it is small. Notice also that the steady state conditions mean that the concentration of the species ES need not be small, just that its gradient with time is small. This is a common misconception in the steady state approach.
# 
# **Q16 answer** Using the substitution $z = dy/dx$, the equation becomes $dy/dx=z$ and $d^2z/dx^2=-xz-y-x$ which has to be split further using $w = dz/dx$ to give 
# $$ \frac{dy}{dx}=z,\quad \frac{dz}{dx}=w,\quad \frac{dw}{dx}=-xz-y+x=1 $$
# 
# The equations are added into the calculation as shown in other answers
# 
# <img src='num-methods-fig36.png' alt='Drawing' style='width:350px;'/>

# In[ ]:





# In[ ]:




