#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib notebook
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sympy import *
#from scipy.integrate import quad,odeint
from scipy.optimize import fsolve
init_printing()                      # allows printing of SymPy results in typeset maths format
plt.rcParams.update({'font.size': 14})  # set font size for plots


# **Q59 answer**  (a) The moment about the C$_4$ axis, which projects out of the plane of the molecule, is $I =4m_FR^2$ if $m_F$ is the mass of a fluorine atom and $R$ the bond length. Its value is $4\times 18.9984\times 1.6604\cdot 10^{-27} \times 4\cdot 10^{-20} =5.047\cdot10^{-45} \,\mathrm{kg\,m^2}$.
# 
# (b) The moment about a diagonal is $2m_FR^2$, because only two F atoms are not on the axis. The moment about the edge,involves calculating the perpendicular distance of the Xe and F atoms from the axis. Using Pythagoras' theorem, the Xe atom is $R/\sqrt{2}$ from the axis and each F atom twice this. The moment is then $m_{Xe}R^2/2 + 2\times 2m_FR^2$, which has a value $1.38\cdot10^{-44}\,\mathrm{ kg\,m^2}$ and is considerably larger than the other two moments because the heavy Xe atom is not on the rotation axis.
# 
# **Q60 answer** (a)and(b) All the masses are at a fixed distance from the z-axis and the total mass of all particles is $M$, therefore, by definition $I_z = MR^2$ we can write down the answer directly. By the perpendicular axis theorem, $\displaystyle I_x=\frac{MR^2}{2}=I_y$. This value is smaller than about the z-axis because the mass is further from this axis than from either the $x$ or $y$.
# 
# (c) Inertia about the edge can be calculated by the parallel axis theorem, and should be that about the centre of mass of the z-axis plus $MR^2$, or $I = 2MR^2$. This is greater than $I_z$ because the mass is further from the axis on average, than is motion about the centre of the loop.

# In[ ]:




