#!/usr/bin/env python
# coding: utf-8

# ## Solutions Q31 - 34

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sympy import *
init_printing()                      # allows printing of SymPy results in typeset maths format
plt.rcParams.update({'font.size': 14})  # set font size for plots


# ### Q31 answer
# The coordinates of vector $V_1$ are $x_1 = r\cos(\alpha)$ and $y_1 = r\sin(\alpha)$ and the second vector is rotated by $\theta$ from the first, so that $V_2$ has x-coordinate 
# 
# $$\displaystyle x_2 = r\cos(\alpha-\theta) = r \cos(\alpha)\cos(\theta) + r \sin(\alpha)\sin(\theta)$$
# 
# and by substitution $x_2 = x_1\cos(\theta) + y_1 \sin(\theta)$. The y-coordinate is
# 
# $$\displaystyle y_2 = r\sin(\alpha-\theta) = -r \cos(\alpha)\sin(\theta) + r \sin(\alpha)\cos(\theta)$$
# 
# and $y_2 = -x_1\sin(\theta) + y_1\cos(\theta)$. Combining the two formulae produces the rotation matrix, equation 11.
# 
# $$\displaystyle \begin{bmatrix} x_2 \\ y_2 \end{bmatrix} = \begin{bmatrix} \cos(\theta) & \sin(\theta)\\ -\sin(\theta) & \cos (\theta) \end{bmatrix} \begin{bmatrix} x_1 \\ y_1 \end{bmatrix}$$
# 
# ### Q32 answer
# The rotation matrix is $\displaystyle \pmb{R}_\theta = \begin{bmatrix} \cos(\theta) & \sin(\theta)\\ -\sin(\theta) & \cos (\theta) \end{bmatrix} $ and its square is 
# 
# $$\displaystyle \pmb{R}_\theta = \begin{bmatrix} \cos(\theta) & \sin(\theta) \\ -\sin(\theta) & \cos(\theta) \end{bmatrix} \begin{bmatrix} \cos(\theta) & \sin(\theta)\\ -\sin(\theta) & \cos (\theta) \end{bmatrix}= \begin{bmatrix} \cos^2(\theta)-\sin^2(\theta) & 2\sin(\theta)\cos(\theta)\\ -2\sin(\theta)\cos(\theta) & \cos^2 (\theta)-\sin^2(\theta) \end{bmatrix}$$
# 
# Substituting the double angle relationships into the last matrix shows that $R(2\theta) = R(\theta)^2$.
# 
# ### Q33 answer
# 
# There is no set answer to this question.
# 
# ### Q34 answer
# As rotation only occurs about the z-axis and by $90^\text{o}$,the product of the three rotation matrices becomes
# 
# $$\displaystyle  \begin{bmatrix} \cos(\theta) & \sin(\theta) & 0\\ -\sin(\theta) & \cos (\theta) & 0 \\ 0 & 0 & 1 \end{bmatrix} =\begin{bmatrix} 0 & 1 & 0\\ -1 & 0 & 0 \\ 0& 0 & 1\end{bmatrix}$$
# 
# An inversion is the matrix $\displaystyle \begin{bmatrix} -1 & 0 & 0\\ 0 & -1 & 0 \\ 0& 0 & -1\end{bmatrix}$ and the rotation-inversion operation is
# 
# $$\displaystyle \begin{bmatrix} -1 & 0 & 0\\ 0 & -1 & 0 \\ 0& 0 & -1\end{bmatrix}\begin{bmatrix} 0 & 1 & 0\\ -1 & 0 & 0 \\ 0& 0 & 1\end{bmatrix}=\begin{bmatrix} 0 & -1 & 0\\ 1 & 0 & 0 \\ 0& 0 & -1\end{bmatrix}$$ 
# 
# (b) Performing the matrix multiplication the other way round produces the same result, therefore $Rot \times Inv = Inv \times  Rot$ and the matrices commute.
# 
# 
# 

# In[ ]:





# In[ ]:




