#!/usr/bin/env python
# coding: utf-8

# ## Solutions Q8 - 16

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
init_printing()                      # allows printing of SymPy results in typeset maths format
plt.rcParams.update({'font.size': 16})  # set font size for plots


# ### Q8 answer
# (a) $A = 0.25$. 
# 
# (b) the angular frequency $\omega$ is $125 \,\mathrm{radians\,s^{-1}}$. Using $\omega = 2\pi\nu$ gives the frequency $\nu$ in $\mathrm{s^{-1}}$ (or Hz) and is $19.89 \mathrm{s^{-1}}$.
# 
# (c) The wavevector is $+3$ and the wavelength is given using $k = 2\pi/\lambda$ therefore $\lambda = 2.094$ m. Note that the wavelength is always positive.
# 
# (d) The phase is $\phi=0.6$ radian.
# 
# (e) The velocity is $dy/dt$, which is $v = 125 \cos(125t - 3k + 0.6)/4 \mathrm{\;m\,s^{-1}}$ because the other terms are constant in time.
# 
# ### Q9 answer
# (a) The sum of the two waves is $\sin(\omega_1 t - kx)+ \sin(\omega_2t - kx)$ and we observe the frequency change in time. This means that the $-kx$ term represents a phase, and is fixed as we observe at a fixed point and it can be set to zero without changing the result as phase does not affect the frequency of a wave just where its amplitude is zero. As the waves' amplitudes are the same, we can choose them to be 1. The identity to start with from Section 5.1 is 
# 
# $$\displaystyle \sin(a - b) + \sin(a + b) = 2 \sin(a)\cos(b)$$
# 
# This is has to be rearranged into the form $\sin(A) + \sin(B)$ which can be done letting $A = a - b$ and $B = a + b$ then 
# 
# $$\displaystyle \sin(A) + \sin(B)=2\sin\left( \frac{A+B}{2} \right)\cos\left( \frac{A-B}{2} \right)$$
# 
# Substituting $A=\omega_1t- kx$ and $B = \omega_2t - kx$ gives
# 
# $$\sin(\omega_1t)+\sin(\omega_2t)=2\sin\left( \frac{(\omega_1+\omega_2)t}{2} \right)\cos\left( \frac{(\omega_1-\omega_2)t}{2} \right)$$
# 
# The beat frequency is $|\omega_1 - \omega_2|$ not half this value, because we observe two beats per period.
# 
# (b) A wave period of $T$ has a frequency of $1/\nu$. In the figure, the longest wavelength has a period of $4$ s making the observed frequency $0.25 \,\mathrm{s^{-1} }$. The beat frequency is, by definition, twice the wave frequency hence this is $0.5 \,\mathrm{^{-1} }$.
# 
# ### Q10 answer
# $5 \times 51, 210, 1547/864$.
# 
# ### Q11 answer
# Because the factorials increase so rapidly, the answers can be obtained by direct multiplication using the recursion relationship $(n + 1)! = (n + 1)n!$
# 
# $$\displaystyle \begin{array}{lll}
# 0 &1\\
# 1 &1\\
# 2 &2\\
# 3 &6\\
# 4 &24\\
# 5 &120 & \text{ first above } 100\\
# 6 &720\\
# 7 &5040 & \text{ first above }1000\\
# 8 &40320\\
# 9 &362880\\
# 10& 3628800& \text{ first above }10^6\\
# 11& 39916800\\
# 12& 479001600\\
# 13& 6227020800 & \text{first above } 10^9\\
# 14& 87178291200\\
# 15& 1307674368000 & \text{first above } 10^{12}
# \end{array}$$
# 
# ### Q12 answer
# When $n=0$ then $\displaystyle \frac{(x+0-1)!}{(x-1)!}=1$, and for other $n$
# 
# $$\displaystyle \begin{align}
# n=1 \qquad& \frac{x!}{(x-1)!}=x\\
# n=2 \qquad& \frac{(x+1)!}{(x-1)!}=\frac{(x+1)x(x-1)(x-2)\cdots}{(x-1)(x-2)\cdots }=x+x^2\\
# n=3 \qquad& \frac{(x+2)!}{(x-1)!}=\frac{(x+2)(x+1)x(x-1)(x-2)\cdots}{(x-1)(x-2)\cdots }=2x+3x^2+x^3\end{align}$$
# 
# the general result ($n>0$) would seem to be $\prod_{i=1}^n (x+i-1)$.
# 
# ### Q13 answer
# Using the same method as Algorithm 6 with appropriate changes produces the results

# In[2]:


x, n = symbols('x,n')     #  Legendre  change  n-> n-1

def Legen(n, x): 
    if n == 0:
        return 1 
    elif n == 1: 
        return x 
    else: 
        return ( (2*(n-1)+1)*x*Legen(n - 1, x) - (n-1)*Legen(n - 2, x))/(n)
#---------------
for i in range(6):
    print(i,expand( Legen(i,x) ))


# In[3]:


x, n = symbols('x,n')  #  Chebychev

def Cheb(n, x): 
    if n == 0:
        return 1 
    elif n == 1: 
        return x 
    else: 
        return  2*x*Cheb(n - 1, x) - Cheb(n - 2, x)
#------------------
for i in range(6):
    print(i,expand( Cheb(i,x) ))


# The Chebychev series can also be obtained from the function $P_n(x) = \cos\left(n \cos^{-1}(x)\right)$ if $| x | \le 1$ and as $P_n(x) = \cosh\left(n \cosh^{-1}(x)\right)$ if $| x | > 1$.
# 
# ### Q14 answer
# The recursion algorithm sets the initial values and iterates around a loop to find the result. How many times this must occur depends on whether a result has converged to a constant value or not. This can be guessed at to begin with, but to reach an answer to a set number of decimal places, this will have to be checked. This is not done in the following calculation, however, after about twelve iterations the result is constant to approximately ten decimal places.

# In[4]:


# Q14  recursive square root 

N = 23
k = 0
while k**2<= N:                # find largest value k^2< N
    k = k+1
k = k - 1
print('k =', k)

r = np.zeros(20,dtype=float)
r[0]=0
r[1]=1
for i in range(2,20):          # recursion
    r[i] = 2*k*r[i-1] + (N-k**2)*r[i-2] 
    if i % 2 ==0:              # print every even result  i % 2 is 1 mod 2
        print(i,r[i]/r[i-1]-k)
    pass


# ### Q15 answer
# (a & b) From the definition in the question, the recursion equation is $f_n = f_{n-1} + f_{n-2}$ and is applied where $n \ge 3$ and the first two terms are both 1. The algorithm can be written as shown below. The ratio is also printed as in part (b). The golden ratio is $1.618034$.

# In[5]:


# Fibonacci series

def fibo(n):
    if n <= 1:
        return 1
    else:
        return ( fibo(n-1)+ fibo(n-2) )
#----------------------              
for i in range(10):
    print( fibo(i),end=',' )    


# In[6]:


# ratio of series
print('{:s}{:10.7f}\n'.format('goldern ratio', ( 1+np.sqrt(5) )/2) )
f = np.zeros(20,dtype=int)
f[0] = 1
f[1] = 1
print('{:s}'.format('i     f_n   f_n/f_(n-1)')) 
for i in range(2,20):
    f[i] = f[i-1] + f[i-2]
    if i % 2 == 0:
        print('{:2d} {:6d}   {:f}'.format( i, f[i], f[i]/f[i-1])  )
    pass


# (d) The reverse series is $\cdots, -8,5,-3,2,-1,1,0,1,1,2,\cdots$
# 
# ### Q16 answer 
# (a) The first few terms are
# 
# $$\displaystyle \begin{array}{cccccccccc}
# & & & & & 1 & & & & \\
# & & & 1 & & 2 & &  1 \\
# & &1 && 3 & & 3 && 1\\
# &1 & &4& & 6 & &4& & 1\\
# 1 &&5 &&10 &&10 &&5&& 1\end{array}$$
# 
# (b) expanding the left-hand side gives eqn 21; 
# 
# $$\displaystyle \binom{n}{q+1} = \frac{n!}{(q+1)!(n-q-1)!}$$
# 
# and the right 
# 
# $$\displaystyle  \frac{n-q}{q+1}\binom{n}{q} = \frac{n-q}{q+1}\cdot\frac{n!}{q!(n-q)!}=\frac{1}{q+1}\cdot\frac{n!}{q!(n-q-1)!}$$
# 
# where $(n-q)!=(n-q)(n-q-1)!$ is used in the second step. As the two results are the same, the recursion equation has been shown to be correct.
# 
# To calculate values for $n=12$
# 
# $$\displaystyle \binom{12}{q+1}=\frac{12-q}{q+1}\binom{12}{q}$$
# 
# has to be calculated from $q = 0 \to 12$. As the series is symmetrical, only the first seven terms are needed. The first term starts the process and is $\displaystyle \binom{12}{0}=1$; the second, with $q = 0$, is $\displaystyle \binom{12}{1}=12$; the next gives 
# 
# $$\displaystyle \binom{12}{2}=\frac{11}{2}\binom{12}{1}\binom{12}{0}=\frac{11}{2}12$$
# 
# and then $\displaystyle \binom{12}{3}=\frac{10}{3}\frac{11}{2}12$ and so forth.
# 
# This can be made into a recursive algorithm. The starting value is $1$ and this has to be multiplied by
# $(n - q)/(q + 1)$ each time round a loop in which $q$ starts at zero and ends at the value $n$.

# In[7]:


# Q 16 Binomial coefficient recursion

def binom(n):
    b = 1
    for q in range(n+1):
        print(round(b),end=' ')
        b = b*(n-q)/(q+1)
        pass
#------------
binom(12)


#  The coefficients are $1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1$.
#  
# 

# In[ ]:




