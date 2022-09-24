#!/usr/bin/env python
# coding: utf-8

# # Questions 1 - 6

# ## Q1 Area under curve
# Calculate the area under the curve $\displaystyle e^{-x} \sin(x^2)$ 
# 
# in the range $0 \to 6$ using the mean-value method. 
# 
# **Strategy:** Use the algorithm in the text modified for the new function and limits.
# 
# ## Q2 Mean value method
# Integrate $\displaystyle \int_0^2\int_0^1\int_0^{3/2}\sqrt{16-x^2-y^2-z^2}dxdydz$  
# 
# using the mean value, Monte Carlo method and accurate to two significant figures.
# 
# **Strategy:** Use equation 5 modified for three variables. Repeat the calculation until the answer becomes accurate enough for your needs.
# 
# 
# ## Q3 Importance sampling
# Integrate $\displaystyle \int_o^\infty e^{-x}\sin(x^2)dx$ using importance sampling.
# 
# **Strategy:** The distribution function can be chosen as $e^{-x}$ and the importance sampling algorithm followed. This method also enables us to calculate from zero to infinity; however, a cut-off value has to be determined and $e^{-10}$ can be chosen because it is so small compared to $1$. The limits are therefore $a = 0 ,\; b = 10$.
# 
# ## Q4 Importance sampling
# Calculate the integral $\displaystyle \int_0^2 e^{-x^2}dx$
# 
# using importance sampling and the distribution function $p(x) = 1/(x^2 + 1/4)$, 
# 
# which has a similar shape to the function in the integral.
# 
# **Strategy:**  Following the text on importance sampling, $p(x)$ has to be made into a distribution. The integral from $0 \to t$ is 
# 
# $$\displaystyle \int_0^t (x^2+1/4)^{-1}dx = 2\tan^{-1}(2t)$$
# 
# and the normalization is therefore $N = 2 \tan^{-1}(4)$. Working out the cumulative distribution and solving for $r$ produces $r = \tan^{-1}(2t)/\tan^{-1}(4)$. The variable $r$ is sampled from a uniform distribution in the range $0 \to 1$.
# 
# ## Q5 Accuracy
# Apparently innocent functions, when calculated by simple Monte Carlo methods, can give incorrect results even when huge numbers of sampling points are taken. One example is the integral 
# 
# $$\displaystyle \gamma = \int_0^1x^\gamma dx=\frac{1}{1+\gamma}\quad\text{ when }\quad\gamma \gt 0$$
# 
# (Krauth 2006, p66).
# 
# (a) Accurate values are obtained when $0.5 \lt \gamma \lt 1$. Show that this is the case using a simple,
# or mean-value Monte Carlo method with $\gamma = 1/2$ and $20000$ samples.
# 
# (b) Using the same method show that an incorrect result is produced when $\gamma = -0.8$, and similar values close to $-1$, even though tens or even hundreds of thousands of samples are taken. (Note that this may take several minutes to complete depending on the computer used and how efficient the algorithm is.)
# 
# (c) Next, use the importance sampling method with the distribution $p(x) = x\lambda$ where $\gamma \lt \lambda \lt 0$, and obtain an accurate result.
# 
# ## Q6 Lennard Jones potential & virial coefficient
# The Virial Coefficients are used in the description of real gases. The compression factor $Z = 1$ for an ideal gas, but is expanded as a series for a real gas, 
# 
# $$\displaystyle  Z=\frac{pV}{nRT}=1+B_2\left(\frac{n}{V}  \right)+ B_3\left(\frac{n}{V}  \right)^2 +\cdots$$
# 
# The constants $B_2,\; B_3$, and so forth are the virial coefficients. The second coefficient $B_2$ can be related to the potential energy of interaction between molecules, which leads to non-ideal behaviour. The constant is (Rigby et al. 1986; Murrell & Jenkins 1994; Stone 1996). 
# 
# $$\displaystyle B_2= \int_0^\infty \left(1-e^{U(r)/k_BT} \right)r^2dr$$
# 
# where $U$ is the interaction potential energy at the separation of $r$ between molecules. In the case of a
# Lennard-Jones 6-12 potential, 
# 
# $$\displaystyle U(r)= 4\epsilon\left(\left(\frac{\sigma}{r}\right)^{12}-\left(\frac{\sigma}{r}\right)^{6}   \right) $$
# 
# the integral has no analytic (algebraic) solution and has to be calculated numerically.
# 
# Using the Monte Carlo method, calculate $B_2$ with the parameters for CO$_2$, which are $\epsilon = 140 \,\mathrm{cm^{-1}},\; \sigma = 0.3943$ nm. Boltzmann's constant is $0.693\,\mathrm{ cm^{-1}\, K^{-1}}$.
# 
# **Strategy:** The limits of the integration need to be addressed, because the limit of infinity is not generally possible with the Monte Carlo or other numerical methods. Additionally, the limit when $r = 0$ needs to be checked because here $U(0) = \infty$. In the latter case, the exponential term rescues the situation because $0(1 - e^{-\infty}) = 0$, therefore the limit $r = 0$ is calculable. When $r = \infty$, $U$ is zero, and the whole expression inside the integral (the integrand) is also zero. This can be seen by plotting
# 
# 
# $$\displaystyle f (r) = (1 - e^{-U(r)/k_BT})r^2$$
# 
# and $U(r)$. In practice, a maximum value has to be put on $r$, and using the values given in the question, a plot indicates that $r = 2$ nm is quite sufficient. The function $f (r)$ is zero initially, and rises as $r^2$ for small $r$; however, this term is overwhelmed by the exponential at larger values of $r$.
# 

# In[28]:




