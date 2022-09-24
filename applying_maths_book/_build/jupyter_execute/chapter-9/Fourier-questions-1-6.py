#!/usr/bin/env python
# coding: utf-8

# # Questions 1 - 6

# ## Q1  Confirm eqn 3
# Confirm equation (3), describing $b_n$ by using a similar calculation to that used to derive coefficients $a_n$.
# 
# ## Q2  Fourier series
# Calculate the Fourier series for $\sin(x)$ over the range $-L \lt x \lt L$.
# 
# ## Q3 
# (a) Calculate the Fourier series of $\pi/2 - x$ over the interval $-\pi$ to $\pi$. Choose at least 10 terms in the series. If you choose more terms, observe that the overshoot persists but is not of constant height as it is in the square wave.
# 
# (b) Using a general code, test it with the functions $f (x) = x - x^3, \; | x |,\; x^2e^{-x^2/2}$ and $\tanh(x)$, and plot the graphs. Generalize the code to make the limits $\pm L$ and recalculate over the range $\pm$20.
# 
# **Strategy:** (a) the function $\pi/2 - x$ is neither odd nor even and both $a$ and $b$ coefficients will have to be calculated.
# 
# 
# ## Q4 Series expansion
# Calculate the series expansion of
# 
# (a) $f = \cos^2(x)e^{-x/2}$, using Hermite polynomials over the range $\pm 7$, and 
# 
# (b) $f = x + x^3/10 - 2x^7$ over the range $\pm 1$, using Chebychev polynomials. 
# 
# **Strategy:** Write some code to calculate the polynomials or use the SymPy/Scipy functions as appropriate.
# Use the algorithm used in the text, taking care to add the correct weighting and normalization terms, these are given in the text also in Section 4. The calculation is a little awkward for the Chebychev polynomials, because an exception has to be made for the term $n$ = 0. 
# 
# ## Q5 Generating function
# (a) Use the generating function method, 
# 
# (b) the derivative formula and 
# 
# (c) the recursion formula
# 
# $$\displaystyle (n+1)L_{n+1}(x)=(2n+1-x)L_n(x)-nL_{n-1}(x), \quad\quad L_0(x)=1,\quad L_1(x)=1-x.$$
# 
# to confirm that these give the same results as the first few Laguerre polynomials. Use SymPy as necessary.
# 
# ## Q6 Polynomials
# In this question the _associated Legendre_ polynomials and spherical harmonics are calculated. The polynomials are obtained by repeated differentiation of the _Legendre_ polynomials $P_l(x)$, defined in Section 4,
# 
# $$\displaystyle P_l^m(x)=(-1)^m(1-x^2)^{m/2}\frac{d^m}{dx^m}P_l(x)$$
# 
# You can see that the Legendre polynomials have to be calculated first then differentiated if this approach is the be used. Alternatively recursion formula may be used then differentiated, alternatively associated Legendre recursion formulae are known.
# 
# The values of $m$ and $l$ are related as 0 $\le m \le l$, meaning that $m$ cannot exceed $l$ or be negative. These functions are used to obtain the _spherical harmonics_, which are the functions that describe the shapes of the atomic orbitals, s, p, d, etc. and other angular momentum properties of atoms and molecules. The spherical harmonics are defined as
# 
# $$Y_{l,m}(\theta,\varphi)=\sqrt{\frac{(2l+1)(l-m)!}{4\pi(l+m)!}} P_l^m(\cos(\theta))e^{im\varphi} \tag{21}$$
# 
# When $m$ is negative then $Y_{l ,-m}(\theta, \varphi) = (-1)^m Y^*_{l ,m}(\theta, \varphi)$ where the * indicates the complex conjugate and $x = \cos(\theta)$. If dealing with atomic orbitals, then $l$ represents the overall angular momentum where $l = 0$ for s orbitals, $1$ for p and $2$ for d. The projection, magnetic, or z-value quantum number is $m$ and defines the orientation in space. In the spherical harmonics, $-l \le m \le l$.
# 
# There is a useful recursion formula on $l$ for the associated Legendre polynomial, which is,
# 
# $$\displaystyle (l - m)P_l^m =x(2l -1)P_{l-1}^m -(l + m - 1)P_{l-2}^m   \tag{22}$$
# 
# and the bracket with $x$ is suppressed for clarity. This formula means that when $m$ is chosen  the polynomial with different $l$ values can be calculated. Note also that to use this equation $l - 2 \ge 0$. 
# 
# A starting value can be found, when $m$ and $l$ are the same with the relationship (see Prest et al. 1986, pp. 180 - 2),
# 
# $$\displaystyle P_m^m=(-1)^m(2m-1)!!(1-x^2)^{m/2}$$
# 
# A double factorial, in general, $n$!!, reduces the index by $2$ each time instead of $1$ as in the normal factorial. The series is thus $n!! = n(n - 2)(n - 4) \cdots$ (6)(4)(2)  if $n$ is even and if $n$ is odd, such as $2n - 1$, the series is one of odd numbers ending in $1$.
# 
# A second starting function, is obtained by substituting $l = m + 1$ into the recursion equation and letting $P_{m-1}^m = 0$. To form the spherical harmonics, use $x = \cos(\theta)$. The formulae are only valid if $|x| \le  1$.
# 
# Using SymPy if necessary, calculate the associated polynomials and the spherical harmonics with $l = 0, 1$, and $2$ and their associated $\pm m$ values.
# 
# **Strategy:** The first step is to calculate the second starting function. Using the information given, and substituting $l = m$ + 1 into (22) produces
# 
# $$\displaystyle P_{m+1}^m = x(2m+1)P_m^m$$
# 
# as the second starting function. Note that in the polynomials 0 $\le m \le l$ , so if $m$ is zero so is $l$.

# In[ ]:




