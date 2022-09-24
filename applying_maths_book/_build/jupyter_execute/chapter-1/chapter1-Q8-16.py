#!/usr/bin/env python
# coding: utf-8

# # Questions 8 - 16

# ## Q8 Waves
# The particles in a wave have displacement defined as $y = 0.25\sin(125t - 3x + 0.6)$ metres. 
# What is 
# 
# (a) the amplitude, 
# 
# (b) the frequency in s$^{-1}$, 
# 
# (c) the wavelength in metres, 
# 
# (d) the phase, 
# 
# (e) the wave velocity? 
# 
# You will need to know how to differentiate to do part (e).
# 
# ## Q9 Beating sound waves
# (a) If two sound waves of equal amplitude and phase and of frequency $\omega_1$ and $\omega_2$ are travelling in the same direction, show that the frequencies of the resulting waves are $\omega_1 \pm \omega_2$ and hence exhibit beating as shown in figure 13 when observed at some position $x$.
# 
# (b) In this figure convince yourself that the beat frequency is 0.5 Hz.
# 
# **Strategy:** Use the appropriate trigonometric identity from Section 5. The beat frequency is the difference of the two frequencies but the lower frequency of the summed wave is half the difference in frequency. Because the beating occurs in time, the term $-kx$ and phase $\phi$ can be ignored in the general wave equation 16.
# 
# ## Q10 Factorials
# Calculate $\displaystyle \frac{52!}{50!},\;\frac{10!}{6!4!},\;\frac{52!}{10!48!}$
# 
# ## Q11 Factorials
# For what $n$ does $n!$ first exceed $100,\; 10^3,\; 10^6,\; 10^9,\; 10^{12}$?
# 
# ## Q12 Polynomial
# Calculate the polynomial functions given by equation 17 when $n = 0, 1, 2$ and $3$.
# 
# ## Q13 Recursion formula
# The recursion equation for the Legendre polynomials is
# 
# $$\displaystyle (n + 1)P_{n+1}(x) = (2n + 1)xP_n(x) - nP_{n-1}(x), \qquad \text{where} \qquad P_0(x) = 1,\; P_1( x) = x$$
# 
# and for the Chebychev polynomials,
# 
# $$\displaystyle T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x),\qquad \text{where}\qquad  T_0(x) = 1,\; T_1(x) = x$$
# 
# Calculate the first six polynomials in each case. 
# 
# ## Q14 An ancient calculation
# An ancient way to calculate the square root of a number $N$ is to use the recursion formula
# 
# $$\displaystyle  r_i=2kr_{i-1}+(N-k^2)r_{i-2}$$
# 
# where $k$ is the largest integer such that $k^2 \lt N$, and then calculate $\displaystyle \frac{r_{i+1}}{r_i}-k$ which approximates the square root. 
# 
# Calculate $\sqrt{23}$. The initial two $r$ values can be chosen to be $0$ and $1$.
# 
# ## Q15 Fibonacci series
# (a) Use a recursion equation and Python if you wish, to calculate the Fibonacci series whose first two values are 1 and all other values are the sum of the previous two.
# 
# (b) Show numerically that the ratio of two adjacent Fibonacci numbers tends to the golden ratio.
# 
# (c) If the recursion expression is $f_n = 2f_{n-1} + f_{n-2}$ 
# 
# show that the ratio of two adjacent numbers tends to $1 + \sqrt{2}$ and if the equation is $f_n = f_{n-1} + 2f_{n-2}$ show that ratio tends to 2. In both these formulae $f_1 = 1$ and $f_2 = 1$.
# 
# (d) Make a Fibonacci series going backwards from the first two terms in the series in (a) by subtracting the next value rather than adding it.
# 
# **Strategy:** Define an array to hold the values $f_n, f_{n-1}$, etc. define the first two values then use a loop to increment values.
# 
# ## Q16 Pascal's Triangle and AX nmr spectra
# Pascal's triangle is a mnemonic for binomial coefficients. If a pyramidal triangle is made, the coefficients are placed in rows one above the other and any value is found by adding together the numbers one to the left and one to the right from the row above. If a right-angled triangle is made, then the numbers added are the one above and the one to the left. 
# 
# The binomial coefficients also form the pattern of splitting, in simple AX type, NMR spectra showing the $n : n + 1$ rule. For example, a CH$_2$ next to CH$_3$ has four lines of intensity $1:3:3:1$.
# 
# (a) Make a Pascal’s triangle by adding numbers as described above.
# 
# (b) Show that the recurrence formula 
# 
# $$\displaystyle \binom{ n}{q+1}=\frac{n-q}{q+1}\binom{n}{q}$$
# 
# where $q=0,1,2\cdots, n-1$, is true and use this to compute all the binomial coefficients for $n=12$ starting with $\displaystyle \binom{n}{0}=1$. See eqn. 21 in the next section 9 (in following pages) for a definition of $\displaystyle\binom{n}{q}$.

# In[ ]:




