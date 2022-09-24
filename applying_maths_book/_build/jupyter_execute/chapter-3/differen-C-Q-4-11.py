#!/usr/bin/env python
# coding: utf-8

# # Questions 1-11

# ## Q1 Gas pressure change
# A gas occupies a volume $V\,\mathrm{ m^3}$ at pressure $p$ bar and at some temperature $pV = 1000$ Joules. Express $V$ in terms of $p$ and $\delta V$ in terms of $p$ and $\delta p$. What is the change in volume when the pressure is increased from $1 \to 1.1$ bar?
# 
# **Strategy:** Write $V = 1000/p$, increment $V$ and $p$ then calculate $\delta V$ as $(V + \delta V) - V$.
# 
# ## Q2 Diatomic vibrational frequency
# A diatomic molecule has a vibrational frequency of $v,\mathrm{ s^{-1}}$, a force constant $k\,\mathrm{ N\, m^{-1}}$, reduced mass $\mu$ kg, and the vibrational frequency is given by $\displaystyle v=\frac{1}{2\pi}\sqrt{\frac{k}{\mu}} $.
# 
# (a) If, by isotopic substitution $\mu$ is increased by $1$%, show that the relative or fractional change is 
# 
# $$\displaystyle \frac{\delta v}{v}=1-\sqrt{\frac{1}{1+\delta \mu/\mu}}$$
# 
# which can be approximated as $\displaystyle \frac{\delta v}{v}=\frac{\delta \mu}{2\mu}$. Justify the approximation you make. You will need the series expansion $\displaystyle (1+x)^{-1/2}=1-x/2+3x^2/8-\cdots$.
# 
# (b) If $k = 518.0\,\mathrm{ N\, m^{-1}}$ and $v = 5889.0\,\mathrm{ cm^{-1}}$ what is the absolute change in frequency in s$^{-1}$ or Hz?
# 
# ## Q3 Thermometer
# In the common thermometer, the thermal expansion of a liquid is measured and calibrated to the temperature rise. Mercury or ethanol is often used. If this is held in a $1$ ml reservoir and the capillary of the thermometer has a $0.12$ mm diameter, work out the sensitivity of this thermometer if $\beta$ is the coefficient of volume expansion. The liquid's volume expands as $V = V_0(1 + \beta\delta T)$ for a temperature rise of $\delta T$. Sensitivity is the change in length of the liquid for a 1 K rise in temperature. The constants are $\beta\text{( Hg )} = 1.81 \cdot 10^{-4}\,\mathrm{ K^{-1}}$, and $\beta \text{( EtOH )} = 1.08 \cdot 10^{-3}\,\mathrm{ K^{-1}}$.
# 
# **Strategy:** The sensitivity you need to work out is $\delta L/\delta T$. Use the volume of the capillary to work out its length.
# 
# ## Q4 Differentiate
# (a) Differentiate with respect to $x$, a being constant.
# 
# $\quad$ (i) $3x^4 + \sin(x)$,  (ii)  $10x^{-4} + 5x^2 + a$,  (iii)  $\ln(x) + (ax)^{-1}$, (iv)  $e^{ax} + x^2$
# 
# (b) Differentiate with respect to $a$, $x$ being constant. 
# 
# $\quad$ (i) $3x^4 + \sin(x)$,   (ii)  $e^{ax} + x^2$, (iii)  $y = ax^2$.
# 
# (c) Differentiate $\sin(ax)$, $n$ times with respect to $x$ when $n$ is even and when $n$ is odd. Differentiate up to $n$ = 5 before deciding on the pattern of equations, then use the identity $\cos(x) = \sin(x + n\pi /2)$.
# 
# ## Q5 Differentiate $n$ times
# Differentiate $x^n$, $n$ times, $n$ being a positive integer, and find $\displaystyle \frac{d^n}{dx^n}x^n$
# 
# **Strategy:** In situations like this, where there are repeated operations and $n$ is undefined, it is best to try to get an answer by induction. Start with $n = 1, 2, \cdots$ and so forth, then build up a pattern. Find the answer for some general or intermediate term, such as the $m^\text{th}$, then finally make $m = n$.
# 
# ## Q6 Differentiate $n$ times
# Differentiate $y = e^{-ax}$, $n$ times.
# 
# ## Q7 Ideal gas
# A gas of volume $V\,\mathrm{ m^3}$ has a pressure $p$ bar at a constant temperature $T$ K.
# 
# (a) Using the ideal gas law, show that $\displaystyle \frac{dp}{dV} = cV^{-2}$ where $c$ is a constant.
# 
# (b) Find $\displaystyle \frac{dV}{dp}$ .
# 
# (c) What is the relationship between $\displaystyle \frac{dV}{dp}$ and $\displaystyle \frac{dp}{dV}$ ?
# 
# ## Q8 Throwing a ball
# (a) If a ball is thrown with an initial velocity $u$, the distance it travels in time $t$ is $\displaystyle s = u + \frac{1}{2}at^2$. Find the velocity $v$ at any time $t$.
# 
# (b) What is the meaning of parameter $a$?
# 
# ## Q9 Electric field of light wave
# (a) The electric field of a laser or other plane light-wave of frequency $\omega$ is $\displaystyle E = A e^{i(\omega t - kx +\varphi)}$, where $x$ is the distance from the source, $k$ the wavevector (2$\pi/\lambda$), $\varphi$ the phase, and $A$ is a constant and is the amplitude of the wave at $t = 0,x = 0,\varphi=0$. 
# 
# Show that the $n^\text{th}$ derivative with respect to time of $E$ is $\displaystyle \frac{d^nE}{dt^n}=(i \omega)^n E$
# 
# (b) Calculate the similar derivative for distance $x$.
# 
# **Strategy:** Start with the first derivatives, look for a pattern and substitute for $E$ into your answer. This equation for the electric field represents a general wave because $e^{i\theta} = \cos(\theta) + i \sin(\theta)$. This equation is described in more detail in Chapter 1.
# 
# ## Q10 Diffusion
# The rate of diffusion / area / time in a solution of ions is described as 
# 
# $$\displaystyle D_F\frac{dc}{dx}=cD\frac{d}{dx}\ln(c\gamma)$$
# 
# where $D_F$ is the Fick's law diffusion coefficient and $D$ is the ( kinematic ) diffusion coefficient, $c$ is the concentration of ions in solution, and $\gamma$ the activity coefficient. This is given by the modified Debye - Huckel expression $\displaystyle \ln(\gamma)=-\frac{A\sqrt{c}}{1+B\sqrt{c}}$ where $A$ and $B$ are constants that depend on the temperature and the solvent and $B$ on the size of the ions.
# 
# (a) Show that $\displaystyle D_F=D\left( 1+\frac{ d\ln(\gamma) }{ d\ln(c) }  \right)$
# 
# (b) Evaluate the derivative.
# 
# **Strategy:** Do not let the unusual form of the differential put you off. Use $d \ln(c) = dc/c$
# 
# ## Q11 Differentiate the integral
# Use eqn. 15 to evaluate $\displaystyle \frac{d}{dx}\int_0^{a/x}\frac{x^2}{e^{-x}+1}dx$

# In[ ]:




