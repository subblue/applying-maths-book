#!/usr/bin/env python
# coding: utf-8

# ## Questions 14 - 27

# 
# These questions examine the mathematical properties of complex numbers.
# 
# ### Q14
# Calculate $e^i$.
# 
# ### Q15
# Find the real and imaginary parts of 
# 
# (a) $ie^{-ix}$, (b) $e^{in\pi}$, and (c) $e^{in\pi/2}$,
# 
# where $n$ is an integer. 
# 
# ### Q16
# Calculate (a) $i^i$ and (b) $i^{1/i}$.
# 
# **Strategy:** Using different bases such as $a^x = e^{x \ln(a)}$ any number can be raised to any power. With complex numbers always try to put the number in terms of Euler's equation.
# 
# ### Q16
# The cosine function is defined as $\displaystyle \cos(x) = \frac{e^{ix}+e^{-ix}}{2}$. What is $\cos^{-1}(x)$ ?
# 
# **Strategy:** This is a case where $x$ and $y$ are swapped about. If $\cos^{-1}(x)$ then $\cos(y) = x$. It is true also that $\displaystyle \cos(y) = \frac{e^{iy}+e^{-iy}}{2}$. Next eliminate the cosine and solve for $y$ and so find the answer.
# 
# ### Q18
# Show that the identity $(\cos(x) + \sin(x))^2 = 1 + sin(2x)$ can be (relatively easily) proved using complex numbers.
# 
# ### Q19
# Show that $\displaystyle 2\sin\left( \frac{a+b}{2} \right)\cos\left( \frac{a-b}{2} \right) =\sin(a)+\sin(b)$
# 
# ### Q20
# Starting with Euler's theorem and letting $\theta = a + b$, calculate $\sin(a + b)$ and $\cos(a + b)$ by equating real and imaginary parts.
# 
# ### Q21
# Find $\sin(\theta)$ in exponential form then calculate $| \sin(i\theta) |$, and compare it with $| \sin(\theta) |$. Plot values of $|\sin(ix) |$ and $|\sin(x) |$ over the range $x = -4 \cdots 4$.
# 
# ### Q22
# (a) If $z = \cos(x) + i \sin(x)$ show that $dz/dx = iz$. 
# 
# (b) Integrate this result and prove Euler's theorem.
# 
# ### Q23
# Calculate the real and imaginary parts of 
# $\displaystyle \frac{1}{\sqrt{2\pi}}\left( \frac{1-e^{i\omega t}}{i\omega} \right)$. 
# 
# This function is the Fourier transform of a square wave of length $t$. See chapter 9.5, and 9.6.
# 
# **Strategy:** use $i = -1/i$ and multiply out the terms.
# 
# ### Q24 FID in NMR
# In an NMR experiment, the FID signal has the form $\displaystyle s(t) = \sum_j a_je^{i\omega_j t-t/\tau_j}$ where $\omega$  is the frequency of the transition, $\tau$ the average of the T$_1$ and T$_2$ lifetimes, and $a$ the amplitude of each signal and there are $j$ parts to the total signal. For simplicity, assume that $\tau_j$ has a constant value $\tau$.
# 
# (a) Calculate the real, imaginary, and absolute value of $s$ if $j = 2$.
#     
# (b) Plot the real part of the signal if $a_1 =a_2 =2$ and $2\pi\omega =1$ Hz and also when $0.2$ Hz and $\tau_1 =\tau_2 =50$ s and also when $\tau_1 = \tau 2 = 500$ s. Comment on the two results.
# 
# (c) Repeat (a) when $a_1 = i$, which means that the initial amplitude is complex, and $a_2 = 1$.
# 
# In spite of the fact that the signal from an experiment cannot be a complex number, this is what appears to be the case here. The reason for this is that in a real NMR experiment two signals are measured, one by a coil on the spectrometer's x-axis and the other by a similar coil on the y-axis. These are at right angles to the z-axis along which the permanent magnetic field is directed. These x and y signals are measured in quadrature, i.e. 90$^\mathrm{o}$ out of phase to one another. One signal is taken to be the real component, and one the imaginary. They are then combined to produce $s(t)$ given above.
# 
# **Strategy:** The question asks you to find the components which when combined make $s(t)$. Use the Euler formula to do this and to simplify the complex exponential. As the signal represents the FID from an NMR experiment it oscillating in a sinusoidal way.
# 
# ### Q25
# Derive the identities 
# 
# (a) $4 \cos(\theta)\sin^2(\theta) = \cos(\theta) - \cos(3\theta)$
# 
# (b) $4 \sin(\theta)\cos^2(\theta) = \sin(\theta) + \sin(3\theta)$
# 
# **Strategy:** Always use the exponential forms of sine and cosine wherever possible for complicated trig functions. These are $\displaystyle \cos(\theta)=\frac{e^{i\theta}+e^{-i\theta}}{2}, \qquad \sin(\theta)=\frac{e^{i\theta}-e^{-i\theta}}{2i}$
# 
# ### Q26
# If $C$ is the series whose $n^\mathrm{th}$ term is $\cos(nx)/n!$, and $S$ the series $\sin(nx)/n!$, calculate the sum from $n = 1 \to \infty$  of $C + iS$, and hence find the sum $C$.
# 
# **Strategy:** Convert to the exponential form using $e^{ix} = \cos(x) + i \sin(x)$, sum the terms then convert back to trig form and separate out the real part of the result.
# 
# ### Q27 Dielectric property of liquids
# In the study of the dielectric properties of liquids and in electrochemical techniques that use potentiometry, the response of the solution to different electrical frequencies is studied. The general term for these experiments is Impedance Spectroscopy. In an experiment where a capacitor $C$ and resistor $R$ are in parallel, the impedance $Z$, which is a complex quantity, is given by $Z = (R^{1} + i\omega C)^{-1}$ where $\omega$ is the frequency applied to the sample.
# 
# (a) Convert $Z$ into the form $Z = Z' - iZ''$.
# 
# (b) Plot $Z''$ as ordinate, and $Z'$ as abscissa. Show that the resulting curve is a semicircle. Use $R = 5$ k$\Omega$ and $C = 1\, \mu$ F. Decide where high frequency is on the plot. This is not obvious from the graph because $\omega$ is not on one of the axes.
# 
# **Strategy:** Multiply top and bottom of the expression by the complex conjugate. Look up the Matplotlib parametric method of plotting graphs.

# In[ ]:





# In[ ]:





# In[ ]:




