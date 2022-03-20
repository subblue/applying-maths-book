#!/usr/bin/env python
# coding: utf-8

# ## Questions 73 - 77

# ### Q73
# (a) For the particle in a one-dimensional box, calculate the variational energy if the trial wavefunction is $\psi= x(L - x)$ where $L$ is the length of the box for which the potential energy $V = 0$. Compare your result with the true energy for the lowest level which is $\displaystyle \frac{h^2}{8mL^2}$. The Schroedinger equation is $\displaystyle -\frac{\hbar^2}{2m}\frac{d^2}{dx^2}\psi=E\psi$.
# 
# (b) Calculate the variational energy for a finite square well using the trial wavefunction $\psi=e^{-a x^2}$. The potential extends to $\pm \infty$ with a height $U$ except in the range $\pm L$ where it is zero. To obtain a numerical answer for the variational energy assume $\hbar=m=1, U=0.5, L=1$. With this potential the exact value for the only bound level is 0.27312. 
# 
# **Strategy:** (a) Calculate the two integrals in equation 47 separately; remember that $H$ acts on the wavefunction to its right and the result is then multiplied by $\psi$ before the integration is evaluated. (b) Calculate the integrals using Sympy, doing so in three parts, $-\infty \to -1; -1\to 1; 1\to \infty$. When trying to find the optimal value of $\alpha$ the LamberW function is encountered. At this point go back a step substitute the numerical value and recalculate. 
# 
# ### Q74*
# Suppose that in the quartic oscillator $V(x) = kx^4$, the trial wavefunction is chosen to be $\displaystyle  =\sqrt{\frac{\alpha}{\sqrt{\pi}} }e^{-\alpha x^2/2}$ where $\alpha$ is the variable parameter. Calculate the minimum variational energy and compare your result with the numerical solution $\displaystyle E_0=1.060k^{1/3}\left( \frac{\hbar^2}{2m}  \right)^{2/3}$
# 
# ### Q75
# The variational treatment can be used to approximate the ground state energy of the He atom. The energy cannot be calculated exactly because there are three particles, the nucleus and two electrons. Try to find a solution using the normalized wavefunction $\displaystyle \varphi=\frac{1}{\pi}\left(\frac{\zeta}{a_0} \right)^3 e^{-\zeta(r_1-r_2)/a_0}$ with $\zeta$ (zeta) as a variational parameter to replace the nuclear charge $Z;\, a_0$ is the Bohr radius and $r_1$ and $r_2$ are the coordinates of the two electrons, which are independent of one another. The parameter $\zeta$ should be smaller than $Z$ because one electron is shielded from the nucleus by the other and vice versa.
# 
# The energy expectation value is a very complicated integral and evaluates to $\displaystyle E=\int \varphi H\varphi du=\frac{e^2}{4\pi\epsilon_0a_0}\left(\zeta^2-2Z\zeta+\frac{5}{8}\zeta\right)$. $H$ is the Hamiltonian operator and $du$ represents integration over all coordinates.
# 
# (a) Look up the formula for the energy of an H atom. Calculate the (zeroth order) ground state energy of the He atom as twice the energy of an H atom with $Z = 2$. What interaction terms are missing in this crude model?
# 
# (b) Calculate the ionization energy of He+ and add this to the experimentally measured first ionization energy of $24.5$ eV then call this the 'experimental' energy of the He atom. Compare this with the energy from (a).
# 
# (c) Find the variational energy of the ground state of the He atom. Compare it with the energy calculated assuming that $\zeta = Z$, and with your 'experimental' value.
# 
# ### Q76
# Using the trial wavefunction $\psi = e^{-ax^2/2}$, calculate the variational energy of the lowest bound state of an electron if the potential has the form of a Gaussian well $V = 1 - e^{-\beta x^2}$. As the lowest level is required expand the potential. Use the Schroedinger equation in atomic units is $\displaystyle \left( -\frac{1}{2}\frac{d^2}{dx^2}+V_x \right)\psi_x=E\psi_x$. Compare the result with the numerical value of $0.5226$ when$\beta=1$ and $0.20473$ when $\beta=1/10$. See chapter 10 for details of numerical methods.
# 
# Comment of the results obtained.
# 
# **Strategy:** The potential is anharmonic with a value of 1 at large values of $\pm x$ and zero at the origin. The expansion of the potential at small $x$ is $1-e^{-\beta x^2}\approx \beta x^2-\beta^2x^4/2\cdots$, and therefore the Gaussian wavefunction suggested as a trial wavefunction should be a good approximation as this is the form of the lowest wavefunction for a harmonic potential. 
# 
# 
# ### Q77
# Using atomic units (see Chapter 1.14.3), the Schroedinger equation for the hydrogen atom is $\displaystyle -\frac{1}{2}\nabla^2\psi-\frac{\psi}{r}=E\psi$, where the operator del or nabla squared is $\displaystyle \nabla^2=\frac{d^2}{dx^2}+\frac{d^2}{dy^2}+\frac{d^2}{dz^2}$.
# 
# Using a trial radial wavefunction $R(r) = e^{\alpha r^2/2}$ and the variational method, show that the ground state energy is $-4/(3\pi) = -0.424$, which is slightly more than the exact energy which is $-1/2$. You will need to use the relationship $\displaystyle \nabla^2f(r)=\frac{1}{r}\frac{d^2}{dr^2}rf(r)$ and the volume elements in spherical coordinates $dxdydz\to r^2\sin(\theta)dr d\theta d\phi$.
# 
# **Strategy:** The Schroedinger equation is given in mixed coordinates Cartesian and radial. This is not unusual because it is assumed that you know how to convert from one to the other; the relationship is given in the question. However, there are three coordinates, not just $r$ to deal with, but the wavefunction is only given in terms of $r$. The reason for this is that the lowest energy corresponds to an s orbital, which is spherically symmetrical. The integrations have the form $\displaystyle \int\int\int R(r)^*R(r)dxdydz\to \int\int\int R(r)^*R(r)r^2\sin(\theta)dr d\theta d\phi$ and because the wavefunction does not depend on $\theta $ and $\phi$ these integrals can be separated out. You must decide what the limits are; the variational parameter is $\alpha$.

# In[ ]:





# In[ ]:





# In[ ]:




