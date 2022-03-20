#!/usr/bin/env python
# coding: utf-8

# ## Questions 87 - 96

# ### Q87 H$_2^+$ molecular orbitals
# This question is about the energy of H$_2^+$ molecular orbitals. The definitions in 11.2 are used. If $\varphi_1$ and $\varphi_2$ are H atom 1S orbitals show that
# 
# (a) $\displaystyle \int\varphi_1 \frac{q^2}{R}\varphi_2 d\tau=\frac{q^2S}{R}$ where $R$ is the internuclear separation.
# 
# (b) Calculate the resonance integral $\displaystyle A=\int\varphi_1 \frac{q^2}{r_2}\varphi_2 d\tau$ belonging to $H_{12}$ and thereby calculate the energy.
# 
# (c) Use the results from the text and those just calculated and plot the energies $E_+$ and $E_-$ with internuclear separation and so reproduce figure 31. Use $\rho=a_0R, a_0=1$ and electronic charge $q=1$.
# 
# ### Q88 Arc length
# Calculate the arc length for 
# 
# (a) a circle of radius $R$, 
# 
# (b) the logarithmic or equiangular spiral $r = e^{-\theta/a}$ from $0 \to 2\pi$, 
# 
# (c) the catenary $y = \cosh(x)$ from $x = 0 \to x_0$, and 
# 
# (d)  the Archimedean spiral $r = a\theta$ from $0 \to 2\pi$.
# 
# **Strategy:** Use equation 83 or 84.
# 
# ### Q89 Area
# Find the area, the $x$ and $y$ centroids and moments of inertia $I-x, I_y$, and $I_z$ of the ellipse $x^2 + y^2 = 1$.
# 
# 
# ### Q90 Mean value
# Calculate the mean value of $r^2 = x^2 + y^2$ over the ellipse defined in the previous question.
# 
# ### Q91
# If $C$ is a line joining $(0, 0)$ to $(a, b)$ calculate $\displaystyle \int_C e^x\sin(y)dx+e^x\cos(y)dy$.
# 
# **Strategy:** Use the two function formula and convert $dy$ into $dy/dx$ where $y$ is determined by the limits on the line, in this case a straight line from the origin to $(a, b)$.
# 
# ### Q92 Area
# (a) Find the area under one arch of the cycloid that is described by the parametric equations
# $x = a(t - \sin(t)),\,  y = a(1 - cos(t))$. A description and sketch of the cycloid is given in Figure 16.
# 
# (b) Find the length of the arch.
# 
# 
# ### Q93 Arc length
# Calculate the arc length for curves **(a)** $r = 1$ and **(b)** $r = e^{-\theta}$ from $0 \to 2\pi$, and **(c)** the catenary $y=\cosh(x)$ from $x=0\to x_0$ where $x_0 \gt 0$.
# 
# ### Q94 Surface area
# The surface area of a function $f(x)$ is given by $\displaystyle A=2\pi\int_a^b f(x)\sqrt{1+f'(x)^2}dx$.
# 
# (a) Show that the surface area of a sphere is $4\pi r^2$ starting with a circle of radius $r$, in which case $f(x) = r^2 - x^2$, and effectively rotating this to form the surface. The integration limits are $\pm r$.
# 
# (b) Work out what fraction of the earth's surface is north of the seaside town of Dunbar, Scotland that is situated at exactly lat $56.00^\mathrm{o}$ N.
# 
# Note: $f'(x)$ is the first derivative. Latitude is the angle from the equator to the pole.
# 
# **Strategy:** In (a) substitute, simplify, and find a very simple integral. In (b) take the south to north axis
# of the earth to be the x-axis and work out the $x$ integration limits.
# 
# ### Q95 State function
# (a) In thermodynamics, what is a state variable?
# 
# (b) The work $w$ required to expand a gas is the line integral $w = -\int p dV$. If $T$ and $p$ are the variables to be used, this equation can be written as 
# 
# $$\displaystyle w =\int p\left(\frac{\partial V}{\partial T}\right)_p dT+p\left(\frac{\partial V}{\partial p}\right)_T dp$$
# 
# For 1 mole of an ideal gas calculate $w$ along each of the two paths used in the example in Section 13.9 and Figure 36 and hence show that $w$ is not a state function.
# 
# **Strategy:** Follow the example and make the integral into one in $dp$ and then $dT$ alone. Substitute for the partial derivatives and use the gas law to substitute variables to make an equation in $p$ or $T$ as necessary. Only then, work out the remaining derivative, $dp/dT$ or $dT/dp$ depending on the path taken.
# 
# ### Q96 Entropy of van-der-Waals gas
# Calculate the entropy for an van-der-Walls gas whose equation of state is $(p+a/V^2)(V-b)=RT$ where $a,\,b$ are constants. Is the entropy different to that of an ideal gas and if so why?
# 
# **Strategy:** Calculate $(\partial V/\partial T)_p$ then use equation 89.  
