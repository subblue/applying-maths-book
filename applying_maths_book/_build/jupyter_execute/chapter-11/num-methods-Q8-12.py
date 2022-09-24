#!/usr/bin/env python
# coding: utf-8

# # Questions 8 - 12

# ## Q8 Numerically evaluate a differential equation
# In this problem, a differential equation is solved in two different ways but by three different numerical methods.
# 
# (a) Solve $dy/dt = y \sin(t^2)$ by separating variables and show that an integral is formed that can only be solved numerically.
# 
# (b) Solve the differential equation by a numerical method with $t_0 = 0,\, y(t_0) = 3$ over the interval $-7 \to 7$.
# 
# (c) Confirm that Python/Scipy's $\mathtt{odeint( \cdots)}$ procedure agrees with your calculation. You will need to add $\mathtt{from\;scipy.integrate\; import\;quad,\; odeint }$ if it is not already included
# 
# **Strategy:** (a) Separating variables means rearranging to put terms only in $t$ and only with $y$ on different sides of the equation. (b) Because the starting time is less than $t_0$, equation 20 should be used up to the value $t_0$. Alternatively, a negative $h$ should be used. The time values start at $t_0$ and decrease or increase depending on the sign of $h$. A simple method is to use a subroutine to work out the values and call this with $h$ positive, and then negative. In this example, $t_0 = 0$ so that the range is symmetrical about zero; if $t_0$ was not zero then more care would have to be taken to produce the range required. Algorithm 9 can be modified to do this calculation.
# 
# ## Q9 Scattering trajectory
# Calculate the scattering trajectories produced by a Coulomb potential and by a screened Coulomb or Yukawa potential, 
# 
# $$\displaystyle U(r) = \alpha e^{-\beta r}/r$$
# 
# Before doing the calculation describe what the scattering trajectories should look like. Consider the case when $\alpha$ is positive and negative.
# 
# ## Q10 Differential cross section
# In a scattering experiment, the incoming atomic or molecular beam has a cylindrical symmetry and the elastically scattered beam retains this symmetry. The *differential cross section* $I(E_0, \chi)$ for scattering is defined as the number of particles entering a unit solid angle, in unit time, divided by the incident flux density. A solid angle, measured in steradians, is really an area in the sense that it is the angle subtended by circle drawn on the surface of a sphere. A sphere subtends a solid angle of $4\pi$ sterads, a hemisphere of $2\pi$ and so forth. The differential cross section is given by 
# 
# $$\displaystyle I(E_0,\chi)=\frac{b}{\sin(\chi)}\bigg| \frac{db}{d\chi} \bigg|$$ 
# 
# Calculate this for the hard sphere and Coulomb potentials.
# 
# ## Q11 Throwing a football
# A football is thrown upwards from a height $h$ with initial velocity $v_0$; the friction slowing the ball is proportional to its velocity. The ball bounces elastically, when it reaches the ground. The equation of motion, found by balancing forces, is
# 
# $$\displaystyle m\frac{d^2y}{dt^2} + mc\frac{dy}{dt} + mg = 0$$
# 
# where $m$ is the mass of the ball, $c$ a damping constant, and $g$ the acceleration due to gravity  = $9.8  \,\mathrm{m\,s^{-2}}$. Calculate how the height and velocity of the ball changes with time. Choose your own values for the initial height, mass, and damping constant, which can be zero, but values of $30$ m and $c$ in the range $0 \to 1\,\mathrm{ kg\, s^{-1}}$ will work with time steps of $0.005$ s. 
#     
# Although this problem concerns a ball, the form of the equation is general and a molecule, such as a protein or DNA experiences acceleration and friction when in an electric field, in a centrifuge or in a flowing fluid. However, the friction imposed by the solvent is usually proportionally far greater compared to that of air on the ball.
# 
# **Strategy:** Use equations 26 and 27 for the velocity and follow the method in Algorithm 13. To make the ball bounce, reverse the coordinates when the value of $y$ becomes less than zero. Plot $y$ vs time and $v$ vs time. When the damping is zero, the ball should always return to the same height and this can be used to check that a sufficiently small time step has been chosen. From the equation of motion, the acceleration is no longer constant and is $d^2y/dt^2 = -cv - g$ where $v = dy/dt$.
# 
# ## Q12 Verlet algorithm
# Re-calculate the last problem using the velocity Verlet algorithm.

# In[ ]:




