#!/usr/bin/env python
# coding: utf-8

# ## Questions 35 - 40

# ## Q35
# Solve $d^2y/dx^2 = 1/(1 + y^2) + x^2$ 
# 
# with boundary values $y_0 = 10,\; y_5 = -2$. This equation is sensitive to the initial value of $\alpha$.
# 
# **Strategy:** Use the shooting method. Try several values of $\alpha$ to find the region where its true value lies, then 'home in' on this with better estimates.
# 
# ## Q36 Potential 'hole'
# The potential $V(x) = -V_0/\cosh^2(\alpha x)$, 
# 
# sometimes called the Poschl-Teller potential, describes a potential hole and is used to approximate that produced by the diffusion of material in a quantum well or similar device used in opto-electronics. The potential is zero at large $\pm x$ and $-V_0$ at the origin, and the constant $\alpha$ determines the width. Calculate all its eigenvalues and wavefunctions if $V_0 = 10$ and $\alpha = 1$. Compare your answer with the analytic eigenvalues 
# 
# $$\displaystyle E_n=-\frac{\alpha^2}{8}\left( \sqrt{1+\frac{8}{\alpha^{2}}V_0}-2n-1 \right)^2$$
# 
# **Strategy:** Plot out the potential first and estimate the maximum $x$ needed for the integration and also the energy range and increment. Use Schroedinger shooting method algorithm to solve the equations. The energy starts at $-10$ and ends at zero. A step size of $0.1$ is a good starting value. It can be made smaller if you suspect that energies have been missed. Using the shooting method algorithm, as we have defined it, will only work if the minimum energy is zero. This means that $V_0$ should be added to the potential energy and of course subtracted from the eigenvalues at the end of the calculation.
# 
# ## Q37 Potential 'hill'
# Calculate the eigenvalues and wavefunctions for levels $3, 6, 7$, and $8$ for the potential hill
# 
# 
# $$\displaystyle V(x) = +V_0/\cosh^2(\alpha x)$$
# 
# centred in an infinitely deep square well of width $10$ and with $V_0 = 5$ and $\alpha = 1$ up to $10$ units of energy. Comment on the separation of the eigenvalues.
# 
# **Strategy:** Setting the maximum and minimum $x$ value effectively defines the size of the infinitely deep square well.
# 
# ## Q38 Tunneling in ammonia
# The reduced mass used in the ammonia example assumes that the H atoms do not change their relative positions. However, as the N atom tunnels from one side of the plane of the H atoms to the other, the NH bonds must compress. If the transition only involves bending, then the bond lengths must remain unchanged and the reduced mass is now $\mu = 3m_H\left(m_N + 3m_H \sin^2(22^\text{o})\right)/(m_N + 3m_H)$.
# 
# (a) Recalculate the ammonia energy levels.
# 
# (b)  Recalculate the energy levels for ND$_3$, using either of the reduced mass formulae (in text and
# here), and comment on why the results are different from that for NH$_3$.
# 
# ## Q39 HCl & I$_2$ energy levels
# Calculate the first ten eigenvalues for HCl and I$_2$ using the experimentally determined values for the force constants. How closely do your calculated results come to the experimentally determined eigenvalues supposing that your potential is harmonic?
# 
# **Strategy:** Using the true constants means having very small or large numbers in the calculation, therefore convert to atomic units. To estimate how large a displacement is needed, use the potential energy to calculate the width of the potential at an energy of $h\nu(n + 1/2)$ and $n = 10$. Make the maximum displacement twice this value. Remember that both even and odd parity results are required, depending upon whether the quantum number is odd or even. The different initial conditions are given in the text; two sets of calculations should be performed to calculate all the eigenvalues.
# 
# ## Q40 H atom energy levels
# Calculate the first four energy levels for the ns orbitals of the hydrogen atom and compare these with the theoretical values.
# 
# **Strategy:** Because this is a real example, work in atomic units. The Schroedinger equation for the H atom is usually written as
# 
# $$\displaystyle -\frac{\hbar^2}{2m_e}\frac{d^2\psi}{dr^2}+\left( \frac{\hbar^2}{2m_e}\frac{\ell(\ell+1)}{r^2}-\frac{e^2}{4\pi\epsilon_0}\frac{1}{r} \right)\psi = E\psi$$
# 
# where the term proportional to $\ell(\ell + 1)/r^2$ is the kinetic energy associated with the angular motion of the electron. This appears in the equation because it acts as an effective potential, keeping wavefunctions away from the origin when $\ell \ne 0$. In this calculation, $\ell = 0$, because s orbitals are involved, and this term can be ignored. Changing to atomic units gives
# 
# $$\displaystyle -\frac{1}{2}\frac{d^2\psi}{dr^2}-\frac{1}{r}\psi=E\psi$$
# 
# where the energy is in hartree ($1$ hartree = $27.211396$ eV). The radial separation of electron and nucleus is $r$ and the nucleus is assumed to be of infinite mass, and the electron is in units of the Bohr radius; $a_0 = 0.529177\cdot 10^{-10}$ m. Levels with the same principal quantum number n, are degenerate; for example, 2s, 2p are degenerate as are the 3s, 3p, 3d orbitals in this model of the H atom. The term in $\ell$ does not appear in the energy and the degeneracy of the p and d orbitals can be considered as being accidental. Additional interactions, such as spin-orbit coupling, are needed besides the Coulomb interaction between the proton and electron to split this degeneracy.
# 
# It is easier to shift the potential to $1-1/r$ so that the minimum is 0 and maximum 1. Additionally a small number must be added to $x$ to prevent division by zero. A maximum $x = 100$ and step size $0.001$ can be used, a termination of $Q \approx 10^{-6}$ is sufficient. 
# 

# In[ ]:




