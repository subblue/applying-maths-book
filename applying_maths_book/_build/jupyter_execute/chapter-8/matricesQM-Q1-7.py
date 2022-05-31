#!/usr/bin/env python
# coding: utf-8

# ## Questions 1 - 7

# ### Q1 Transition moments in NMR
# (a) Using the $s, m_z$ basis set of the NMR example, work out the transition moment matrix needed to predict the intensities of the lines in the two-spin NMR spectrum.
# 
# (b) Using the energy levels calculated in the example show that the transitions form an AX spectrum as shown in Figure 3.
# 
# The transition matrix required is the $4\times 4$ matrix of all possible NMR transitions between the four energy levels in a two-spin $1/2$ system. The transitions occur because of the magnetic-dipole interaction with the spins when using right or left circularly polarized radiation in the $x-y$ plane, but none in the z-direction, which is the direction in which the magnetic field $B_0$ is applied. Left or right circularly polarized radiation can only couple transitions between m states, with selection rules $\Delta m=\pm 1;\;\Delta s=0$.
# 
# The total magnetic dipole operator for two spins is $\pmb{\mu} = \gamma_a\pmb{I}_a  + \gamma_b\pmb{I}_b$ , which has units of rad J T$^{-1}$. In the two-spin system, each state is a product of two wavefunctions and so the $i-j^{th}$ entry in the transition probability matrix is
# 
# $$\displaystyle  T_{ij} = \langle sm_{ai},sm_{bi} |\pmb{\mu}|sm_{aj},sm_{bj}\rangle $$
# 
# and as both $s$ quantum numbers are $1/2$, the subscripts have been dropped. This is a similar equation to that in the example but with a different operator. The angular momentum $\pmb{I}$ has $x, y,z$ -direction components, and these generate the magnetic dipole transition moments shown in the table; the dipole $\pmb{\mu}_{\pm} = \pmb{\mu}_x \pm \pmb{\mu}_y$ describes left and right circularly polarized light, and $\pmb{\mu}_z = \mu m$, linearly polarized light. All other combinations of quantum number changes, other than those shown in the table, have zero transition moments. It is only necessary to work with the $\pmb{\mu}_{\pm}$ operators and the moments they produce. There is one $\pmb{\mu}_{\pm}$ term for each spin $a$ and $b$ so $\pmb{\mu} = \gamma a\pmb{\mu}_{\pm a} + \gamma_b\pmb{\mu}_{\pm b}$ and each $\pmb{\mu}_{\pm}$ is two terms as shown in the table.
# 
# $$\displaystyle \begin{array}{c|c|c}
# \hline
# \text{magnetic dipole }\pmb{\mu} & \text{Transition moment non-zero when:} & \text{Transition moment}\\
# \hline
# \text{linearly polarised along}\; z& \pmb{\mu}_z,\; \Delta s=0,\; \Delta m=0 & \mu m \\
# \hline
# \text{Right and left polarised} &  \pmb{\mu}_{\pm},\; \Delta s=0,\Delta m=\pm 1&\pmb{\mu}_z=\mu\sqrt{s(s+1)-m(m\pm 1)} \\
# \text{in x-y plane}& \pmb{\mu}_{\pm}=\pmb{\mu}_x\pm \pmb{\mu}_y & \\ \hline \end{array}$$
#  
#  
# The basis set can be the same as in the text example, equation 24.
# 
# **Strategy:** The integrals $T_{ij}$ have to be calculated using the terms in the table for each of the combinations of $s$ and $m$, of which there are 16, but not as many as this need be calculated because the matrix is symmetrical. The subscripts $i,j$ give the position in the matrix; $i$ is the row, $j$ the column. The vital point is that radiation has one unit of angular momentum. This means that only changes of the $m$ quantum number by $\pm 1$ on one spin of the pair can be coupled by the radiation at any time and hence produce the spectrum. Recall that the wavefunctions of each individual spin are normalized and orthogonal; $\langle m_a | m_a\rangle = 1$ and $\langle m_a | m_b\rangle = 0$. Because the nuclear spin quantum number $s$ is the same for both nuclei the subscripts can be dropped for clarity.
# 
# ### Q2 H atom proton-electron spin coupling
# In the ground state of a hydrogen atom, there is a hyperfine coupling which is caused by the coupling of the proton's spin with that of the electron and this causes a small change in the energy levels, the Hamiltonian has a term $a\pmb{S}\cdot \pmb{I}$ to account for this and $a = 1.42$ GHz. The energy levels in a hydrogen atom are also split in a magnetic field by the electron and nuclear Zeeman effect. If the field is in the
# $z$ direction only, the Hamiltonian is
# 
# $$\displaystyle H = g\beta S_zB - g_n\beta_nI_zB + a\pmb{S}\cdot \pmb{I}$$
# 
# where $g$ is the $g-$factor for the electron and $\beta$ the Bohr magneton, and $g_n$ and $\beta_n$ are the proton $g$ value and nuclear magneton respectively. $S_z$ and $I_z$ are the electron and nuclear (proton) spin quantum numbers respectively.
# 
# Calculate the energy levels and plot a labelled graph vs $B$ in tesla, ($B$ is often called the magnetic field strength; however, it is the magnetic flux density and has the units T = kg A$^{-1}$ s$^{-2}$.) The constants are $g = 2.002, \; \beta = 9.274 \cdot 10^{-24}\,\mathrm{ JT^{-1}}, g = 5.586, \beta = 5.051\cdot 10^{-27}\,\mathrm{ JT^{-1}}$. See Foote  (2005), Cohen-Tannoudji et al. (1977), or Carrington & McLachlan (1969) for a thorough discussion of this problem.
# 
# **Strategy:**  Modify the method used to calculate the NMR spectrum shown in the text. The matrix of expectation values can then be written down directly by analogy with equation 25. To convert to Hz and tesla, divide the constants by Planck's constant.
# 
# ### Q3 Three NMR nuclei
# Write down the Hamiltonian for the interaction of three nuclei such as in an NMR experiment. Construct the matrix using the method outlined in the text or using python by adapting the code given in Algorithm 4. It is possible to decide which entries are zero by examining the basis set because the matrix only has entries $H_{i,j }$ when _both_ $m$ values change by $+1$ and $-1$ or vice versa, or $s$ and $m$ are the same see equations 19 to 21. The basis set has eight terms each with three spin states, $(\alpha, \alpha, \alpha), (\alpha, \alpha, \beta)$, and so forth. The matrix can be blocked into two $1\times 1$ matrices and two $3 \times 3$ matrices if the ordering is so arranged. However, finding the eigenvalues (using Python/numpy) for the $3 \times 3$ matrices produces an exceptionally complex set of equations, and if you want to look at the values, it is best to do so numerically.
# 
# **Strategy:** The three spins produce a matrix of order $2^3 = 8$. The three nuclei will have three chemical shifts but interact pair-wise with $J$ coupling terms, $J_{12}, J_{13}, J_{23}$. The wavefunction is the product of those for three spins but the operator for nucleus a can only work on a spin, not b or c, thus the generic form is
# 
# $$\displaystyle \langle sm_{a1}m_{b1}m_{c1} | op_a | sm_{a2}m_{b2}m_{c2}\rangle = \langle sm_{a1} | op_a | sm_{a2}\rangle\langle m_{b1} | m_{b2}\rangle\langle m_{c1} | m_{c2}\rangle $$
# 
# The non-zero terms in the matrix can be written down once the basis set ordering is decided on,
# only those two entries where $m$ changes by $+1$ and $-1$ are non-zero; for example, the element between two pairs of basis set values $(\beta, \alpha, \beta) \to (\beta, \beta, \alpha)$ is non-zero.
# 
# ### Q4 Hindered rotation in a molecule
# In a molecule such as ethane the two methyl groups do not undergo free rotation but are hindered by one another. The free-rotor model of methyl group rotation is therefore not very accurate, and instead, the molecule moves in a sinusoidal, threefold potential, with eclipsed configurations having more energy than the staggered ones. Consider that the wavefunction for the rotation of the methyl groups is that of a free rotor but modified by a sinusoidal potential of the form
# 
# $$\displaystyle V=\frac{V_3}{2}\left(1-\cos(3\theta)\right)$$
# 
# where $V_3$ is a constant and $\theta$ the rotation angle. This extra potential is going to change the energy levels. The normalized free-rotor wavefunction is
# 
# $$\displaystyle \varphi_m=\frac{1}{\sqrt{2\pi}}e^{im\theta}$$
# 
# with quantum number $m$ restricted to values $0, \pm 1, \pm 2, \cdots$ with Hamiltonian
# 
# $$ \displaystyle H^0=-\frac{\hbar^2}{2I}\frac{d^2}{d\theta^2}$$
# 
# where, in this case, $I$ is the moment of inertia of the methyl group. The Schroedinger equation is therefore $\displaystyle -\frac{\hbar^2}{2I}\frac{d^2\varphi_m}{d\theta^2}=E_m\varphi_m$. 
# 
# 
# (a) Show that $\varphi_m$ is a solution of this last equation with quantum numbers $m$. 
# 
# Find the energy $E_m$ and show that the diagonal matrix elements are $m^2\hbar^2/2I$ and that the off-diagonal values are zero.
# 
# (b) If the total Hamiltonian is changed to $H^0 + H^1$, show that the matrix elements of $H^1$, using the
# potential $V$ and the wavefunctions $\varphi$ are
# 
# $$ \displaystyle H^1_{m,m'}=\frac{V_3}{2}\delta_{m,m'}-\frac{V_3}{4}\delta_{m',m\pm 3}$$
# 
# (c) Calculate the matrix of expectation values in the basis $m = 0, 1, -1, 2, -2, 3, -3, \cdots $  which
# means that the wavefunction in the presence of the potential has the form of equation 11 *viz*; $\psi= \sum_{i=0} v_i\varphi_i $.
# 
# (d) Solve this problem using python/numpy/sympy, and calculate the new energy levels in the presence of the potential $V$. Check that the levels are sorted in order of their energy. Assume that $m = 3$ and, using the algebraic expression, compare the energies as $m$ increases versus those for the free rotor. Use the value $A = \hbar^2/2I = 17\,\mathrm{ cm^{-1}}$, which is the value for protons in methanol rotating about the CO bond and, assume $V = 200\,\mathrm{ cm^{-1}}$ for the restricted rotor and zero for the free rotor.
# 
# **Strategy:** In (a) you are asked to show that the wavefunction is a solution, not prove it; therefore substitute and simplify to get the answer and do the same for part (b). The solution can be found using methods described in Chapter 10.6. Having these answers, construct the basis set and matrix with the same ordering, any basis set ordering can be chosen as long as it is used throughout. Finally, diagonalize the matrix using Python/Sympy. The matrix element for any Hamiltonian H and quantum numbers $m$ and $m'$ is $\int\psi^*_mH\psi_m d\theta \equiv \langle m|H|m'\rangle$ The wavefunction is complex; therefore, remember to take the complex conjugate where necessary.
# 
# The expectation values of operator $H^0$ produce only diagonal terms in the matrix, because, as the saying goes, '$H^0$ is diagonal in its own eigenstates'. This means that $H^0$ exactly solves the Schroedinger equation, $H^0\varphi = E_0\varphi$, and $E_0$ is the energy for each quantum level. In this example, $H^0$ would correspond to that for the rigid rotor with no internal rotation. The other operator $H^1$, has the effect of coupling energy levels and therefore, in calculating the expectation values, the two numbers $m$ and $m'$ are different.
# 
# ### Q5 Hindered rotation in $\mathrm{CH_3CH_2F}$
# The first three transitions observed in the hindered rotational spectrum of $\mathrm{CH_3CH_2F}$, are at $242.7, 225.5, 208.4$ and $177.0\,\mathrm{ cm^{-1}}$ (Sage & Klemperer 1963). Using the method of the previous question, and by fitting the observed energy difference to trial values, find the value of $A = \hbar^2/2I$ in $\mathrm{ cm^{-1}}$. The potential's magnitude is $V_3 = 1158\, \mathrm{ cm^{-1}}$.
# 
# By guessing some values and repeating the calculation with a sufficiently large basis set a value of
# $A = 6.3\,\mathrm{ cm^{-1}}$ provides a fairly good fit to the data. Do you agree?
# 
# ### Q6 Hindered rotation trans, gauche, eclipsed
# When there are trans and gauche as well as eclipsed forms, the potential has a more complicated shape with terms in $\theta$ as well as $3\theta$% and $6\theta$.
# 
# (a) In the free rotor basis used in previous questions, work out the expectation values for the potential
# 
# $$\displaystyle V=\frac{V_1}{2}(1-\cos(\theta) )+\frac{V_3}{2}(1-\cos(3\theta) )+\frac{V_6}{2}(1-\cos(6\theta) )$$
# 
# (b) Construct the matrix of expectation values.
# 
# (c) If you feel confident, write a python/Sympy procedure to work out the eigenvalues.
# 
# ### Q7 Electron in a 2p orbital
# An ion with an electron in one of the 2p orbitals is placed in a field with orthorhombic symmetry with a potential 
# 
# $$\displaystyle V = Ax^2 + By^2 + Cz^2$$
# 
# Except at the origin, this potential must obey Laplace's equation 
# 
# $$\displaystyle \frac{\partial^2 V}{\partial x^2}+\frac{\partial^2 V}{\partial y^2}++\frac{\partial^2 V}{\partial z^2}=0$$
# 
# therefore $A+B+C= 0$ or $C= -A-B$ making the potential 
# 
# $$\displaystyle V = A(x^2 - z^2) + B(y^2 - z^2)$$
# 
# ( An orthorhombic crystal has $90^\mathrm{o}$ angles but different lengths along the $x, y, z$-axis.)
# 
# (a) Calculate the orbitals' energies by setting up the secular equation using the spherical harmonic functions ($Y$)  given below to describe the wavefunctions. Examine the matrix element integrals of the form $\langle Y_{0,\pm 1} | V | Y_{0,\pm 1}\rangle$  and decide which ones are zero without calculating them all. The radial part of any wavefunction can be ignored as it does not depend on $A$ and $B$ and is the same in all directions. The angular parts of the orbitals are
# 
# $$\displaystyle Y_{10}=n\sqrt{2}\cos(\theta),\qquad Y_{1,\pm 1}=\mp n\sin(\theta)e^{\pm i\varphi}$$
# 
# where the first subscript identifies the angular momentum, which is $1$ for a P state, and the second, the $m$ component $0, \pm 1$. Use the $m$ quantum numbers as a basis set, therefore the secular determinant will be $3 \times 3$. Simplify the result by separating out $A$ and $B$ as factors and replacing other terms with a constant and this will be easier to do if the integrals are evaluated with python/Sympy.
# 
# (b) If the orbitals are made into linear combinations as $p_x, p_y$, and $p_z$, without any calculation, explain the form of the matrix where elements are $\langle p_x |V | p_y \rangle$ and so forth.
# 
# (This question is based on a similar one by Squires 1995, Chapter 10.)
# 
# **Strategy:** The potential is in $x, y, z$, the wavefunctions in $r, \theta, \varphi$, so the conversions from Cartesian to spherical polar coordinates are needed. These are 
# 
# $$\displaystyle z = r \cos(\theta),\quad y = r \sin(\theta)\sin(\varphi),\quad z = r \sin(\theta)\cos(\varphi)$$
# 
# see Chapter 4.11. The Jacobian for the change from Cartesian to spherical polar coordinates is also needed, this is 
# 
# $$\displaystyle dxdydz \to r^2 \sin(\theta)drd\theta d\varphi$$
# 
# Examining the integrals first can determine whether they are zero or not. The integration limits on $\theta$ are $ 0 \to \pi$, and $\varphi$ are $0 \to 2\phi$. There are three values of the $m$ quantum numbers therefore, ordering of the basis set could be $0, 1, -1$. The following results are useful;
# 
# $$\displaystyle \begin{align} &\int_0^{2\pi}\cos^2(\varphi)d\varphi =\int_0^{2\pi}\sin^2(\varphi)d\varphi=\pi \\
# &\int_0^{2\pi}\cos^n(\varphi)d\varphi   =\int_0^{2\pi}\sin^n(\varphi)d\varphi= (0 \; \text{if }n\;  \text{odd and}\; \ne 0 \; n\;\text{even} ) \\
# &\int_0^{2\pi}\cos^2(\varphi)e^{\pm 2i\varphi} d\varphi  =-\int_0^{2\pi}\sin^2(\varphi)e^{\pm 2i\varphi} d\varphi\\
# &\int_0^{\pi}\sin^m(\theta)\cos^n(\theta) d\theta = (0 \; \text{if }n\;  \text{odd and}\; \ne 0 \; n\;\text{even} )\end{align} $$

# In[ ]:




