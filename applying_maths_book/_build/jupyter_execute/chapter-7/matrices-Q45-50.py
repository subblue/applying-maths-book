#!/usr/bin/env python
# coding: utf-8

# # Questions 45 - 50

# ## Q45 Eigenvectors & eigenvalues 
# Write down the characteristic equations and find the eigenvectors and normalized eigenvalues of the following two matrices. Check whether the eigenvectors are orthogonal.
# 
# $$(a) \displaystyle \begin{bmatrix}4 & -i\\i & 2 \end{bmatrix} , \qquad (b) \begin{bmatrix} 1& -i\\ -i & 1 \end{bmatrix}$$
# 
# **Strategy:** Expand the determinants of the matrices to find the characteristic equations, then use Python/Sympy, as shown in the text, or, if you wish, do the calculation by hand. The normalization term is $\sqrt{v\cdot v}$. The numpy/Sympy instruction for a dot product is a.dot(b) 
# 
# ## Q46 Eigenvalues
# Find the eigenvalues of the matrix by hand
# 
# $$\displaystyle \begin{bmatrix}
# 1 & 0 & 0 & 0 & 0 & 0 \\
# 0 & 2 & 4 & 0 & 0 & 0 \\
# 0 & 5 & 3 & 0 & 0 & 0 \\
# 0 & 0 & 0 & 2 & 3 & 0 \\
# 0 & 0 & 0 & 3 & 4 & 0 \\
# 0 & 0 & 0 & 0 & 0 & 2 \\
# \end{bmatrix} $$
# 
# and confirm using a computer.
# 
# **Strategy:** This is a block diagonal matrix so use this property to make smaller matrices.
# 
# ## Q47 Solve equations
# Find the characteristic equation, eigenvalues, and eigenvectors of $\displaystyle \begin{bmatrix}0 & -u & v\\u & 0 & 0 \\ -v & 0 & 0 \end{bmatrix}$ Simplify your
# answers using $z^2 = u^2 + v^2$.
# 
# **Strategy:** Use python/Sympy to solve the characteristic equation. The pattern of cofactors are shown in eqn2.
# 
# ## Q48 Butadiene MO energies
# In Section 2.5(iii) the MO energies of butadiene were calculated by the Huckel method, using as a basis set the atomic wavefunctions $(\psi_1, \psi_2, \psi_3, \psi_4)$, where the subscript labels the $n = 4$ atoms. The Huckel matrix is
# 
# $$\displaystyle \begin{bmatrix}
# x & 1 & 0 & 0  \\
# 1 & x & 1 & 0  \\
# 0 & 1 & x & 1  \\
# 0 & 0 & 1 & x  \\
# \end{bmatrix} =0, \qquad \text{where} \qquad x\frac{\alpha-E}{\beta}$$
# 
# (a) Using the eigenvalue - eigenvector method, calculate not only the energies, but also the orbital coefficients, which are the eigenvectors.
# 
# (b) Calculate the delocalization energy, which is the Huckel energy less $n(\alpha + \beta)$ where $\alpha$ is the Coulomb self-energy of a $\pi$ electron and $\beta$ the overlap energy. 
# 
# (c) Calculate the bond order, charge density, and dipole moment. The bond order is 
# 
# $$\displaystyle \rho_{ab}=\sum_i^n m_ic_{ai}c_{bi}$$
# 
# where $c_{ai}$ is the coefficient on carbon atom $a$ and of orbital $i$ and $m_i = 0, 1,  2 $ and is the number of electrons in orbital $i$. The total bond order is larger by $1$ when the $\sigma$ bond order is added. 
# 
# The charge density is 
# 
# $$\displaystyle q_a = \sum_i m_i|c_{ia}|^2$$
# 
# and dipole moment 
# 
# $$\displaystyle d_\pi=\sum_a(1-q_a)r_a$$
# 
# where $r_a$ is the coordinate of atom $a$. The dipole moment of a CH bond is 0.3 D.
# 
# **Strategy:** As this is a largish matrix and not block diagonal, use python/Sympy to perform the calculation. Each MO with index $i$ is the wavefunction $\Psi_i = c_{1i}\psi_1 + c_{2i}\psi_2 + c_{3i}\psi_1 + c_{4i}\psi_1$ where the $c$'s are the elements of the $i^{th}$ eigenvector; the second (column) index identifies the eigenvalue i, the first the atom.
# 
# The full spatial dependence of the orbitals would involve calculating the $\psi$ in three dimensions; instead, and just as effectively, the coefficients $c$ are used to represent the $\pi$ electron density on each atom which allows us to find the MO's pattern and hence the number of nodes. The node pattern can be used to determine the energy ordering; as a rule of thumb, the larger the number of nodes the higher the energy.
# 
# ## Q49 Fulvalene MO energies
# Repeat the calculation of the previous question, (Q48), for fulvalene; see Q8 for the numbering of the atoms. Confirm that the dipole is $-0.711eL$ or $0.48$ D where $e$ is the charge on the electron (in Coulombs) and $L$ is the bond length, which is $\approx 140$ pm. The experimentally measured dipole is $0.4$ D. Take atom 2 to be at the origin ($x=y=0$) Confirm also that the $\pi$ bond order between atoms is as shown below:
# 
# $$\displaystyle \begin{array}{c|cc} 
# \text{bond number} & \text{Bond order}\\
# \hline
# 1 \to 2 & 0.759\\
# 2 \to 3 & 0.499\\
# 3 \to 4 & 0.788\\
# 4 \to 5 & 0.520 \\
# \hline \end{array}$$
# 
# The remainder of the bond orders follow by symmetry.
# 
# ## Q50 Benzene MO energies
# Repeat the Huckel MO calculation for benzene; the matrix is worked out in Q7. Calculate the eigenvectors and plot out the MO coefficients.
# 
# (a) Is the pattern what you expect? You should find that some MOs are simply rotations of others. What distinguishes these?
# 
# (b) Make linear combinations of these MOs to form new MOs with the usual orbital shapes. Draw out the results you obtain.
# 
# **Strategy:** for the first part, use python/Sympy to obtain the eigenvectors.

# In[ ]:




