#!/usr/bin/env python
# coding: utf-8

# ## Questions 8 - 12

# ### Q8
# Using the bra-ket notation, work out the following matrix multiplications in the orthonormal basis of an s and two p orbitals $(|s\rangle,|p_x\rangle,|p_y\rangle)$, for the linear combination of wavefunctions
# 
# $$\displaystyle |\psi\rangle=c_s|s\rangle+c_x|px\rangle+c_y |p_y\rangle,\\ |\varphi\rangle=b_s|s\rangle+b_x|p_x\rangle+by |p_y\rangle$$
# 
# Assume that $\psi$ and $\varphi$ are normalized, which means for example that, $c^*_s c_s + c_x^*c_x + c_y^*c_y = |c_s |^2 + |c_x |^2 + |c_y |^2 = 1$ and similarly for coefficients $b$.
# 
# (a) Calculate $\langle|s\rangle$ and $\langle s|p_x\rangle$ and similarly for the other basis vectors and show that the basis is orthonormal.
# 
# (b) 
# 
# $$\displaystyle \begin{array}\\
# (i)\;|s\rangle\langle s|, & (ii)\; |s\rangle \langle p_x|,& (iii)\; |s\rangle \langle s|\varphi\rangle, &(iv)\; |s\rangle \langle p_x| \varphi\rangle, \\ (v)\; \langle\varphi|\psi\rangle, &(vi)\; \langle p_y| \psi\rangle, & (vii)\; |\varphi\rangle \langle\psi |, & (viii)\; |s\rangle \langle p_x| \varphi\rangle
# \end{array}$$
# 
# **Strategy:** Check that the matrices are commensurate for each calculation, the column of the left matrix (or vector) must have the same number or rows as the right matrix (or vector), and if so multiply them out. The next important point is to note that in vector form the standard basis sets are orthogonal and normalized, which means they are
# 
# $$\displaystyle |s\rangle =\begin{bmatrix} 1\\0\\0\end{bmatrix};\quad |p_x\rangle= \begin{bmatrix} 0\\1\\0\end{bmatrix};\quad |p_y\rangle =\begin{bmatrix} 0\\0\\1\end{bmatrix}$$
# 
# and the wavefunctions the kets;
# 
# $$ \displaystyle |\psi\rangle =\begin{bmatrix} c_s\\c_x\\c_y\end{bmatrix};\quad |\varphi\rangle =\begin{bmatrix} b_s\\b_x\\b_y\end{bmatrix}$$
# 
# ### Q9
# Using $\psi$ and $\varphi$ defined in question 8, show that $\langle \psi|\varphi\rangle = \langle\varphi|\psi\rangle^*$ is a general statement for any bra-ket pair.
#         
#         
# ### Q10
# In the NMR spectroscopy of a single $^1$H, with spin $1/2$ quantum number, a two element basis set can be defined as $(\alpha, \beta)$ to represent the spin state. The $\alpha$ spin state $(s = 1/2, m = 1/2)$ is represented as the column vector $\alpha =\begin{bmatrix} 1\\0\end{bmatrix}$ and as this is a basis element (basis ket) of the standard basis set it is zero except for one element which is unity. Similarly $\beta \equiv (s = 1/2, m = -1/2)$ is represented as the column vector $\beta =\begin{bmatrix} 0\\1\end{bmatrix}$.
# 
# The angular momentum operator $\pmb{I}$ has components in the $x-, y-, z$-directions and because it is an operator it is represented as a matrix. Angular momentum has units of $\hbar$ and the components of the angular momentum operators can be represented as the (Pauli) matrices,
# 
# $$\displaystyle \pmb{I}_x=\frac{\hbar}{2}\begin{bmatrix}0&1\\1& 0 \end{bmatrix}, \quad\pmb{I}_y=\frac{i\hbar}{2}\begin{bmatrix}0&-1\\1& 0 \end{bmatrix}, \quad\pmb{I}_z=\frac{\hbar}{2}\begin{bmatrix}1&0\\0& -1 \end{bmatrix}$$
# 
# (a)  Show that if $\pmb{I}^2 =\pmb{I}_x^2 +\pmb{I}_y^2 +\pmb{I}_z^2$ then 
# 
# $$\displaystyle \pmb{I}^2 =\frac{\hbar^2}{4}\begin{bmatrix}3&0\\0&3\end{bmatrix}=\frac{\hbar^2}{4}\pmb{I}$$
# 
# where $\pmb{I}$ is the unit diagonal matrix.
# 
# (b) Show that $\pmb{I}^2$ and $\pmb{I}_z$ commute, $[\pmb{I}^2,\pmb{I}_z] = 0$, but that $[\pmb{I}_x, \pmb{I}_y] = i\hbar \pmb{I}_z$ and the other combinations of $x, y, z$ components do not commute.
# 
# (c) Show that when $\pmb{I}^2$ operates on $\alpha$, the eigenvalue is $3\hbar^2/4$, i.e. show that $\pmb{I}^2\alpha =(3\hbar^2/4)\alpha$ and calculate $\pmb{I}^2\beta, \pmb{I}_Z\alpha$ and $\pmb{I}_z\beta$. Comment on the results. These equations are eigenvalue-eigenvector equations, as is the Schroedinger equation $H\psi = E\psi$ where $\psi$ is the wavefunction, $E$ the energy, and $H$ the operator.
# 
# (d)  Calculate $\pmb{I}_x\alpha, \pmb{I}_x \beta$ and similar terms for the $y$ and $z$ components.
# 
# (e) Using the operators $\pmb{I}^+ = \pmb{I}_x + i\pmb{I}_y$ and $\pmb{I}^- = \pmb{I}_x - i\pmb{I}_y$, show that these are raising or lowering operators which convert the eigenstate $\alpha$ or $\beta$ into the other.
# 
# ### Q11
# Angular momentum raising and lowering (shift) operators $\pmb{L}$ move a system from one state to another. They have the property
# 
# $$\displaystyle \pmb{L}^+|L,m\rangle=\sqrt{L(L+1)-m(m+1)} |L,m+1\rangle \\\pmb{L}^-|L,m\rangle=\sqrt{L(L+1)-m(m-1)} |L,m-1\rangle $$
# 
# where the angular momentum quantum number is $\pmb{L}$ and the projection or $z$ quantum number is $m$. The values $m$ takes run from $-L \to +L$ in unit steps.
# 
# (a)  Show that the $Lm$ basis set used is orthonormal.
# 
# (b) Calculate the raising and lowering operators for a state with angular momentum $3/2$ in the $L, m$ basis set. Note the representation of the operators will be a matrix.
# 
# (c) Using the result from (a) show that the commutator $[\pmb{L}^+, \pmb{L}^-] = 2\pmb{L}_z$ where $\pmb{L}$ is given by equation 21. (assume $\hbar\equiv 1$ and change $s\to L$).
# 
# (d) Calculate the $\pmb{L}^+$ operator for spins, $0, 1, 2, 3$ and then for half unit spins $1/2, 3/2, 5/2$ in the $L,m$ basis set.
# 
# **Strategy:** (b) The basis set must contain all $m$ values therefore the set could be 
# 
# $$(3/2, -3/2),\quad (3/2, -1/2),\quad (3/2, 1/2),\quad (3/2, 3/2)$$
# 
# if $L = 3/2$. Define vectors so that they are orthonormal. (c) The raising and lowering operators form a block diagonal matrix. Individual blocks of which can be calculated as in (a) or a more extensive basis set formed and the whole matrix calculated. It may be useful to define a python/Sympy  function to work out terms in the operator according to the $\pmb{L}^+, \pmb{L}^-$ equations.
# 
# 
# ### Q12
# (a) Using the raising and lowering operator in the previous question produce equations 19, 20 starting from 
# 
# $$\displaystyle \pmb{L} = (\pmb{L}^+ + \pmb{L}^-)\,/\,2\quad\text{and}\quad \pmb{L} = (\pmb{L}^+ - \pmb{L}^-)\,/\,2i$$
# 
# (Assume $\hbar \equiv 1$.)
# 
# (b) Show that the commutator is 
# 
# $$\displaystyle [\pmb{L}^+, \pmb{L}^-] = 2\pmb{L}_z$$
# 
# where $\pmb{L}$ is given by a similar equation to 21, $\pmb{L}_z|Lm\rangle=m|Lm\rangle$

# In[ ]:




