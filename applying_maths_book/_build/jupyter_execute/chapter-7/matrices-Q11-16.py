#!/usr/bin/env python
# coding: utf-8

# # Questions  11 - 16

# ## Q11
# Find $\pmb{AB}$ and $\pmb{BA}$ if $\displaystyle \pmb{A} =\begin{bmatrix} 1 & 2\\ 3& 4\end{bmatrix}$  and $\displaystyle \pmb{B} =\begin{bmatrix} 5 & 6\\ 7& 8\end{bmatrix}$ . Do these matrices commute and if not what is $[\pmb{AB}]$?
# 
# ## Q12
# (a) Find $\pmb{A}^2,\pmb{B}^2,\pmb{AB},\pmb{BA},\pmb{A}^2\pmb{B}$,and $\pmb{B}^2\pmb{A}$  if $\displaystyle \pmb{A} =\begin{bmatrix} 1 & 1\\ 0& 1\end{bmatrix}$ and $\displaystyle \pmb{B} =\begin{bmatrix} 1 & 0\\ 1& 1\end{bmatrix}$.
# 
# (b) Do $\pmb{A}^2\pmb{B}$ and $\pmb{B}^2\pmb{A}$ commute ?
# 
# ## Q13
# If $\pmb{x}=[x_1, x_2 ],\pmb{y}=[y_1,y_2]^T$ ,and $\displaystyle \pmb{Q}=\begin{bmatrix} 1 &-4\\ -9& 6\end{bmatrix}$, find $\pmb{xQy}$ and $\pmb{yQx}$.
# 
# ## Q14
# (a) Explain why if $j$ is a constant factor that $|j\pmb{M}| = j^n|\pmb{M}|$ for an $n \times n$ matrix.
# 
# (b)  Confirm equation 4 if $\displaystyle \pmb{A} =\begin{bmatrix} a & b\\ c& d\end{bmatrix}$  by calculating $\pmb{A}^{-1}$ and $| \pmb{A}^{-1} |$.
# 
# (c) Using Sympy (or by hand ) find $\pmb{A}^{-1}$ and $|\pmb{A}^{-1} |$ for the $3\times 3$ matrix $\displaystyle \pmb{A} =\begin{bmatrix} 0 & b & c\\ b& 0 & d\\ c & d & 0\end{bmatrix}$.
# 
# **Strategy:** See Section 4.14, but now the problem is algebraic not numerical.
# 
# ## Q15 Commuting operators
# (a) Show that if $\pmb{P}$ and $\pmb{Q}$ are linear operators, not necessarily matrices, $[\pmb{P},\pmb{Q}]=- [\pmb{Q},\pmb{P}]$.
# 
# (b) (i) find $[ d/dx, x]\sin(x)$ , (ii) $[d/dx, x]f(x)$, and (iii) $[d/dx, x]$ and
# 
# (c) Show that operators $d/dx$ and $x$ do not commute.
# 
# (d) Write a Python/Sympy commutator function and test your results.
# 
# (e) Do $\displaystyle \frac{df(x)}{dx}$ and $\displaystyle \int_a^bf(x)dx$ commute?
# 
# (f) Does the operator $df/dx$ and a displacement operator $\Delta f = f (x + c)$ commute?
# 
# (g) Does the operator $xx$ which means multiply twice by $x$, commute with the inversion operator, $Inv( f(x)) = f (-x)$?
# 
# ## Q16 Commuting matrices
# $\pmb{B}$ and $\pmb{C}$ are commuting square matrices and the matrix $\pmb{A}$ is defined as $\pmb{A} \equiv e^{\pmb{B}} =\pmb{1}+\pmb{B}+\pmb{B}^2/2! +\cdots $ show that:
# 
# (a) $e^{\pmb{B}}e^{\pmb{C}} = e^{\pmb{B+C}}$,
# 
# (b) $\pmb{A}^{-1} = e^{-\pmb{B}}$,
# 
# (c) $e^{\pmb{CBC}^{-1}} = \pmb{CAC}^{-1}$.
# 
# **Strategy:** (a) Expand out the exponentials as shown in Section 5.5 and collect terms and try to reform an exponential series in $\pmb{B + C}$. (b) Use the fact that for an inverse matrix $\pmb{AA}^{-1} = \pmb{1}$. (c) The expression $\pmb{CAC}^{-1}$ is a similarity transform and is itself a square matrix. To prove this result, expand the exponential on both sides of the equation

# In[ ]:




