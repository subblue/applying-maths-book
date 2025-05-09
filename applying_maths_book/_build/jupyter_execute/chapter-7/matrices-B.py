#!/usr/bin/env python
# coding: utf-8

# # 4 Matrices

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import numpy.linalg as La
import matplotlib.pyplot as plt
from sympy import *
init_printing()                      # allows printing of SymPy results in typeset maths format
plt.rcParams.update({'font.size': 14})  # set font size for plots


# 
# ## 4 Basic properties of Matrices
# The rest of this chapter describes the properties and uses of matrices. The determinant described in the previous sections can be considered as one of the many properties of a matrix. In several places, reference is made to eigenvalues and eigenvectors before they have been fully explained, which is done in Section 12.3. The meaning of these words is outlined below but how they are obtained is left unanswered for the moment.
# The eigenvalue - eigenvector equation is met, for example, in solving problems in Quantum Mechanics and Chemical Kinetics and has the form,
# 
# $$\displaystyle  \mathrm{operator \otimes function = constant \otimes same\; function}$$
# 
# provided the function is not zero. The operator $\otimes$ means matrix multiplication but we never need use it as we will always know when vectors and or matrices are being multiplied together, not only from the context but also because these are written as bold characters. In matrix form, the eigenvector - eigenvalue equation is
# 
# $$\displaystyle \pmb{A x} = \lambda \pmb{x}$$
# 
# where $\pmb A$ is an $n \times n$ square matrix, $\pmb x$ is one of possibly $n$, one-dimensional column matrices (column vectors) each of length $n$ and each one is called an *eigenvector*, $\lambda$  represents one of $n$ numbers, and each $\lambda$ is called an *eigenvalue*. What this equation shows us is that the operator (matrix $\pmb A$) changes a vector $\pmb x$ by stretching or contracting it by an amount $\lambda$ but leaves it pointing in the same direction.
# 
# ### **(i) Addition and subtraction**
# 
# To add or subtract two matrices they must have the same shape. To add or subtract a constant or another matrix their individual elements are added or subtracted. Similarly, to multiply or divide by a constant number (a scalar), each element is also multiplied or divided by this value. Multiplying two matrices together is more complicated and depends on the order with which this is done and the shapes of the two matrices; sometimes it is just not possible.
# 
# The notation used, is that bold italic letters indicate a matrix and the numbers $\pmb 0$ and $\pmb 1$ for the null and unit matrix respectively.
# 
# Suppose two matrices are $\displaystyle \pmb{A}=\begin{bmatrix} 2&1&5\\-5 & 2 & 6\end{bmatrix}$ and $\displaystyle \pmb{B}=\begin{bmatrix} 1&10&8\\-3 & 3 & 9\end{bmatrix}$ calculating 
# 
# $$\displaystyle \pmb{A}+\pmb{B},\; \pmb{A}-\pmb{B},\;\pmb{3A}+\pmb{0.5B}\quad\text{etc.}$$
# 
# is easy, for example, taking elements individually,
# 
# $$\displaystyle \pmb{A}+\pmb{B}=\begin{bmatrix} 3&11&13\\-8 & 5 & 16\end{bmatrix}$$
# 
# and similarly for the other calculations.
# 
# ### **(ii) Division**
# One matrix _cannot_ be divided by another; the inverse matrix of the divisor is formed instead and then these matrices are multiplied together. The inverse of a matrix $\pmb M$ is always written as $\pmb{M}^{-1}$. Generating the inverse of a matrix is difficult unless the matrix is small, and one would normally use Python/Sympy to do this.
# 
# ### **(iii) Multiplication**
# Matrices can be multiplied together. The multiplication order is always important; $\pmb{AB}$ is not necessarily the same as $\pmb{BA}$. With three or more matrices $\pmb{ABC}$, the multiplication sequence does not matter as long as the ordering is the same. This is the associatite property
# 
# $$\displaystyle \pmb{ABC}\equiv \pmb{A}(\pmb{BC})\equiv (\pmb{AB})\pmb{C}$$
# 
# If $\pmb{BC}$ are multiplied first, then the result left multiplied by $\pmb{A}$, this is the same as multiplying $\pmb{AB}$ to form a new matrix, $\pmb{D}$ for example, then performing the multiplication $\pmb{DC}$. Habitually, we start at the right and multiply the $\pmb{BC}$ pair first then left multiply by $\pmb{A}$. The details are described in Section 5.
# 
# There are a few special matrices and several types operations that can be performed on matrices and these are describe next.
# 
# ## 4.1 The zero or null matrix
# As you would expect it is possible to have a matrix of zeros; adding this to another matrix changes nothing, but multiplying by it annihilates the other matrix leaving only the zero matrix as you would expect.
# 
# $$\displaystyle \pmb{0}=\begin{bmatrix} 0 & 0 & \cdots \\0 & 0 \cdots \\ \vdots & \vdots & \ddots \end{bmatrix} \qquad \pmb{A}+\pmb{0} =\pmb{A};\; \pmb{A0}=\pmb{0};\; \pmb{0A}=\pmb{0}$$
# 
# ## 4.2 The unit or identity matrix  $\pmb{1}$.
# 
# This is a matrix of unit diagonals with all other elements being zero. The unit matrix is also called the _multiplicative identity_.
# 
# $$\displaystyle \pmb{1}=\begin{bmatrix} 1 & 0 & 0& \cdots \\0 & 1 & 0 &\cdots \\ 0 & 0 & 1 & 0\\ \vdots  & & &\ddots\end{bmatrix}\qquad \pmb{A1}=\pmb{A};\; \pmb{1A}=\pmb{A}$$ 
# 
# ## 4.3 Diagonal matrix
# 
# This is a matrix whose diagonals can take any value, but all other elements are zero. The unit matrix is a special case. In general
# 
# $$\displaystyle \begin{bmatrix} a & 0 & 0& \cdots \\0 & b & 0 &\cdots \\ 0 & 0 & c & 0\\ \vdots  & & &\ddots\end{bmatrix}$$
# 
# ## 4.4 Trace or character of a matrix
# The trace, spur, or character of any matrix $\pmb M$, is the sum of its diagonal elements only, 
# 
# $$\displaystyle Tr(\pmb{M})=\sum_{i=0}a_{i,i}$$
# 
# for example 
# 
# $$\displaystyle \pmb{M}= \begin{bmatrix} 1 & 0 & 2 \\ 3 & -1 & 0  \\ 0 & 4 & -1 \end{bmatrix} \qquad Tr(\pmb{M})= -1$$
# 
# By multiplying out the product of matrices, the trace of the product is found to be unchanged if they are permuted cyclically:
# 
# $$Tr(\pmb{ABC}) = Tr(\pmb{CAB}) = Tr(\pmb{BCA})$$
# 
# This fact is very useful in group theory because the trace of a matrix forms a _representation_ of a molecule's symmetry. Because of this rule, the trace, and hence representation, is independent of the basis set used to describe the symmetry.
# 
# The trace of a square matrix is the sum of its eigenvalues $\lambda$, i.e.
# 
# $$\displaystyle Tr(\pmb M) = \sum_i\lambda_i$$
# 
# and the determinant of such a matrix is the product of eigenvalues 
# 
# $$\displaystyle |\pmb M|= \prod_i \lambda_i$$
# 
# ## 4.5 Matrix transpose $\pmb{M}^T$
# 
# The transpose operation replaces rows and columns with one another, and so  exchanges element $(1, 2)$ with $(2, 1)$, and $(1, 3)$ with $(3, 1)$ and so on. For example;
# 
# $$\displaystyle\begin{bmatrix} a & b & c \\ d & e & f   \end{bmatrix}^T= \begin{bmatrix} a & d \\b & e\\ c & f \end{bmatrix}$$ 
# 
# and for a square matrix this operation leaves the diagonal unchanged, and is, in effect, a reflection about the diagonal,
# 
# $$\displaystyle\begin{bmatrix} a & b & c \\ d & e & f \\g & h & i  \end{bmatrix}^{\,T}= \begin{bmatrix} a & d & g\\b & e & h\\ c & f & i\end{bmatrix}$$ 
# 
# You can appreciate that two transposes reproduce the initial matrix $(\pmb{M}^T)^T = \pmb{M}$.
# 
# A _symmetric_ matrix is equal to its transpose and therefore must also be square and have off-diagonal elements $a_{ij} = a_{ji}$. An _antisymmetric_ matrix satisfies the identity $\pmb{A} = -\pmb{A}^T$ and must therefore have zeros on its diagonal and have components $a_{ij} = -a_{ji}$; for example, such a matrix and its transpose is
# 
# $$\displaystyle\begin{bmatrix} 0 & 2 & 3 \\ -2 & 0 & 4 \\-3 & -4 & 0  \end{bmatrix}^{\,T}= \begin{bmatrix} 0 & -2 & 3\\2 & 0 & -4\\ 3 & 4 & 0\end{bmatrix}=-\begin{bmatrix} 0 & 2 & 3 \\ -2 & 0 & 4 \\-3 & -4 & 0  \end{bmatrix}$$
# 
# When two matrices are multiplied, then the transpose reorders the matrix multiplication because rows are converted into columns and vice versa. This means that
# 
# $$\displaystyle (\pmb{AB})^T=\pmb{B}^T\pmb{A}^T$$
# 
# Matrix multiplication is described in section 5 and if for example if $\displaystyle \pmb{A}=\begin{bmatrix} a & b  \\ c & d \end{bmatrix},\; \pmb{B}=\begin{bmatrix} 1 & 2  \\ 3 & 4 \end{bmatrix}$, then 
# 
# $$\displaystyle \pmb{AB}^T = \begin{bmatrix} a +3b & 2a+4b  \\ c+3d & 2c+4d \end{bmatrix}^T=\begin{bmatrix} a +3b & c+3d  \\ 2a+4b & 2c+4d \end{bmatrix}$$
# 
# Performing the transpose then multiplying in reverse order gives the same result.
# 
# $$\displaystyle \pmb{b}^T\pmb{A}^T = \begin{bmatrix} 1 & 3  \\ 2& 4 \end{bmatrix}\begin{bmatrix} a & c  \\ b& d \end{bmatrix}=\begin{bmatrix} a +3b & c+3d  \\ 2a+4b & 2c+4d \end{bmatrix}$$

# ## 4.6 Complex conjugate of a matrix M*
# This changes each matrix element with its complex conjugate, assuming there are complex numbers in the matrix; if not, it has no effect. 
# 
# The conjugate of a matrix $\pmb{M}$ is labelled $\pmb{M}^*$. In making a complex conjugate each $i$ is replaced by $-i$ where $i =\sqrt{-1}$, for example $[3\; i\;4]^* =[3\;-i \;4]$.
# Some properties are
# 
# $$\displaystyle \begin{align}&(\pmb{M}^*)^* = \pmb{M}, \\&(\pmb{AB})^* = \pmb{A}^*\pmb{B}^*, \\ &|\pmb{M}^*|=|\pmb{M}|^*\end{align}$$
# 
# where the last property describes the effect on the determinant of the matrix.
# 
# ## 4.7 Adjoint matrix, $\pmb{M}^\dagger$
# 
# This grand sounding name simply means
# 
# $\qquad\qquad$_'Form the complex conjugate, then transpose the matrix or vice versa'_
# 
# The special symbol $\dagger$ is conventionally used as a superscript, e.g.  
# 
# $$\displaystyle  \pmb M=\begin{bmatrix} 1&0&0\\0&0&w\\0&w^2&0 \end{bmatrix},\qquad \pmb M^\dagger=\begin{bmatrix} 1&0&0\\0&0&w^{*2}\\0&w^*&0 \end{bmatrix}$$
# 
# where $w$ is complex. The effect of the adjoint operation is 
# 
# $$\displaystyle \pmb{M}^\dagger = (\pmb{M}^*)^T=(\pmb{M}^T)^*$$
# 
# Other properties are, where $A$ and $B$ are matrices and $\lambda$ a (complex) number,
# 
# $$\displaystyle (\pmb{A}^\dagger)^\dagger=\pmb{A},\qquad (\pmb{A}^\dagger)^{-1}=(\pmb{A}^{-1})^\dagger, \quad (\lambda\pmb{A})^\dagger=\lambda^*\pmb{A}^\dagger$$
# 
# $$\displaystyle (\pmb{A}+\pmb{B})^\dagger=\pmb{A}^\dagger+\pmb{B}^\dagger,\qquad  (\pmb{A}\pmb{B})^\dagger=\pmb{B}^\dagger\pmb{A}^\dagger,\quad |\pmb A^\dagger|=|\pmb A|^*$$
# 
# In quantum mechanics, the adjoint is always used to convert a 'ket' into a 'bra' and vice versa, see Chapter 8.
# 
# (Warning: In pure mathematics, in particular, the adjoint is defined differently as the transposed matrix of cofactors and call our adjoint the conjugate transpose or the Hermitian adjoint.)
# 
# ## 4.8 Hermitian or self-adjoint matrix $\pmb{M}^\dagger=\pmb{M}$
# In these matrices the diagonals are real and the off-diagonals $a_{i,j} = a_{j,i}^*$. The Pauli spin matrices of quantum mechanics are Hermitian; for example
# 
# $$\displaystyle \pmb{\sigma}_2=\begin{bmatrix} 0 & i \\ -i & 0\end{bmatrix}$$
# 
# Taking the transpose then the complex conjugate produces a matrix that is clearly the same as $\pmb{\sigma}_2$. 
# 
# $$\displaystyle \begin{bmatrix} 0 & i \\ -i & 0\end{bmatrix}^T=\begin{bmatrix} 0 & -i \\ i & 0\end{bmatrix}\quad\text{then}\quad
# \begin{bmatrix} 0 & -i \\ i & 0\end{bmatrix}^*=\begin{bmatrix} 0 & i \\ -i & 0\end{bmatrix}$$
# 
# If the Hermitian matrix is real, which means that it contains only real numbers, then it must be symmetric. 
# 
# In quantum mechanics, a matrix containing the integrals that evaluate to expectation values is always a real Hermitian matrix whose eigenvalues are real and eigenvectors orthogonal, but not necessarily normalized.
# 
# An _anti-Hermitian_ matrix is defined as $\pmb{M}^\dagger = -\pmb{M}$.
# 
# ## 4.9 Inverse of a square matrix, $\pmb{M}^{-1}$
# 
# One matrix cannot be divided into another or into a constant so the operation $1/\pmb{M}$ is not allowed; instead the matrix inverse must be formed, this is labelled  $\pmb{M}^{-1}$ where the $-1$ superscript indicates the operations involved in forming the inverse. Furthermore the conventional inverse is defined only for a square matrix and then only if its determinant is not zero, $|\pmb{M}| \ne 0$. If the columns or rows of a determinant are the same this could be a sign that an error has occurred somewhere else in the calculation. However, if the determinant is zero a pseudo-inverse can be made via singular value decomposition (SVD), which avoids dividing by the determinant, and is called the Moore-Penrose inverse.
# 
# The inverse is another matrix labelled $\pmb{M}^{-1}$, such that
# 
# $$\displaystyle \pmb{MM}^{-1} =\pmb{1}=\pmb{M}^{-1}\pmb{M}$$
# 
# where $\pmb{1}$  is the unit (diagonal) matrix. Formally, we can find the inverse matrix element by element with the $ij^{th}$ element given by
# 
# $$\displaystyle (\pmb{M}^{-1})_{i,j}=\frac{(-1)^{i+j}}{|\pmb{M}|}|\pmb{C}_{j,i}|$$
# 
# where $| \pmb{M} | \ne 0$ is the determinant and $| \pmb{C}_{j,i} |$ is the cofactor matrix of row $j$ and column $i$. Note the change in ordering of the indices. Cofactors were described in Section 2.2. Generally, it is best to use Python/NymPy to calculate the inverse, because, besides being tedious, the chance of making an error is very high. The matrix inverse is met again when solving equations.
# 
# Using Python/ NumPy to invert a matrix is shown next. The NumPy linear algebra module has to be imported first as done at the top of this page.

# In[2]:


M    = np.array([ [2,2,3],[-1,0,4],[6,5,4] ])   # make matrix
invM = La.inv(M)
invM             # inverse


# Now check that this is the inverse $\pmb{MM}^{-1} =\pmb{1}$.  The ampersand @ is the matrix multiplier operator.

# In[3]:


M @ invM


# which shows this is the inverse to numerical accuracy.
# 
# ### **(i) Inverse of a diagonal matrix**
# If the matrix is diagonal then its inverse comprises the reciprocal of each term.
# 
# $$\displaystyle\begin{bmatrix} a & 0 & 0 \\ 0 & b & 0 \\0 & 0 & c  \end{bmatrix}^{-1}= \begin{bmatrix} 1/a & 0 & 0\\0 & 1/b & 0\\ 0 & 0 & 1/c\end{bmatrix}$$
# 
# and this is used in the calculation of vibrational normal modes.
# 
# ### **(ii) Inverse of the rotation matrix**
# The rotation matrix is often needed when using molecular point groups
# 
# $$\displaystyle \pmb M_R=\begin{bmatrix} \cos(\theta)&\sin(\theta) &0\\ -\sin(\theta)&\cos(\theta) &0\\0&0&1\end{bmatrix}$$
# 
# has the inverse
# 
# $$\displaystyle \pmb M_R^{-1}=\begin{bmatrix} \cos(\theta)&-\sin(\theta) &0\\ \sin(\theta)&\cos(\theta) &0\\0&0&1\end{bmatrix}$$
# 
# as can be seen by showing that $\pmb M_R\pmb M_R^{-1}=\pmb 1$.
# 
# ## 4.10 Singular matrix, determinant $|\pmb{M}|=0$
# 
# The determinant is zero and the matrix has no inverse. This is not the same as the null matrix, $\pmb{0}$. Odd sized $n \times n$, anti-symmetric matrices are singular.
# 
# 
# ## 4.11 Unitary matrix $\pmb M^\dagger = \pmb M^{-1}$ or $\pmb {M}^\dagger \pmb{M} = \pmb{1}$ or $\pmb {M} \pmb{M}^\dagger = \pmb{1}$ and $ |\pmb {M} | = 1$
# 
# A matrix is unitary if its adjoint is equal to its inverse and all unitary matrices must be square.
# 
# An example of a unitary matrix is  $\displaystyle \quad\begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1\\1 & 0 & 0 \end{bmatrix}$,
# 
# which has the determinant $0(0 - 1) - 1(0 - 1) + 0(0 - 0) = 1$, 
# 
# The rotation matrix is also unitary.
# 
# $$\displaystyle \begin{bmatrix} \cos(\theta) & \sin(\theta)  \\ -\sin(\theta) & \cos(\theta)  \end{bmatrix} \qquad\qquad\text{(3)}$$
# 
# whose determinant is unity because $\sin^2(\theta)+\cos^2(\theta)=1$.  
# 
# ## 4.12 Orthogonal matrices
# 
# Orthogonal matrices are always square and have the properties that if $\pmb M$ is a matrix then
# 
# $$ \displaystyle \pmb{M}^T = \pmb{M}^{-1} \text{ or equivalently}\quad \pmb{M}^T\pmb{M} = \pmb 1$$
# 
# which means that the matrix transpose is its inverse and when the matrix and its transpose are multiplied together a unit diagonal matrix results. The orthogonal matrix has a determinant that is $\pm 1$; the eigenvalues are all $+1$. However, a determinant of $\pm 1$ does not mean that a matrix is orthogonal. The product of two orthogonal matrices is also an orthogonal matrix.
# 
# The rotation matrix, equation 3, is orthogonal, as is the matrix $\displaystyle \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1  \\ 1 & -1  \end{bmatrix}$ which shows that a real orthogonal matrix is the same as a unitary matrix.
# 
# In an orthogonal matrix, rows and columns form an _orthonormal basis_ and each row has a length of one. When an orthogonal matrix operates on (linearly transforms) a vector or a matrix representing an object, such as a molecule, the molecule is unchanged, meaning that relative bond angles and lengths remain the same but the coordinate axes moved so the object appears to have been rotated when viewed on the computer screen. Orthogonal matrices are used to represent reflections, rotations, and inversions in molecular group theory. Matrix decomposition methods make use of orthogonal matrices. One matrix method called _singular value decomposition_ is used in data analysis to separate out overlapping spectra into their constituent parts.
# 
# ## 4.13 Determinants
# 
# Evaluating determinants was described in Section 2. The determinants of different types of square matrices have the following properties that do not involve evaluating the determinant.
# 
# $$\displaystyle \qquad\qquad\begin{align} \big|\,\pmb{ABC}\,\big|&=\big|\,\pmb{A}\,\big|\big|\,\pmb{B}\,\big|\big|\,\pmb{C}\,\big|\\
# \\ \big|\pmb{A}\big|^T &= \big|\pmb{A}\big| \\
# \big|\pmb{A} \big|^{-1} &= \frac{1}{ \big|\pmb{A}\big| }\\
# \big|a\pmb{A}\big| &= a^n\big|\pmb{A}\big|\end{align}\qquad\qquad\qquad\qquad\qquad\qquad\text{(4)}$$ 
# 
# where $n$ is the length of the determinant's row or column.

# ## 5 Matrix multiplication
# 
# The _shape_ of the matrices and their multiplication _order_ must both be respected. If multiplication is possible, and it not always is, then, in general, multiplication $\pmb{AB}$ does not give the same result as $\pmb{BA}$. The matrices
# 
# $$\displaystyle \pmb{A}=\begin{bmatrix} 2 & 1 & 5 \\ -5 & 6 & 2 \end{bmatrix} \; \text{  and  } \pmb{B}=\begin{bmatrix} 12 & 11 & 6 \\ -3 & 3 & 9 \end{bmatrix} $$
# 
# cannot be multiplied together, because multiplication is only possible if the number of columns of the left-hand matrix is equal to the number of rows of the right-hand one. The matrices are then _commensurate_ or _conformable_. The result is a matrix whose size is determined by the _number of rows_ of the left hand matrix and the _number of columns_ of the right hand matrix, e.g.
# 
# $$\displaystyle \pmb{A} \qquad \pmb{B}\qquad \pmb{C}\\
# (n\times m)(m\times r) \to (n\times r)\\ 
#  \text{same number of columns in A as rows in B}$$
# 
# The calculation below shows the multiplication $\pmb{C} = \pmb{AB}$; the arrows show how the top element in $\pmb{C}$ is calculated as the product of element 1 of row 1 with element 1 of column 1. To this value is added the product of element 2 of row 1 and element 2 of column 1 and so on for all the elements in a row. This is why the number of rows and columns must be the same. You can also envisage the multiplication as the dot product of the each row with each column in turn and this is shown diagrammatically below.
# 
# ![Drawing](matrix-pics2.png) 
# 
# Multiplying two $2 \times 2$ matrices produces
# 
# $$\displaystyle \begin{align} \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \times \begin{bmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \end{bmatrix} 
# &=\begin{bmatrix}row(1)\cdot col(1) & row(1)\cdot col(2)\\ row(2)\cdot col(1)& row(2)\cdot col(2)\end{bmatrix}\\\\
# &=\begin{bmatrix} a_{11}b_{11}+a_{12}b_{21} & a_{11}b_{12}+a_{12}b_{22} \\ a_{21}b_{11}+a_{22}b_{21} & a_{21}b_{12}+a_{22}b_{22} \end{bmatrix}\end{align}$$
# 
# and it is useful to look at the pattern of elements where you can see the $\pmb{A}$ matrix in the pattern of $a$'s and columns in $\pmb{B}$ appear as rows in the sums. If the matrices are not square, multiplying a $n \times  m$ matrix by a $m \times r$ one results in a $n \times r$ matrix as shown above. The equation with which to calculate the (row - column) $ij^{th}$ element of any matrix multiplication is
# 
# $$\displaystyle C_{ij}=\sum_{k=1}^m a_{ik}b_{kj} \qquad\tag{5}$$
# 
# where the sum index $k$ runs from $1 \to m$, the number of columns in matrix $\pmb{A}$ or rows in $\pmb{B}$ and $i$ takes values $1 \to n$ and j from $1 \to r$. Multiplying a $2 \times 3$  matrix with a $3 \times 2$ one, the number of columns in the left matrix is the same as the number of rows in the right-hand one and therefore the result is a $2 \times 2$ square matrix,
# 
# $$\displaystyle\begin{align}  & 
# \begin{bmatrix} a_{11} & a_{12} &a_{13} \\ a_{21} & a_{22} &a_{23} \end{bmatrix} \times \begin{bmatrix} b_{11} & b_{12} \\ b_{21} & b_{22}\\ b_{31} & b_{32} \end{bmatrix} =\begin{bmatrix} P & Q \\ R & S \end{bmatrix}\qquad\qquad\text{(6)} \\
# &\quad\quad \uparrow \qquad \qquad\qquad\quad \uparrow \qquad\qquad \uparrow \\ 
# &\text{ 3 cols 2 rows } \qquad  \text{2 cols 3 rows  }\qquad  \text{2 by 2 matrix} \end{align} $$
# 
# where element $R$ is the dot product  
# 
# $$\displaystyle \begin{bmatrix} a_{21}& a_{22}&a_{23}\end{bmatrix}\times  \begin{bmatrix} b_{11}\\ b_{21}\\b_{31}\end{bmatrix}=a_{21}b_{11} + a_{22}b_{21} + a_{23}b_{31}$$
# 
# and element $S= a_{21}b_{12} + a_{22}b_{22} + a_{23}b_{32}$. 
# 
# A $2 \times 3$ and a $3 \times 1$ matrix produce a $2 \times 1$ column matrix
# 
# $$\displaystyle \qquad\qquad\begin{bmatrix} a_{11} & a_{12} &a_{13} \\ a_{21} & a_{22} &a_{23} \end{bmatrix} \times \begin{bmatrix} b_{1} \\ b_{2}\\b_3  \end{bmatrix} =\begin{bmatrix} P \\ R \end{bmatrix} \qquad\qquad \qquad\qquad\text{(7)}$$
# 
# where $P = a_{11}b_1 + a_{12}b_2 + a_{13}b_3$ and $R = a_{21}b_1 + a_{22}b_2 + a_{23}b_3$.
# 
# Suppose there are three matrices $\pmb{ABC}$, then the safest rule to follow is to left-multiply $\pmb{C}$ by
# $\pmb{B}$ first, then to left-multiply the result by $\pmb{A}$. The same rule is applied to several matrices; start at the right and work to the left. However, by the associative rule, Section 4.1, as long as the order $\pmb{ABCD}$ is maintained, this product can be multiplied in any order.
# 
# Should you have a row vector left multiplying a matrix and prefer to right multiply a matrix by a column vector then both vector and matrix must be transposed, and their order reversed i.e.
# 
# $$\displaystyle \begin{bmatrix}b_1 & b_2 & b_3 \end{bmatrix}\begin{bmatrix} a_{11} & a_{12} \\a_{21} & a_{22} \\ a_{31} &a_{32} \end{bmatrix} = \begin{bmatrix} a_{11} & a_{21} &a_{31} \\ a_{12} & a_{22} &a_{32} \end{bmatrix}  \begin{bmatrix} b_{1} \\ b_{2}\\b_3  \end{bmatrix} $$
# 
# 
# ### **Schematic of matrix multiplication by dot products**
# If you are familiar with taking the dot product of two vectors then the simplest way to do a matrix multiplication by hand can follow the scheme shown below. Each row and column chosen becomes, in effect, a vector making it easy, knowing the row and column indices, where the resultant dot product, which is a number, has to go. You can also immediately appreciate *via* the dot product why the number of columns in the left-hand matrix has to be equal to the number of rows in the right-hand one. This method can not be used when right multiplying a column by a row as in the outer product. 
# 
# ![Drawing](matrix-pics3.png)
# ______________________
# 
# ### **Outer product vector multiplication**
# In the case of the outer product the calculation is
# 
# $$\displaystyle \begin{bmatrix} b_1\\b_2\\b_3\\\vdots\end{bmatrix}\begin{bmatrix} a_1 & a_2 & a_3 \cdots\end{bmatrix}=\begin{bmatrix}b_1a_1 & b_1a_2 & b_1a_3\cdots \\ b_2a_1 & b_2a_2 &\cdots \\ \vdots & \vdots& \ddots\end{bmatrix} $$
# 
# ### **Schematics of matrix multiplication**
# The diagrams in Fig. 7 (below) show, diagrammatically, the result of multiplying differently shaped matrices. Only these multiplications are defines. The 'bra-ket' notation is shown also. 
# 
# ![Drawing](matrices-fig7a.png) 
# _____
# ![Drawing](matrices-fig7b.png)
# _____
# ![Drawing](matrices-fig7c.png) 
# 
# ![Drawing](matrices-fig7d.png)
# Figure 7. Pictorial representation of allowed matrix multiplication. Only these types of multiplications are possible. The lowest diagram shows the general case. The bra-ket notation used in quantum mechanics is shown also.  The dot product is also called the *inner product*.
# _______
# 
# ## 5.1 Matrix sum
# 
# To sum the terms in a square matrix left-multiply by a unit row vector and right-multiply
# by a unit column vector. For example
# 
# $$\displaystyle \begin{bmatrix} 1 & 1 \end{bmatrix}  \begin{bmatrix}a & b \\ c & b  \end{bmatrix}\begin{bmatrix} 1 \\ 1 \end{bmatrix} =a+b+c+d$$
# 
# ## 5.2 bra-ket notation
# 
# Often the bra-ket notation is used in quantum mechanics. The _ket_ $|k\rangle$  is a single column matrix (a column vector) $\displaystyle | k\rangle =\begin{bmatrix} a \\ b \\ \vdots \end{bmatrix}$ and the _bra_  is the single row matrix (or vector) and is always the complex conjugate of the ket. $\displaystyle \langle j|=\begin{bmatrix} a^* & b^* & \cdots \end{bmatrix}$. The $\langle j |k\rangle$ is a scalar number, and $|k\rangle \langle j|$ a square matrix.
# 
# These objects are discussed in more detail in chapters 6 and are mentioned here as they provide a shorthand way of visualizing matrix multiplication; see Figure 7. There is no specific symbol for a scalar or a square matrix; they have to be represented by the bra-ket pair.
# 
# ## 5.3 Commutators
# If matrices are not square then there is only one way to multiply them: the left-hand matrix's number of columns must equal the right-hand matrix's number of rows. When both matrices are square, to be multiplied they must be of the same size and we can then choose which is to be the left-hand and which the right-hand side of the multiplication, and now the _order_ of multiplication does matter as illustrated on the previous page. Sometimes the result is the same, i.e. $\pmb{AB} = \pmb{BA}$ then the matrices are said to *commute*, but generally they do not and, therefore, $\pmb{AB} \ne \pmb{BA}$.
# 
# The commutator of two square matrices $\pmb{A}$ and $\pmb{B}$ is also a matrix and is defined as
# 
# $$\displaystyle \begin{bmatrix} \pmb{A},\pmb{B} \end {bmatrix} = \pmb{AB}-\pmb{BA} \qquad\tag{8}$$
# 
# If the result is the null matrix $\pmb{0}$, which is full of zeros, $\pmb{A}$ and $\pmb{B}$ are said to *commute*: 
# 
# $$\displaystyle [ \pmb{A},\pmb{B} ] = \pmb{AB}-\pmb{BA}=\pmb{0}$$ 
# 
# Sometimes expressions such as $[\pmb{C},[\pmb{A,B}]]$ may be met. This means: work out the inside commutation first, then the commutation with each term that this produces: 
# 
# $$\displaystyle [\pmb{C},[\pmb{A},\pmb{B}]] = [\pmb{C}, \pmb{AB} - \pmb{BA}] = [\pmb{C}, \pmb{AB}] - [\pmb{C}, -\pmb{BA}]$$
# 
# Commutation is very important in quantum mechanics; only observables that commute can be observed simultaneously. Position and momentum, which do not commute, or those components of angular momentum, which also do not commute, have values that are restricted by an amount determined by the Heisenberg Uncertainty Principle when being observed simultaneously.
# 
# Commutation is common of operators in general. The commutator relationship applies to any two operators $\pmb{P}$ and $\pmb{Q}$, which need not be matrices, but could be the differential operator such as $d/dx$ or $x$ itself. This parallel means that we can consider matrices as operators. The commutator will in general operate on a function, for example,
# 
# $$\displaystyle [\pmb{P},\pmb{Q}]f= \pmb{PQ}f - \pmb{QP}f$$
# 
# and $f$ can be any normal function, $\ln(x),\, \sin(x)$, and so on. In molecular group theory, the five types of operators that are the identity, rotation, reflection, inversion, and improper rotation (combined rotation and reflection) sometimes commute with one another although this depends on the point group. Section 6 describes these operations.
# 
# 
# ## 5.4 Integral powers of square matrices
# 
# Only integral powers of matrices are defined and are calculated by repeated multiplication, for example,
# 
# $$\displaystyle \pmb{M}^0=\pmb{1}, \quad \pmb{M}^1=\pmb{M}, \quad \pmb{M}^2=\pmb{MM}, \quad \pmb{M}^3=\pmb{MMM} $$
# 
# and so on, for example   
# 
# $$\displaystyle \begin{bmatrix} 1 & 2  \\ 3 & 4 \end{bmatrix}^2=\begin{bmatrix} 1 & 2  \\ 3 & 4 \end{bmatrix}\begin{bmatrix} 1 & 2  \\ 3 & 4 \end{bmatrix}=\begin{bmatrix} 7 & 10  \\ 15 & 22 \end{bmatrix}$$
# 
# The *similarity transform*, see Section 13.4, can be used to obtain high integer powers of matrices, however, if the matrix is diagonal then taking the power is easy because the diagonal elements are raised to the power. The power need not be positive; therefore, the inverse of a diagonal matrix is easily obtained.
# 
# $$\displaystyle \begin{bmatrix} a & 0 & 0 & \cdots  \\ 0 &b & 0 & \cdots   \\ 0& \vdots &\ddots\end{bmatrix}^{\,n}= \begin{bmatrix} a^n & 0 & 0 & \cdots  \\ 0 &b^n & 0 & \cdots   \\0& \vdots & \ddots\end{bmatrix}$$
# 
# ## 5.5 Functions of matrices
# 
# In the study of the theory of nuclear magnetic resonance, NMR, a quantum mechanical description of nuclear spin is essential. To explore the rotation of the magnetization as used in inversion recovery, spin-echo, or a complicated two-dimensional experiment, such as COSY, requires the exponentiation of matrices (Levitt 2001). Exponentiation can only be performed by expanding the exponential and evaluating the terms in the series one by one as
# 
# $$\displaystyle e^{\pmb{M}}=\pmb{1} + \pmb{M}+ \frac{\pmb{M}^2}{2!}+\cdots; \quad e^{-\pmb{M}}=\pmb{1} - \pmb{M}+ \frac{\pmb{M}^2}{2!}-\cdots$$
# 
# and is $x$ is a variable $\displaystyle e^{-x\pmb{M}}=\pmb{1} -x\pmb{M}+ x^2\frac{\pmb{M}^2}{2!}-\cdots$
# 
# The matrix $\pmb{1}$ is a unit diagonal matrix. Similar expressions are formed with trig and log functions according to their expansion formula and with Taylor or Maclaurin series for other functions. The familiar relationship $e^Ae^B = e^{A+B}$ is only true if the matrices $A$ and $B$ commute. In Section 13.2 and 13.3 a transformation is describe which enables the exponential of a matrix to be found.
# 
# 
# ## 5.6 Block diagonal matrices
# 
# In many instances, a matrix can be blocked into smaller ones symmetrically disposed along the diagonal. The result of this is that the problem reduces to the lesser one of solving several matrices where each is much smaller than the whole and is therefore more easily solved. Why should we bother with this if the computer can diagonalize any matrix we give it to do? The reason is that eigenvalues can more easily be identified within the basis set by doing the calculation this way. Recall that the basis set you choose to use, for example in a quantum problem, determines the ordering of elements in a matrix. Why does this matter? It matters because when the spectrum from a molecule is observed, which measures only the difference in energy levels, we would like to know what quantum numbers give rise to what spectral lines. If the matrix is a block diagonal one, then this is made somewhat easier because we know what parts of the basis set elements are involved because each block when diagonalized contains only that part of the basis set that was in it in the first place. If the whole matrix is diagonalized blind, as it were, and without thinking about the problem beforehand, this information can be lost because all the elements and hence eigenvalues can be mixed up. The elements in the basis set can be ordered in any way you want, and different basis sets can be chosen for the same problem. By trying different ordering, it is sometimes possible to discover a block diagonal form for a matrix and so aid its solution. In the study of group theory, blocking matrices proves to be a powerful way of determining the irreducible representation; see Section 6. 
# 
# The following matrix has a $2 \times 2$, a $3 \times 3$, and a $1 \times 1$ block.
# 
# ![Drawing](matrices-blockmatrix.png) 
# _________
# 
# ## 5.7 The special case of the  2 x 2  matrix
# 
#  $$\displaystyle \pmb{M}= \begin{bmatrix} A & B \\C & D \end{bmatrix}$$ 
# 
# ### **(i) Determinant**
# 
# The determinant is $\displaystyle |\pmb{M}|=AD-BC$
# 
# ### **(ii) Trace**
# 
# The trace is $ Tr(\pmb{M}) = A + D $
# 
# ### **(iii) Inverse**
# 
# The inverse is inverse $\displaystyle \pmb{M}^{-1}= \frac{1}{|\pmb{M}|}\begin{bmatrix} D & -B \\-C & A \end{bmatrix}$ 
# 
# ### **(iv) Eigenvalues**
# 
# The eigenvalues (see section 12.3) are solutions of the characteristic equation 
# 
# $$\displaystyle \lambda^2-\lambda(A+D)-(AD-BC)=0$$
# 
# or equivalently 
# 
# $$\displaystyle \lambda^2-\lambda Tr(\pmb{M})+|\pmb{M}|=0$$
# 
# which are
# 
# $$\displaystyle \lambda_{1,2}= \frac{A+D\pm \sqrt{(A+D)^2-4(AD-BC)}}{2}$$
# 
# As a check $\lambda_1+\lambda_2=Tr(\pmb{M})$ and $\lambda_1\lambda_2=|\pmb{M}|$.
# 
# The eigenvectors are $\displaystyle v_1=k \begin{bmatrix} B\\\lambda_1-A \end{bmatrix}$ and $\displaystyle v_2=k\begin{bmatrix} B\\\lambda_2-A \end{bmatrix}$ where $k$ is an arbitrary constant, for example to normalise the eigenvectors.
# 
# ## 5.8 Using matrices in Python and SymPy.
# 
# The is a distinction between doing numerical and symbolic calculations. Python/NumPy is used for numerical work and SymPy for algebraic/symbolic calculations. The notation is slightly different depending on whether you use SymPy or NumPy.
# 
# ### **(i) Symbolic calculations using SymPy**

# In[4]:


M, N, a, b, c, d = symbols('M, N, a, b, c, d')    # define symbols to use        
M = Matrix( [[a, b], [c, d]]   )              # note double sets of brackets and capital M
M


# In[5]:


M.det()                         # determinant


# In[6]:


N = Matrix([[d,a],[c,b] ])

N*M                             # matrix multiply


# In[7]:


M*N                              # matrix multiply 


# In[8]:


N*M - M*N                        # M and N do not commute


# In[9]:


V = Matrix([2,3])                # define vector column
V


# In[10]:


W = Matrix ([5,4])
V.dot(W)                         # dot product is a scalar number


# In[11]:


Transpose(V)*W                   # same as dot product 


# In[12]:


V*transpose(W)                   # outer product is a matrix see figure 7


# ### **(ii) Using NumPy for numerical calculation.  Note that the notation is different to that of SymPy** 

# In[13]:


a = np.array([[1, 3],  [5, 1]])  # note that an array is now a treated as a matrix
b = np.array([[4, 1], [2, 2]])
print('a and b')
a,b


# In[14]:


np.matmul(a, b)     # matrix multiply


# In[15]:


a @ b               # equivalent to np.matmul(a,b) 


# In[16]:


a * b               # this is NOT matrix multiply but element by element multiply 


# In[17]:


v = np.array([2,3])


# In[18]:


v.dot(v)            # dot product with itself is scalar


# In[19]:


np.dot(v, v)        # alternative way of doing dot product


# In[20]:


v @ a               # will automatically make transpose


# In[21]:


a @ v               # will automatically make transpose


# In[22]:


a @ b - b @ a       # commute ? No!


# In[ ]:




