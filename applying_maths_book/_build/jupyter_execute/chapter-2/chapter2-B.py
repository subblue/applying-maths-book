#!/usr/bin/env python
# coding: utf-8

# ## De Moivre's theorem and integer powers of complex numbers

# ### Complex number as $z = r\left(\cos(\theta) + i \sin(\theta)\right)$
# 
# A complex number $z=a+ib$ can be written equivalently as
# 
# $$\displaystyle z = r\left(\cos(\theta) + i \sin(\theta)\right)$$
# 
# and if $n$ is an integer, what is $z^n = r^n\left(\cos(\theta) + i \sin(\theta)\right)^n$ ? 
# 
# The trigonometric part can be shown to have the simple form,
# 
# $$\displaystyle \left(\;\cos(\theta) + i \sin(\theta)\;\right)^n = \cos(n\theta) + i \sin(n\theta) \tag{8}$$
# 
# therefore 
# 
# $$\displaystyle z^n = r^n\left(\cos(n\theta) + i \sin(n\theta) \right)\tag{9}$$
# 
# which is called De Moivre's theorem and is essential to calculating powers of complex numbers. One of the unexpected things that can be done is to find the $n^\mathrm{th}$ root of $1,\, i, \,-3$ or any other number for that matter.
# 
# To demonstrate that De Moivre's theorem is correct, calculate the product of two complex numbers expressed in angular form, and then let $\theta_1 = \theta_2$. Suppose, for simplicity, that $r_1 = r_2 = 1$, then the product of two numbers is
# 
# $$\displaystyle \begin{align}
# \big(\cos(\theta_1) + i\sin(\theta_1)\big)\big(\cos(\theta_2) + i\sin(\theta_2)\big) 
# & = \cos(\theta_1)\cos(\theta_2) + i \cos(\theta_1)\sin(\theta_2) + i\sin(\theta_1)\cos(\theta_2) -\sin(\theta_1)\sin(\theta_2)\\
# &=\cos(\theta_1 +\theta_2)+i\sin(\theta_1 +\theta_2)
# \end{align}$$
# 
# The double angle formula (Chapter 1.5.1) was used in the last step, and letting $\theta_1 = \theta_2$ produces
# 
# $$\displaystyle \big(\cos(\theta) + i\sin(\theta)\big)^2 = \cos(2\theta) + i\sin(2\theta)$$
# 
# as predicted by De Moivre's theorem. This result can be generalized to any power of a real or complex value $n$.
# 
# The product $z_1z_2$ and quotient $z_1/z_2$ of two complex numbers are written in this form as
# 
# $$\displaystyle z_1z_2 = r_1r_2\big(\cos(\theta_1 + \theta_2) + i \sin(\theta_1 + \theta_2)\big) \tag{10}$$
# 
# where the angles add, and provided that $z_2 \ne 0$,
# 
# $$\displaystyle \frac{z_1}{z_2} =\frac{r_1}{r_2} \big(\cos(\theta_1 -\theta_2)+i\sin(\theta_1 -\theta_2)\big)$$
# 
# where the angles subtract. 
# 
# There is a geometrical interpretation to multiplying two complex numbers. If their moduli are unity, $z_1 = \cos(\theta_1) + i \sin(\theta_1)$ and $z_2 = \cos(\theta_2) + i \sin(\theta_2)$, then multiplication results in rotation about the origin, equation (10). Geometrically this is shown in figure 5.
# 
# ![Drawing](chapter2-fig5.png )
# 
# Figure 5. Geometrical interpretation of the multiplication of two complex numbers.
# _____
# 
# ### Hyperbolic functions and complex numbers
# 
# In the case of hyperbolic functions there are related formulae since $\displaystyle \cosh(x) + \sinh(x) = e^x$ then 
# 
# $$\displaystyle \big(\cosh(x)+\sinh(x)\big)^n=\cosh(nx)+\sinh(nx) $$
# 
# and for a complex number $z=x+iy$ 
# 
# $$\displaystyle \big(\cos(z)+\sin(z)\big)^n=\cos(nz)+\sin(nz) $$
# 
# 
# ### 3.1 Roots of a complex number
# 
# Suppose that $w$ is a real or complex number whose roots we need to find, then mathematicians have shown that, in general, the answer will be a complex number. If the $n$ roots of a number $z$ are expressed as $w = z^{1/n}$, then the equation to examine is $w^n = z$.
# 
# We will let both sides of this equation be different complex numbers. Expressing the left-hand side in angular form using De Moivre's theorem with a polar angle $\varphi$ gives
# 
# $$\displaystyle w^n = R^n(\cos(n\varphi) + i \sin(n\varphi)) \tag{11}$$ 
# 
# The right-hand side of the equation is
# 
# $$\displaystyle z = r(\cos(\theta) + i \sin(\theta)) \tag{12}$$
# 
# since any complex number can be written in this way. Therefore, $R^n = r$ where both $R$ and $r$ are real numbers. The angles $\varphi$ and $\theta$ are related in the most general way as
# 
# $$\displaystyle n\varphi = \theta + 2\pi k \tag{13}$$
# 
# where $k = 0,\, 1,\, 2, \cdots n - 1$ because sine and cosine are cyclic functions; $\sin(\theta) = \sin(\theta + 2\pi) = \sin(\theta + 4\pi)$ and so forth, therefore there will be more than one root to the equation. Using $n\varphi = \theta$ only allows one root to be found. Using equations 11 and 13, gives
# 
# $$\displaystyle w=R^{1/n}\left( \cos\left( \frac{\theta+2\pi k}{n} \right) +i\sin\left( \frac{\theta+2\pi k}{n} \right)  \right) \tag{14}$$
#  
# In the special case of calculating the $n^\mathrm{th}$ root of unity, $w^n = 1$ and $z = 1$, then from equation 12, $r = 1,\; \theta = 0$ and therefore, 
# 
# $$\displaystyle w= \cos\left( \frac{2\pi k}{n} \right) +i\sin\left( \frac{2\pi k}{n} \right)   \tag{15}$$
#  
# There is always one real root and the other roots fall on the vertices of a polygon which is formed inside a circle of unit radius and touches the circle only at its vertices.
# 
# To illustrate the method, $w^5 = 1$ is solved to find the five, fifth roots of unity. The equation to use is $w^n = z$ with $n = 5$ and $z = 1$. The roots are the solution of equation 15 with $n = 5$,
# 
# $$\displaystyle z=1^{1/5}= \cos\left( \frac{2\pi k}{5} \right) \pm i\sin\left( \frac{2\pi k}{n} \right)   $$
# 
# where $k = 0, \,1, \,2, \,3, \,4$. The principal value of the equation is the one solved with $k = 0$. The
# five roots are then
# 
# $$\displaystyle \begin{align} w = 1,\qquad & \cos(2\pi /5) + i \sin(2\pi /5), \qquad \cos(4\pi /5) + i \sin(4\pi /5),\\
# &\cos(6\pi /5) + i \sin(6\pi /5),\qquad \cos(8\pi /5) + i \sin(8\pi /5) \end{align}$$
# 
# and as $\sin(2\pi /5) = -\sin(8\pi /5)$ and so forth, only the positive terms need be used. Only one of the roots is not a complex number and as this first root lies on the real axis, the angle to the next root is
# 
# $$\displaystyle \theta = \tan^{-1}\left(\frac{\sin(2\pi/5)}{\cos(2\pi/5)}\right) \equiv 72^\mathrm{o}$$
#  
# and the other roots are separated from each other by the same angle as expected for a pentagon, see figure 6.
# 
# ![Drawing](chapter2-fig6.png )
# 
# Figure 6. The five roots of the equation $z^5 = 1$ form a pentagon. The radial lines to each root are $72^\text{o}$ apart. 

# In[ ]:




