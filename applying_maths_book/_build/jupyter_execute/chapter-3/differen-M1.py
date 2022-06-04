#!/usr/bin/env python
# coding: utf-8

# ## Differentiation of vectors 

# ### 13.1 Del, div, grad, Laplacian, and curl
# 
# There are four common vector operators involved in differentiation and the _del_ operator. They are widely used in describing the physics of electrostatics, magnetism, and flowing liquids, and in the properties of fields in general. They are only infrequently met in chemistry or the biosciences.  Only the briefest outline is given here; see a specialist text such as 'Div, Grad, Curl and All That' (Shey 1993) for more details. The _del_ symbol $\nabla$ is also called nabla.
# 
# ### 13.2 Del, $\nabla$
# The _vector operator 'del' is defined in the $(\pmb{i, j, k})$ basis as
# 
# $$\displaystyle \nabla =\frac{\partial}{\partial x}\pmb{ i}+\frac{\partial}{\partial y}\pmb{ j}+\frac{\partial}{\partial z}\pmb{ k}$$
# 
# and the _gradient_ or rate of change of a scalar function $f(x,y,z)$ is the vector 
# 
# $$\displaystyle\nabla f=\frac{\partial f}{\partial x}\pmb{ i}+\frac{\partial f}{\partial y}\pmb{ j}+\frac{\partial f}{\partial z}\pmb{ k}$$
# 
# ### 13.3 Grad $\nabla f$
# 
# The gradient is a vector that gives the maximum rate of change of a function f and its magnitude in a given direction. For example, if the function is $f = \sin(x)yz^2$ then the rate of change is the vector
# 
# $$\displaystyle\nabla f=\frac{\partial f}{\partial x}\pmb{ i}+\frac{\partial f}{\partial y}\pmb{ j}+\frac{\partial f}{\partial z}\pmb{ k}=\cos(x)yz^2\pmb{i}+\sin(x)z^2\pmb{j}+2\sin(x)yz\pmb{k}$$
# 
# which can be resolved into components along the base vector's $\pmb{i, j, k}$ directions. 
# 
# ### 13.4 Div, $\nabla \cdot v$
# If a vector $v$ is represented in the $(\pmb{i, j, k})$ basis as $\pmb{v} = a\pmb{i} + b\pmb{j} + c\pmb{k}$ then the dot product with the vector $\nabla$ is called the _divergence_ and is a scalar defined as
# 
# $$\begin{align}
# \displaystyle \nabla \cdot v = &\left(\frac{\partial }{\partial x}\pmb{ i}+\frac{\partial }{\partial y}\pmb{ j}+\frac{\partial }{\partial z}\pmb{ k}\right) \cdot (a\pmb{i}+b\pmb{j}+c\pmb{k})\\ 
# = & \frac{\partial a}{\partial x}+\frac{\partial b}{\partial y}+\frac{\partial c}{\partial z} \end{align}$$
# 
# and the $a, b$, and $c$ must depend on $x, y$, and $z$ otherwise the differential would be zero. The base vector multiplication rules are $\pmb{i}\cdot \pmb{i}=1$ and similarly for $\pmb{j}, \pmb{k}$ and product between different vectors are zero because these base vectors have unit length and are orthogonal, i.e. at right angles to one another. 
# 
# Without specifying at what point we want the gradient to be calculated, the calculation cannot be continued further as was the case $\nabla f$ calculated above. Suppose that the function is again $f = \sin(x)yz^2$  and the point is $(\pi/2, 2, 3)$ then the gradient, or rate of change at this point, is the vector $\nabla f = 0\pmb{i} + 9\pmb{j} + 12\pmb{k}$; its magnitude is the absolute value of the vector which is $\sqrt{81 + 144} = 15$. Furthermore suppose that we want the gradient from our point in the direction towards the origin $(0,0,0)$. The vector to the origin is $\pmb{v} = -(\pi/2)\pmb{i} - 2\pmb{j} - 3\pmb{k}$ which has magnitude $\sqrt{\pi^2/4+4+9}$. To make it a unit vector $\pmb{u}$, it is divided by its length
# 
# $$\displaystyle \pmb{u}=\frac{-(\pi/2)\pmb{i} - 2\pmb{j} - 3\pmb{k}}{\sqrt{\pi^2/4+13}}$$
# 
# The magnitude of the rate of change is therefore
# 
# $$\displaystyle \nabla f\cdot\pmb{u}= \frac{(-(\pi/2)\pmb{i} - 2\pmb{j} - 3\pmb{k})\cdot(0\pmb{i} + 9\pmb{j} + 12\pmb{k})}{\sqrt{\pi^2/4+13}}=-11.18$$
# 
# and lies in the direction given by $\nabla f = 0\pmb{i} + 9\pmb{j} + 12\pmb{k}$, and the maximum rate of change is the size of this vector which is $15$.
# 
# ### 13.5 Laplacian, $\nabla\cdot\nabla f$
# 
# The dot product of the operator del with itself is a scalar function called the _Laplacian_, which is
# 
# $$\displaystyle \nabla ^2 f=\nabla\cdot(\nabla f)=\frac{\partial^2 f}{\partial x^2}+\frac{\partial^2 f}{\partial y^2}+\frac{\partial^2 f}{\partial z^2}$$
# 
# ### 13.6 Curl, $\nabla \times \pmb{v}$
# 
# The cross product of del with a vector $v$ is a vector called the curl
# 
# $$ \displaystyle \nabla \times \pmb{v} = \begin{bmatrix}
# \pmb{i} & \pmb{j} & \pmb{k} \\
# \displaystyle\frac{\partial }{\partial x} & \displaystyle\frac{\partial}{\partial y} & \displaystyle\frac{\partial }{\partial z}\\
# a & b & c 
# \end{bmatrix}$$
# 
# This vector is best left as a determinant in which form it is easier to remember.
