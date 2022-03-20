#!/usr/bin/env python
# coding: utf-8

# ## Limits, maxima/minima and Calculus of Variations. 

# ### 6  l' Hopital's rule
# In many calculations limits are encountered. For example, in the theory of diffraction the function $\displaystyle \frac{\sin(x)}{x}$ is met which, when $x \to 0$, has the form $0/0$ and at first sight this ratio seem to be indeterminate. There are other forms similar to this such as $\infty/\infty,\, \infty/0,\, 0\times \infty,\, 0\times 0,\, \infty-\infty$ and l'Hopital's rule is a method, sometimes used with a little additional ingenuity, of determining these limits. This topic is discussed here, not only because it requires differentiation, but also because it will be needed in the next section. It seems that Johann Bernoulli first worked this out, but it is named after l'Hopital who was one of his pupils.
# 
# The method is
# 
# (a) Rearrange the limit required, if necessary, so that it becomes a ratio. This may require some cunning.
# 
# (b) Differentiate top and bottom separately with respect to the limit variable. 
# 
# (c) Substitute in the limit and check if the ratio is still indeterminate. If it is return to (b) if not the answer has been found.
# 
# An example should make this clearer; the ratio $\displaystyle \frac{\sin(x)}{x}$ appears to be $0/0$ as $x \to  0$ but it has a finite value which is determined using l'Hopital's Rule:
# 
# $$\displaystyle  \lim_{x\to o} \frac{\sin(x)}{x} \to \frac{\frac{d}{dx}\sin(x)}{\frac{d}{dx}x}=\frac{\cos(x)}{1}=1$$
# 
# and the limiting value of $x = 0$ is only applied in the last step. As a check, one way to determine any limits close to zero is to plot the function. You will see that $\displaystyle \lim_{x\to 0}\frac{\sin(x)}{x}$ does indeed have a value of 1 at $x = 0$. See question 47 for other examples.
# 
# Expressions such as $\displaystyle \frac{e^x-1}{x^2}$ often need to be evaluated when $x \to 0$ or $x \to \infty$. In statistical mechanics, for example, $x$ may be $-E/k_BT$, a ratio of energies where $k_B$ is the Boltzmann constant and $T$ temperature. When $x \to 0$, which corresponds to high temperatures, the function $\displaystyle(e^x - 1)/x^2$ appears to have the indeterminate form $0/0$, but the limit by l'Hopital's is
# 
# $$\displaystyle  \lim_{x\to 0}\frac{e^x -1}{x^2}\to \frac{e^x}{2x}\to \frac{e^x}{x}=\frac{1}{2} \text{ this is wrong!}$$
# 
# where differentiation was performed twice over.  However, this is _wrong_: no check for an indeterminate  ratio was made after the first differentiation and doing this gives, 
# 
# $$\displaystyle  \lim_{x\to 0}\frac{e^x -1}{x^2}\to \frac{e^x}{2x}= \frac{1}{0}=\infty$$
# 
# The other limiting case $x \to \infty$ is also infinity: note that after the first differentiation checking the ratio gives $\displaystyle \frac{e^x}{2x} = \frac{\infty}{\infty}$ so indeterminate and so continue to differentiate again
# 
# $$\displaystyle  \lim_{x\to \infty}\frac{e^x -1}{x^2}\to \frac{e^x}{2x}\to \frac{e^x}{2}=\infty$$
# 
# and this result is because $e^x$ increases faster with $x$ than $x^2$ does. You could plot a graph to see that this is true.
# 
# The limit $\displaystyle \lim_{x\to\infty} \frac{e^{-x}-1}{x^2-3}$  can perhaps be appreciated by looking at the ratio and noticing that when $x$ is large $e^{-x}$ becomes small but $x^2$ large so the limit is expected to be zero. Checking this properly gives the same result as intuition:
# 
# $$\displaystyle  \lim_{x\to \infty}\frac{e^{-x} -1}{x^2-3}\to \frac{-e^{-x}}{2x}\to \frac{e^{-x}}{2}=0$$
# 
# When the limit is a product this need to be rearranged first, for instance $\displaystyle \lim_{x\to 0} x \ln(x)$ should be rearranged to
# 
# $$\displaystyle  \lim_{x\to 0} \frac{ \ln(x) }{ 1/x } \to \frac{1}{ x(-x^{-2})}=-x \to 0$$
# 
# Fractions are treated similarly; the following fraction is nominally undefined as $\infty -\infty$  but is rearranged into a ratio to become undefined as $0/0$;
# 
# $$\displaystyle   \lim_{x\to 0} \left(\frac{1}{x}-\frac{1}{\sin(x)}\right)= \lim_{x\to 0} \frac{\sin(x)-x}{x\sin(x)}\to \frac{\cos(x)-1}{x\cos(x)+\sin(x)}  \to \frac{-\sin(x)}{-x\sin(x)+2\cos(x)} \to 0$$
# 
# In the last step substituting for $x=0$ gives the ratio as $0/1$ so  the limit is zero.
# 
# ### 6.1 Beware of 'False fractions'
# 
# Sometimes a limit is required but the expression is not really a fraction, for example $\displaystyle \lim_{x\to 1} \frac{x^4-1}{x-1} $ which looks like a fraction but is simplified to $\displaystyle \lim_{x\to 1} \frac{(x-1)(x^3+x^2+x+1)}{(x-1)} $ and because the value $x = 1$ is never reached ( we are seeking the limit to not the value at $x = 1$) the $x-1$ terms can be cancelled out leaving $\lim_{x\to 1} (x^3+x^2+x+1)=4 $
# 
# ## 7 Extrema: maxima, minima and inflection points
# 
# One very useful property of derivatives is that they allow us to find the maxima and minima of functions; these are also called stationary points of the function. The extrema might be the maximum or minimum but can also be the limit where the function goes to $\pm \infty$.
# 
# The maximum of a curve is a point in whose locality all surrounding points have smaller $y$ values. The minimum is defined similarly but points adjacent to it have larger y values and in both case the gradient is exactly zero. In Fig. 12 it is clear that the gradient is zero at the maximum and again at the minimum of the curve. The graph shows the first $f'$ and second $f''$ derivatives. An inflexion point can occur when the gradient is zero but $y$ is smaller on one side of the point than it is on the other. An inflexion can also occur when the gradient is not zero in which case the curvature of the line changes from concave to convex or vice versa.
# 
# The first derivative is zero at $x$ = 4/3 and 3, which are the maximum and minimum respectively. The second derivative, which is the straight line, is negative at the maximum and positive at the minimum, so that the maximum and minimum of a function can be found from knowledge of the first and second derivatives.
# 
# The function shown in Fig. 12 is the equation $y = 2(x - 2)^3 - x^2 + 4$, which has three roots, where $x = 0$ at $x \approx 0.8, 2$, and $\approx 3.7$. The first derivative is $y' = 6(x - 3)(x - 4/3)$, which has roots at $x = 4/3$ and $3$, which from the graph (left) are the maximum and minimum. The second derivative $y'' = 12x - 26$ is negative where $x \lt 13/6$ ($\approx 2.2$) as can also be seen from the graph.
# 
# ![Drawing](differen-fig12.png)
# 
# Figure 12. The curve $f(x) = 2(x - 2)^3 - x^2 + 4$ showing its maximum and minimum, and its first $f'(x)$ and second $f''(x)$ derivatives. Notice how the first derivative is zero at the maximum and also the minimum and that the second derivative determines which is which in cases when a plot is not made.
# _____

# ### 7.1 Summary
# 
# A function $f(x)$ has a maximum or minimum when $\displaystyle \frac{d}{dx}f(x)=0 \tag{25}$
# 
# The maximum occurs when $\displaystyle \frac{d^2}{dx^2}f(x) \lt 0\tag{26}$
# 
# The minimum occurs when $\displaystyle \frac{d^2}{dx^2}f(x) \gt 0 \tag{27}$ 
# 
# If both first and second derivatives are zero at the same $x$ then this is a _point of inflexion_. 
# 
# ![Drawing](differen-fig13.png)
# 
# Figure. 13(a) Left. A point of inflexion occurs when both first and second derivatives are 0 at the same point on the curve,in this curve $y=2(x-1)^3$ +1 this is at the point {1,1}. Fig 13b Right. shows the curve $y=1/(1+x^2)$ and its inflexion points.
# _____
# The curve $\displaystyle y=\frac{1}{1+x^2}$ has inflection points as shown in fig 13(b).  The first and second derivatives are 
# 
# $$\displaystyle   y'=-\frac{2x}{(1+x^2)^2}, \qquad y''=\frac{6x^2-2}{(1+x^2)^3}$$
# 
# The first derivative is zero at $x=0$, and the second zero at $\displaystyle x=\pm\frac{1}{\sqrt{3}}$  and $\displaystyle y=\pm\frac{3}{4}$

# ## 8 The Calculus of Variations
# 
# Instead of finding the maximum or minimum of a curve, consider finding the shortest distance between two points on the earth's surface or on a cone, or finding the equation giving the minimum area of a surface, or the curve of fastest descent between two points. The _Calculus of Variations_ allows us to work out solutions to problems of this type.
# 
# You can imagine a graph that shows the many possible different curves that will fit between any two points. A practical example would be all the different routes by which you could travel from, say, Paris to Berlin.  The calculus of variations is so named because it varies any path by an amount $\delta y$ and, as the variation, $\delta y \to 0$, the path along $y$ becomes the same as that along $y + \delta y$. Because the whole path is sought, the problem is to find a function $f$ that, provided that it exists, will make an _integral have a stationary_ value, also called an _extremal_, which usually means that it has the smallest possible value. This condition is written as
# 
# $$\displaystyle   I=\int_a^b f\left(x,y,\frac{dy}{dx}\right)\; dx \tag{28}$$
# 
# Notice that the function $f(\cdots)$ normally includes three terms, one in the independent variable $x$, the next in the dependent one $y$ and the last in the derivative $dy/dx$. Although this equation contains an integral, when solved to find its minimum, differentiation is mainly involved. You may need to consult Chapter 4 on integration to complete the last step in this type of problem.
# 
# Consider, for example, finding the equation that describes the minimum value of _all_ possible surfaces of revolution. A surface of revolution is the surface obtained by rotating a curve, such as a parabola, about an axis; Fig. 8 shows the shape of a parabola and its surface of revolution is shaped somewhat like a bowl. Whatever the equation, $y = \cdots$ is, the surface area is always given by the integral
# 
# $$\displaystyle   2\pi \int_a^b y ds=2\pi \int_a^b y\sqrt{1+\left( \frac{dy}{dx} \right)^2}dx  \tag{29}$$
# 
# and a straightforward integration with the parabola $y = 2\sqrt{ax}$ will produce the parabola's surface area. The term in the square root is the length of a small element of the curve, see Fig. 19 (Q51) and the integral has the form of equation (28). 
# 
# Imagine now a surface film suspended between two similar wire hoops at $x = a$ and $b$, in practice this could be a soap film, see Fig. 27. Now, suppose that the problem is to find that one particular curve, of all possible curves, that will produce the minimum surface area within the two rings; this minimum area surface is the surface formed by a soap film, and its profile is called the Catenary. The equation for the film $y = \cdots$ was not known before starting the calculation. The calculus of variations allows it be found by first finding $dy/dx$ and then integrating it.
# 
# The calculus of variations defines a formula, variously called The Euler or the Euler - Lagrange equation, by which it is possible to evaluate the integral (28) _so that it has its minimum value_. The Euler equation is
# 
# $$\displaystyle   \frac{\partial f}{\partial y}-\frac{d}{dx}\frac{\partial f}{\partial y_x}=0 \tag{30}$$
# 
# where in this instance the function $f$ is
# 
# $$\displaystyle  f= y\sqrt{1+\left( \frac{dy}{dx} \right)^2} $$
# 
# and $y_x = dy/dx$ is the expression we want to find, which, when integrated, produces $y$, the equation of the curve required. Notice that the equation tells us to differentiate with respect to $dy_x$; see Section 5.6, which describes differentiation with respect to a function. Should the equation for $f$ _not explicitly contain_ $x$ then a simpler version can be used which is
# 
# $$\displaystyle   \frac{\partial f}{\partial y}-\frac{d}{dx}\left(f-y_x\frac{\partial f}{\partial y_x}\right)=0 \tag{31}$$
# 
# and the extremal is found from 
# 
# $$\displaystyle  f-y_x\frac{\partial f}{\partial y_x} = const \tag{32}$$
# 
# This is sometimes called the 'Beltrami' identity. The words 'not explicit' mean that $f$ is a function of $y$ or $dy/dx$ such as $ f = y^2 + dy/dx$ which does _not_ explicitly depend on $x$; the function $f = x^2$ explicitly depends on $x$.
# 
# ### 8.1 Calculating the shape of the minimum surface of revolution.
# 
# Using the Euler equation is actually not that difficult and as an example the minimum surface of revolution of all possible surfaces for a film suspended between two rings is calculated.  A surface of revolution is given by equation (29) but if the surface is to be a minimum then (30) or (32) must also apply. The latter equation is easier to use with $f =y \sqrt{1 + (dy/dx)^2}$ because this does not explicitly contain $x$. 
# 
# Using the notation $\displaystyle y_x = \frac{dy}{dx}$ then $f =y \sqrt{1 + y_x^2}$ and the derivative in eqn 32 becomes, 
# 
# $$\displaystyle  \frac{\partial f }{\partial y_x}  = \frac{1}{2}(1+y_x^2)^{-1/2}2y_x$$
# 
# Substituting into eqn 32 and changing to full notation produces 
# 
# $$\displaystyle   y\left[ 1+\left(\frac{dy}{dx}\right)^2   \right]^{1/2}  - y\left(\frac{dy}{dx} \right)^2 \left[ 1+\left(\frac{dy}{dx}\right)^2  \right]^{-1/2}=y\left[   1+\left(\frac{dy}{dx} \right)^2\right]^{-1/2}= a$$
# 
# where $a$ is a constant. Rearranging the result leads to 
# 
# $$\displaystyle \frac{dy}{dx}= \sqrt{ \left( \frac{y}{a} \right)^2 - 1 } $$ 
# 
# which, when integrated, gives 
# 
# $$\displaystyle y = a \cosh\left(\frac{x}{a} + b\right)$$
# 
# where $b$ is a constant of integration. The constant $b$ sets the position of the minimum of the curve; $b = 0$ sets the minimum at $x = 0$, whereas $a$ determines the depth of the curve . This equation describes the Catenary or the curve describing the shape of the edge of the minimum surface of revolution. It is also the shape produced by a flexible cable or chain hanging under its own weight i.e. under a constant uniform force such as gravity. The shape is very close to, but distinct from that of  parabola. 

# ### 8.2 The Brachistochrone, the Tautochrone and the Cycloid
# 
# The Brachistochrone is the name of the curve a frictionless particle will travel along to pass between two points in the shortest time when acted on by a force such as gravity. The time taken is much less than that taken to move down a straight slope or, in fact, any other slope. The Tautochrone is the same curve but is used to describe the fact that a particle set in motion down the curve will arrive at the horizontal part at the same time irrespective of where it starts from. The inverted Brachistochrone is the cycloid, see question 53.
# 
# As it will simplify the calculation, we will suppose that a particle of mass $m$ travels downwards in the +$x$ direction and moves to the right as the +$y$ direction after starting at rest at the origin (Margenau & Murphy, 1943). Using conservation of potential and kinetic energy,
# 
# $$\displaystyle  mgx = \frac{mv^2}{2}$$
# 
# where $v$ is the velocity at any point on the path and $g$ the acceleration due to gravity. The velocity is found starting with a small (infinitesimal) element $ds$ of the path and using Pythagoras ($\displaystyle ds^2=dx^2+dy^2$) is therefore valid and gives
# 
# $$\displaystyle   v=\frac{ds}{dt}=\frac{\sqrt{dx^2+dy^2}}{dt}$$
# 
# then substituting for $v$ and rearranging gives the change in time, 
# 
# $$\displaystyle dt= \frac{\sqrt{1+(dy/dx)^2} }{\sqrt{2gx}}dx $$
# 
# The integral to minimize is therefore the time producing
# 
# $$\displaystyle  t=\frac{1}{\sqrt{2g}}\int_{x_0}^{x} \frac{\sqrt{1+y_x^2}}{\sqrt{x}}dx$$
# 
# where $y_x \equiv dy/dx $. The constant $2g$ can be ignored for it will not enter into the shape of the curve. The result of the calculation is an equation $y$ vs $x$ that the particle follows. 
# 
# ![Drawing](differen-fig14.png)
# 
# Figure 14. Part of the Brachistochrone, notice that the axes are rotated clockwise by $90$ degrees compared to convention.
# ____
# 
# Using the Euler equation (30) with $\displaystyle f=\frac{\sqrt{1+y_x^2}}{\sqrt{x}}$ and because $\displaystyle df/dy$=0, this leaves just the second term in eqn. 30 as
# 
# $$\displaystyle  \frac{d}{dx}\left(\frac{y_x}{\sqrt{x}\sqrt{1+y_x^2}}    \right) =0$$
# 
# To work towards obtaining the equation for  $y$ this equation is integrated once to produce
# 
# $$\displaystyle  \frac{y_x}{\sqrt{x}\sqrt{1+y_x^2}}=\sqrt{c}$$
# 
# The constant of integration is chosen as  $\sqrt{c}$ rather than $c$  to make the following equations simpler. To find $y_x=dy/dx$ both sides are squared and after multiplying top and bottom inside the square root by $(1-cx)$ and rearranging this gives
# 
# $$\displaystyle  \frac{dy}{dx}={\frac{\sqrt{cx(1-cx)}}{1-cx}}$$
# 
# Integrating produces with $cx\lt 1$,
# 
# $$\displaystyle  y=\frac{\sin^{-1}\left(\sqrt{1-cx}\right)}{c}+\sqrt{\frac{x(1-cx)}{c}}+c_1$$
# 
# where $c_1$ is a second constant of integration. If the particle starts at the origin then $y = 0$ when $x = 0$ and therefore $c_1 = -\pi/2c$. This is the curve shown in fig. 14 but plotted with axes as shown and not in the conventional way.
# 
# The solution to this problem was found independently by the Bernoulli brothers, and by Leibniz, Huygens and Newton towards the end of the seventeenth and beginning of the eighteenth centuries.

# In[ ]:




