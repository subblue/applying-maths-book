#!/usr/bin/env python
# coding: utf-8

# # 14 Linear equations with variable coefficients

# ## 14.1 Linear equations with variable coefficients
# A more difficult equation to solve than those met previously is
# 
# $$\displaystyle a(x)\frac{d^2y}{dx^2}+b(x)\frac{dy}{dx}+c(x)y=g(x)$$
# 
# where the coefficients $a$, $b$, $c$ are functions of $x$. This equation is usually put into the form
# 
# $$\displaystyle \frac{d^2y}{dx^2}+P(x)\frac{dy}{dx}+Q(x)y=f(x)$$
# 
# by dividing through by $a(x)$. The general method to use is the _variation of parameters_, because this can be applied to all differential equations. However, it is quite complicated to use and specialized texts should be consulted (Jeffery 1990; Bronson 1994; Aratyn & Rasinariu 2006). There are, however, at least two methods that can still be used: the main one is to solve the equation as a series expansion and the second reduces the equation to a simpler one, but this can be used only in certain cases.
# 
# ## 14.1 Reduction to simpler forms by change of variable and substitution
# 
# ### **(i) Substitution**
# Sometimes a simple substitution can produce constant coefficients, but success in this depends very much on the exact form of the equation. The equation 
# 
# $$\displaystyle \cos(x)\frac{d^2y}{dx^2}+\sin(x)\frac{dy}{dx}+a\cos^3(x)y=0$$
# 
# can be simplified with $u = \sin(x)$. The derivatives are 
# 
# $$\displaystyle \frac{du}{dx} = \cos(x), \; \frac{d^2u}{dx^2} = -\sin(x)$$
# 
# and as 
# 
# $$\displaystyle \frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx},\quad\text{ then }\quad\frac{dy}{dx}=\cos(x)\frac{dy}{du}$$
# 
# and
# 
# $$\displaystyle \frac{d^2y}{dx^2}=-\sin(x)\frac{dy}{du}+\cos(x)\frac{d^2y}{du^2}\frac{du}{dx}=-\sin(x)\frac{dy}{du}+\cos^2(x)\frac{d^2y}{du^2}$$
# 
# substituting into the differential equation produces 
# 
# $$\displaystyle \frac{d^2y}{du^2}+ay=0$$
# 
# which is readily solved by standard methods to find $y$ in terms of $u$. Notice that if the $\cos^3(x)y$ term in the original equation was replaced by $\cos(x)y$, then the equation produced is not as simple and is 
# 
# $$\displaystyle (1-u^2)\frac{d^2y}{du^2}+ay=0$$
# 
# ### **(ii)  Euler or Cauchy eqns.**
# Equations of the particular type 
# 
# $$\displaystyle x^3\frac{d^3y}{dx^3}+ax^2\frac{d^2y}{dx^2}+bx\frac{dy}{dx}+cy=f(x)$$ 
# 
# where terms in $x$ have the same power as the differential are called Euler's or Cauchy's equation and can be simplified with the substitution 
# 
# $$\displaystyle x = e^{-z}$$
# 
# The derivative is $\displaystyle dz/dx = e^{−z}$ and if $D \equiv d/dz$ then
# 
# $$\displaystyle \frac{dy}{dx}=\frac{dy}{dz}\frac{dz}{dx}=e^{-z}Dy, \qquad \frac{d^2y}{dx^2}=\frac{d^2y}{dz^2}\left(\frac{dz}{dx}\right)^2+\frac{dy}{dz}\frac{d^2z}{dx^2}=e^{-2z}(D^2-D)y$$
# 
# The last derivative here was obtained as 
# 
# $$\displaystyle \frac{d^2y}{dx^2}=\frac{d}{dz}\left(\frac{dy}{dz}\frac{dz}{dx} \right)\frac{dz}{dx}$$
# 
# and using the product rule and $\displaystyle \frac{d}{dx}\frac{dz}{dx}\equiv\frac{d}{dx}$
# 
# The different derivatives can be written down as
# 
# $$\displaystyle x\frac{dy}{dx}=Dy, \qquad x^2\frac{d^2y}{dx^2}=(D^2-D)y, \qquad x^3\frac{d^3y}{dx^3}=(D^3-3D^2+2D)y$$
# 
# and used to simplify the differential equation. Note that the right-hand side has derivatives in $dy/dz$.
# 
# ### **(iii) Using derivatives**
# To solve the equation 
# 
# $$\displaystyle x^2\frac{d^2y}{dx^2}+2x\frac{dy}{dx}-y=0\quad\text{ or }\quad (D^2+D-1)y=0$$ 
# 
# first find the roots of the characteristic equation which are 
# 
# $$\displaystyle \frac{\sqrt{5}-1}{2},\;\frac{-\sqrt{5}-1}{2}$$
# 
# However, the operator $D$ is a function of $z$ not of $x$, hence, the solution is a function of $z$, giving the solution
# 
# $$\displaystyle y=Ae^{(\sqrt{5}-1)z/2}+Be^{(-\sqrt{5}-1)z/2}$$
# 
# where $A$ and $B$ are constants determined by the initial conditions. As $\displaystyle x = e^z$ or $z = \ln(x)$, the solution becomes 
# 
# $$\displaystyle y=Ae^{(\sqrt{5}-1)\ln(x)/2}+Be^{(-\sqrt{5}-1)\ln(x)/2}$$
# 
# which can be simplified to
# 
# $$\displaystyle y=Ax^{\large{(-1+\sqrt(5))/2}}+Bx^{\large{(-1-\sqrt(5))/2}}$$
# 
# ### **(iv) Special form**
# If the equation can be put into the form, 
# 
# $$\displaystyle \frac{d^2y}{dx^2}+P(x)\frac{dy}{dx}+Q(x)y = 0 $$ 
# 
# then it can be reduced by the transformation $y = uv$ where 
# 
# $$\displaystyle v = e^{-(1/2)\large{\int} pdx}$$
# 
# The equation becomes
# 
# $$\displaystyle \frac{d^2u}{dx^2}-w(x)u=0, \qquad w(x)=\frac{dP}{dx}+\left(\frac{P}{Q}  \right)^2-Q$$
# 
# and the final solution is $y=uv$.
# 
# ### **(v) Special form**
# The equation 
# 
# $$\displaystyle \frac{d^2 y}{dx^2}+x\frac{dy}{dx}+(x-1)y = 0$$ 
# 
# can be solved with the transformation method ($y=uv$) if 
# 
# $$\displaystyle v=e^{-(1/2)\large{\int}xdx}= e^{-x^2/4}$$
# 
# and 
# 
# $$\displaystyle w=3/2+x^2/4-x$$
# 
# which produces the equation with which to find $u$,
# 
# $$\displaystyle \frac{d^2u}{dx^2}=\left(\frac{3}{2}+\frac{x^2}{4} -x \right)u = 0$$
#   

# ## 14.2 Series solution of differential equations
# 
# While many types of equations can be solved using the methods described so far, there are a number of problems whose equations can only be solved by a series expansion. These are often different forms of the Schroedinger equation, examples of which are the quantum harmonic oscillator and the radial and angular solutions to the hydrogen atom. 
# 
# While the series solution method will be described in general, in quantum mechanics and in some other problems, the equations often have a form whose solution is well known because the equation has a specific name. The harmonic oscillator is solved using Hermite's differential equation and the hydrogen atom requires Legendre's and Laguerre's equations. Other commonly used equations are named after Helmholtz, Laplace, and Bessel.
# 
# Many functions can be expanded as a power series that has the form
# 
# $$\displaystyle  y=a_0 +a_1x+a_2x^2 +a_3x^3 +\cdots+a_nx^n +\cdots \qquad\tag{38}$$
# 
# with constants $a_0, a_1$, etc. as described in Chapter 5. 
# 
# By taking derivatives of $y$ based on this expansion, the differential equation can be reconstructed in terms of the powers of $x$ and these constants. To find the constants, the powers of $x$ are collected together and the resulting group of constants solved assuming that each group of them is zero. What results is a recursion formula with which to calculate the constants and so the equation is solved without any formal integration. It is assumed that the solution can be found with this method, because this is 'usually' the case for problems in chemical physics. There is a simple way to check if a series solution is possible and if not, a related method due to Frobenius can be tried in these cases.
# 
# ### **(i) The basic series method** 
# The basic series method is illustrated with the equation 
# 
# $$\displaystyle  \frac{d^2y}{dx^2}+x\frac{dy}{dx}+y=0 $$
# 
# and the solution is assumed to have the form of equation (38). The strategy is
# 
# **(a)**$\quad$ to differentiate the series solution, and substitute the results into the differential equation;
# 
# **(b)**$\quad$ collect together coefficients with the same power of $x$ and set each result to zero;
# 
# **(c)**$\quad$ find a recursion equation in the coefficients $a_0,\; a_1, \cdots$ using the initial conditions
# as the starting points.
# 
# **step (a)** The differential equation is formed out of the series solution by taking the derivatives. These are
# 
# $$\displaystyle \begin{align}
# \frac{dy}{dx}&=a_1+2a_2x+3a_3x^2+\cdots + (n-1)a_{n-1}x^{n-2} +na_nx^{n-1}+(n+1)a_{n+1}x^n\cdots \\
# \frac{d^2y}{dx^2}&=2a_2+6a_3x+12a_4x^2 +\cdots +n(n-1)a_{n}x^{n-2}+n(n+1)a_{n+1}x^{n-1}\cdots \end{align} $$
# 
# and the $n$ - 2, $n$ - 1 and n<sup>th</sup> terms are tabulated below as these are needed later on. Notice that these derivatives are the _same for all differential equations_ solved by the expansion in (38). These coefficients are shown in the table and terms that are useful in other equations are also shown.
# 
# $$\displaystyle \begin{array}{c|ccc}
#      & x^{n-2}      & x^{n-1} & x^n \\
# \hline
#  y   & a_{n-2}      & a_{n-1} & a_n \\ 
#  y'  & (n-1)a_{n-1} & na_{n}  & (n+1)a_{n+1} \\ 
#  y'' & n(n-1)a_{n}  & n(n+1)a_{n+1} & (n+1)(n+2)a_{n+2} \\
#  xy  & a_{n-3}      & a_{n-2}       & a_{n-1} \\
#  xy' & (n-2)a_{n-2} & (n-1)a_{n-1}  & na_n\\
#  \hline 
#  \end{array} $$
# 
# **step (b)** The next step is to substitute these derivatives into the differential equation and then to group all the coefficients with $x^0,\; x^1,\; x^2,\; x^3 \cdots$ together and make each group zero. The groups of coefficients for each power of _x_ must be zero, as the differential equation is itself equal to zero. For this particular equation, substituting produces
# 
# $$\displaystyle \begin{array}{lll}
# 2a_2+6a_3x+12a_4x^2+&\cdots  +(n+2)(n+1)a_{n+2}x^n+\cdots \\
# +a_1x + 2a_2x^2 + 3a_3x^3+&\cdots  +na_nx^n+\cdots \\
# +a_0 +a_1x+\;a_2x^2 +a_3x^3 +& +a_nx^n+\cdots=0 \end{array} $$
# 
# Grouping the coefficients gives
# 
# $$\displaystyle \begin{array}{lc}
#   & \text{coefficients}\\
# \hline
#  a_0 + 2a_2 = 0, &x^0  \\
#  6a_3 + 2a_1 = 0, & x^1 \\
# 12a_4 + 2a_2 + a_2 = 0, &  x^2 \\
#  (n + 2)(n + 1)a_{n+2} + na_n + a_n = 0, & x^n \\
#  \hline \end{array}$$
# 
# **step (c)** The recursion formula for the $x^n$ term is 
# 
# $$\displaystyle a_{n+2} =-\frac{(n+1)a_n}{(n+1)(n+2)}$$
# 
# The first two coefficients are $a_0,\;a_1$ and the other coefficients are expressed in terms of these as 
# 
# $$\displaystyle n=0:\; a_2=a_0/2, \quad n=1:\; a_3=a_1/3,\quad n=2:\; a_4=a_2/4=a_0/8,\quad n=3:\; a_5=a_1/15$$
# 
# The solution is therefore,
# 
# $$\displaystyle y = a_0\left(1-\frac{x^2}{2}+\frac{x^4}{8}-\frac{x^6}{48}  +\cdots \right)+a_1\left(-\frac{x}{3}+\frac{x^3}{15}-\frac{x^5}{105}  +\cdots \right) $$
# 
# The two solutions are independent of one another, and the even powered series is the expansion of $\displaystyle e^{-x^2/2}$, while the odd series is not so easily identified. The coefficients are determined by the initial conditions and, if $y(0) = a_0$ and $dy/dx $= 0 at $x$ = 0, then the solution is $\displaystyle y = a_0e^{-x^2/2}$.
# 
# 
# ### **(ii) Example**
# The equation is 
# 
# $$\displaystyle \frac{d^2y}{dx^2}+(\alpha-\beta x^2)y=0$$
# 
# and following the previous calculation, the recursion equation is
# 
# $$\displaystyle a_{n+2}=\frac{\beta a_{n-2}-\alpha a_n}{(n+1)(n+2)}$$
# 
# and is not valid when $n$ is 0 or 1 because $a_{-2}$ and $a_{-1}$ are not defined. The coefficients of the
# first two terms in the expansion must then be examined. These are 
# 
# $$\displaystyle 2a_2+\alpha a_0\quad\text{and}\quad \displaystyle 6a_3+\alpha a_1 =0$$
# 
# from which $a_2$ and $a_3$ can be found in terms of $a_0$ and $a_1$ and these then used as the starting points for the recursion formula. The series solution can be obtained using recursion. The first few terms are 
# 
# $$\displaystyle  \begin{align}a_2&=-\alpha a_0/2,  \quad a_3=-\alpha a_1/6,\\ \quad a_4&=\alpha a_0/24+\beta a_0/12,\quad a_5=\alpha^2 a_1/120+\beta a_1/20 \end{align}$$ 
# 
# and the solution 
# 
# $$\displaystyle y = a_0 + a_1x - \frac{1}{2}\alpha a_0x^2-\frac{1}{6}\alpha a_1x^3+\left( \frac{1}{24}\alpha^2a_0+\frac{1}{12}\beta a_0 \right)x^4+\cdots$$ 
# 
# There are two series in the result; one with terms in $a_0$ and the other in $a_1$. These are independent solutions.
# 
# ### **(iii) The Hermite equation**
# The Hermite equation is important because it leads to the solution of the Schroedinger equation for the quantum mechanical harmonic oscillator. This is discussed at the end of this section but first a method of solution is determined. The equation has the form
# 
# $$\displaystyle \frac{d^2y}{dx^2}-2x\frac{dy}{dx}+2\gamma y=0  \qquad\tag{39}$$
# 
# where $\gamma$ is a real number, Solving as a series leads to the coefficients
# 
# $$\displaystyle \begin{align}2a_2+6a_3x+12a_4x^2+\cdots  \quad + (n+1)(n+2)a_{n+2}x^n&+\cdots \\
# -2a_1x-4a_2x^2-\cdots \quad\qquad\qquad -2na_nx^n&+\cdots\\
# +2\gamma a_0+2\gamma a_1 x+2\gamma a_2x^2+\cdots \quad\qquad\qquad+2\gamma a_nx^n&+\cdots =0\end{align}$$
# 
# The recursion is therefore 
# 
# $$\displaystyle a_{n+2}=2\frac{n-\gamma}{(n+1)(n+2)}a_n$$
# 
# and evaluating the coefficients gives the series 
# 
# $$\displaystyle \begin{align} y_\gamma &=a_0\left(1-\gamma x^2 -\frac{(2-\gamma)\gamma}{6}x^4 -\frac{(4-\gamma)(2-\gamma)\gamma}{90}x^6 -\cdots \right) \\&+a_1\left( x+\frac{1-\gamma}{3}x^3 +\frac{(3-\gamma)(1-\gamma)}{30}x^5+\cdots\right) +\end{align}$$
# 
# These series form the Hermite polynomials when $\gamma$ is a positive integer. For example when $\gamma$ = 3 the odd power series terminates to give 
# 
# $$\displaystyle y_3=a_1\left(x-\frac{2}{3}x^3  \right)$$
# 
# and when $\gamma$ = 4 the even power series terminates and is 
# 
# $$\displaystyle y_4=a_0\left(1-4x^2+\frac{4}{3}x^4  \right) $$
# 
# Each series is limited to a few terms because of the way the coefficients are formed. The first few Hermite polynomials are
# 
# $$\displaystyle \begin{array}{cc}
# H_0(x)=1 & H_1(x)=x \\ H_2(x)=4x^2-2 & H_3(x)=8x^3+12x \\H_4(x)=16x^4-48x^2+12 & H_5(x)=32x^5-160x^3+120x\end{array}$$
# 
# and choosing the constants as 
# 
# $$\displaystyle a_0=(-1)^{\gamma/2}\frac{\gamma!}{(\gamma/2)!}$$
# 
# with $\gamma$ as an even integer and 
# 
# $$\displaystyle a_1=(-1)^{(\gamma +1)/2}\frac{(\gamma +1)!}{((\gamma +1)/2)!}$$
# 
# with $\gamma$ as an odd integer, converts the series solution to the differential equation into the sum of two Hermite polynomials. 
# 
# The general solution can then be written as 
# 
# $$\displaystyle  y_{\gamma} = c_1H_{\gamma}(x)+ c_2H_{\gamma +1}(x)$$
# 
# where $c_1$ and $c_2$ are two new constants that are determined by the initial conditions.
# 
# The equation 
# 
# $$\displaystyle \frac{d^2y}{dx^2}+(2\gamma-1-x^2)y=0 \qquad\tag{40}$$
# 
# with $\gamma$ as a constant is related to the Hermite equation if $\displaystyle y=ve^{-x^2/2}$, where $v$ is a function of $x$, then the second derivative is 
# 
# $$\displaystyle  \frac{d^2y}{dx^2}=\left(\frac{d^2v}{dx^2}-2x\frac{dv}{dx}+(x^2-1)v  \right)e^{-x^2/2}$$
# 
# and substituting this into eqn (40) and simplifying this  becomes 
# 
# $$\displaystyle \frac{d^2v}{dx^2}-2x\frac{dv}{dx}+2\gamma v=0$$
# 
# which is identical to (39). The solutions of (40) are therefore Hermite polynomials multiplied by the Gaussian $\displaystyle e^{−x^2/2}$. 
# 
# The Schroedinger equation for the harmonic oscillator can be written as
# 
# $$\displaystyle \frac{d^2\psi}{dx^2}+\frac{2m}{\hbar^2}\left( E-\frac{k}{2}x^2 \right)\psi=0$$
# 
# where _k_ is the force constant related to the frequency as $\displaystyle \omega= \sqrt{k/m}$. The equation can be simplified to 
# 
# $$\displaystyle \frac{d^2\psi}{dx^2}+(a-b^2x^2)\psi=0$$
# 
# with the abbreviations $\displaystyle a = 2mE/\hbar^2$ and $\displaystyle b^2 = mk/\hbar^2$. Next, the substitution $z =bx$ is used to make the equation similar to (40).  The solutions are Hermite polynomials multiplied by $\displaystyle \exp(−z^2/2)$ and these have the characteristic form of the harmonic oscillator wavefunctions. These decay exponentially to zero when $z$ has a large positive or negative value but oscillate near to zero. The different wavefunctions are found by substituting $E = \hbar\omega(n + 1/2)$ for each quantum number $n$. The next figure (19a) shows some wavefunctions for parameters corresponding to the iodine molecule, assuming that it has a harmonic potential. Of course any diatomic molecule does not have a harmonic potential since all real molecules must dissociate into their respective atoms if given enough energy, but the harmonic potential is a good approximation at low energy.
# 
# ![Drawing](diffeqn-fig19a.png)
# 
# Figure 19a. harmonic oscillator wavefunctions assuming a harmonic potential. The frequency is 214.5 cm$^{-1}$ and the force constant $172$ N/m. The levels have quantum numbers $n=$ 0, 1, 2,$\cdots$.
# ____
# 
# ## 14.3 Checking whether a series solution is possible
# 
# A solution is usually expanded as a series about $x$ = 0; therefore, the first thing to check is whether the series method is applicable. This means examining the differential equation to ensure that $x$ = 0 is analytic, hence not singular, which in turn means that $x$ = 0 does not produce infinity when substituted into the equation.
# The general equation is
# 
# $$ \displaystyle \frac{d^2y}{dx^2}+P(x)\frac{dy}{dx}+Q(x)y=f(x)$$
# 
# If the equation is 
# 
# $$\displaystyle \frac{d^2y}{dx^2}+\frac{dy}{dx}+y=0$$ 
# 
# then this is analytic at $x$ = 0, as $P$ and $Q$ are both 1. 
# 
# The equation 
# 
# $$\displaystyle \frac{d^2y}{dx^2}+\frac{x}{x-1}\frac{dy}{dx}+\frac{1}{x-1}y=0$$ 
# 
# is also analytic at $x$ = 0, and an expansion about 0 is possible. The point $x$ = 1 is singular, and the function $1/(x - 1)$ is then not defined, so the series expansion will not be possible here and the series solution is valid only in the range -1 $\lt x \lt$ 1. In contrast the equation 
# 
# $$\displaystyle \frac{d^2y}{dx^2}+x(x-1)\frac{dy}{dx}+(x-1)y = 0$$
# 
# can be expanded, because the $x(x - 1)$ and $(x - 1)$ are polynomials and every point is normal.
# 
# Bessel's equation is 
# 
# $$\displaystyle x^2\frac{d^2y}{dx^2}+x\frac{dy}{dx}+(x^2-n^2)y=0$$
# 
# where $n$ is a constant and when rearranged is 
# 
# $$\displaystyle \frac{d^2y}{dx^2}+\frac{1}{x}\frac{dy}{dx}+\frac{(x^2-n^2)}{x^2}y=0$$
# 
# so has a singular point at $x$=0. In this case, the equation can be shown to have solutions 
# 
# $$\displaystyle y=x^m \sum \limits_{k=0}^\infty  a_kx^k$$
# 
# and where $a_0 \ne$ 0. Using this series is often called Frobenius’ method and both the index $m$ and coefficients $a_k$ have to be determined. The method is very similar to that already described and is given in more advanced textbooks (Boas 1983; Jeffery 1990; Bronson 1994; Aratyn & Rasinariu 2006).

# In[ ]:




