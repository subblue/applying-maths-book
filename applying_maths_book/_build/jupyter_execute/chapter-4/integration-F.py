#!/usr/bin/env python
# coding: utf-8

# # 10 Multiple integrals 

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
init_printing()                         # allows printing of SymPy results
plt.rcParams.update({'font.size': 16})  # set font size for plots


# ## 10 Introduction
# Many functions contain two or more variables; for instance, the distance of the electron in a hydrogen atom depends on its x, y, and z-position from the nucleus. Any point on a plane is given by its x- and y-coordinates or on the earth by its latitude and longitude. To evaluate a double or triple integral, each integral is performed with the same rules as a single integral but care must be taken to sort out the integration limits. The use of multiple integrals quite often involves transforming variables, from Cartesian $x, y, z$ to different forms of polar or cylindrical coordinates depending on the problem. The only reason for doing this is to simplify the integration and the only difficulty is unfamiliarity with the new coordinates.
# 
# Integrating over a surface such as $f (x, y)$ with the double integral $\int\int f (x, y)dydx$ means integrating along the x- and y-axes to obtain the volume between the surface and the $x-y$ plane. The extent of the integration is determined by the limits to the integration in the $x-y$ plane. Figure 22 shows the case when the limits are constants and then the integral is written as
# 
# $$\displaystyle A=\int_a^b\int_c^d f(x,y)dydx \tag{49}$$
# 
# ![Drawing](integration-fig22.png)
# 
# Figure 22. A double integral pictured as the volume between a surface, shown as a blue mesh, and the $x-y$ plane, shown as a circle.
# _____
# The volume produced is like a rectangular rod with a flat base but a top cut to the shape of the function. The next integral is easily evaluated as two separate ones because the integration limits are constants and the terms can be separated. Note how the integral signs and the variables are written - inside to inside and first to last:
# 
# $$\displaystyle \int_0^\pi\int_0^{\pi/2} \cos(\theta)\sin(\phi)d\theta d\phi =\int_0^\pi \sin(\phi) d\phi\int_0^{\pi /2} \cos(\theta)d\theta=2$$
# 
# In quantum mechanics, double and triple integrals often have the form where one coordinate's function is unity and integration is made over some range of angles; $\varphi$ ranges from $0 \to \pi$ in the following integral
# 
# $$\displaystyle \int_0^\pi d\varphi \int_0^{\pi/2} \cos(\theta) \sin(\theta) d\theta=\pi\int_0^{\pi/2} \cos(\theta) \sin(\theta)d\theta=\frac{\pi}{2}$$
# 
# ![Drawing](integration-fig23.png)
# 
# Figure 23. Limits to the double integral in the $x-y$ plane. The integral is th volume above the shaded area and towards the reader and extends to the function $f(x,y)$.
# _____
# More interesting is the case when the double integral is bounded in the plane by variable limits. As shown in figure 23, $x$ has limits $a \to b$ but $y$ has limits which are the functions $g$ and $h$, which both depend on $x$. As the $y$ limits are not constant the integral is written as
# 
# $$\displaystyle \int_R\int f(x,y)dydx \equiv \int_a^b\left( \int_{g(x)}^{h(x)} f(x,y)dy \right)dx \tag{50}$$
# 
# Notice also the order of integration. As written, the integration on $y$ must be performed first and the result is a function of $x$ and is integrated last. The $y$ integration has to have limits depending on $x$ or ones that are constant. It cannot have limits depending on y because the result could not be integrated by $x$.
# 
# Suppose that $f=\ln(x)$ and the two functions $g$ and $h$ are $g=x^2$ and $h=2-x^2$ and the $x$ integration range is from $-1 \to 1$, as shown in Figure 24. The double integral is
# 
# $$\displaystyle \int_R\int f(x,y)dxdy=\int_{-1}^1\int_{x^2}^{2-x^2}\ln(y)dydx \tag{51}$$
# 
# The inner integral has to be found first which will give a result in $x$. This is then integrated. The inner integral is a standard one and using Sympy to do the calculation and put in limits gives

# In[2]:


x,y = symbols('x,y', positive = True)
ansy = integrate(log(y),(y,x**2,2-x**2))  


# Integrating this result with $x$ from $-1 \to 1$ gives

# In[3]:


simplify( integrate(ansy,(x,-1,1) ) )


# and this result can also be expressed using $\displaystyle 2\tanh^{-1}(x)=\frac{\ln(1+x)}{\ln(1-x)}$ as $\displaystyle \int_{-1}^1\int_{x^2}^{2-x^2}\ln(y)dydx=-\frac{64}{9}+\frac{16}{3}\tanh^{-1}\left(\frac{1}{\sqrt{2}} \right)$.
# 
# ## 10.1 Mean values, moments of inertia, and centroids
# 
# The double integral is useful in obtaining the mean value, centroids and moments of inertia of functions and is an alternative way to that described in Section 8.2. In these equations the order of integration is not necessarily implied by the way in which they are written. The integration order is unimportant if the integration limits are constants but is if they depend on x then the y integral must be done first as in equation 51.
# 
# The integral
# 
# $$\displaystyle A=\int\int 1\,dxdy \qquad\tag{52}$$
# 
# is a _volume_ of unit thickness. The result $A$ is also called a _lamina_, which is a sheet of unit but uniform thickness.
# 
# The _x-centroid_, the average position of the x-coordinate of the function is 
# 
# $$\displaystyle \langle x\rangle =\frac{1}{A}\int\int x\,dxdy \qquad\tag{53}$$
# 
# and similarly for the y-coordinate $\displaystyle \langle y\rangle =\frac{1}{A}\int\int y\,dxdy$. 
# 
# If the density is not constant but is described by a function $f(x, y)$ then the formulae are changed to;
# 
# $$\displaystyle  \langle x\rangle =\frac{1}{A}\int\int xf(x,y)\,dxdy,\quad \langle y\rangle =\frac{1}{A}\int\int yf(x,y)\,dxdy,\quad A=\int\int f(x,y)\,dxdy \qquad\tag{54}$$
# 
# which is essentially equation 28. The position of the centroids is often called the _centre of gravity_ of the object.
# 
# By definition, the moment of inertia of an object about an axis is the product of the distance squared from the axis times the mass; $I = mL^2$. If the body is extended, then the integration must be performed over the entire shape. Should the mass is distributed as $f (x, y)$ then the x-direction moment of inertia depends on the distance from the y-axis and is
# 
# $$\displaystyle  I_x=\int\int y^2f(x,y)dydx \qquad\tag{55}$$
# 
# The y-direction moment is calculated similarly using $x^2$, 
# 
# $$\displaystyle I_y=\int\int x^2f(x,y)dxdy $$
# 
# If the mass is uniform then the function f is a constant and can be taken outside the integration.
# 
# These various calculations are now illustrated. To calculate the lamina's area using equation 52 the limits have to be defined. In Figure 24 the closed area is that bounded by $g = x^2$ and $h = 2 - x^2$ from $x = -1 \to 1$ and these will determine the integration limits. Notice that the $y$ integral is performed first which produces a result in $x$ that is then integrated. The area is calculated as
# 
# $$\displaystyle A=\int_{-1}^1\int_{x^2}^{2-x^2} dydx =\int_{-1}^1\left( y\,\bigg|_{x^2}^{2-x^2}\right)dx =\int_{-1}^1 2(1-x^2)dx=\frac{8}{3}$$
# 
# The centroids are
# 
# $$\displaystyle  \langle x\rangle =\frac{1}{A}\int_{-1}^1\int_{x^2}^{2-x^2}x  \, dydx =\frac{1}{A}\int_{-1}^1 x \left( y\,\bigg|_{x^2}^{2-x^2}\right)dx =0 $$
# 
# $$\displaystyle  \langle y\rangle =\frac{1}{A}\int_{-1}^1\int_{x^2}^{2-x^2}y\,  dydx =\frac{1}{A}\int_{-1}^1  \left( \frac{y^2}{2}\,\bigg|_{x^2}^{2-x^2}\right)dx =1$$
# 
# as would be expected for this symmetrical shape.
# 
# ![Drawing](integration-fig24.png)
# 
# Figure 24. Area bound by $x^2$ and $2 - x^2$ between $-1$ and $1$ is shown as the solid line. Rotation about $x$ and $y$ is indicated illustrating that the x-axis moment of inertia will be largest. The centroid ($0,\,1$) is marked.
# 
# ____
# 
# The moments of inertia of an object, even with some symmetry, are generally quite different along its various axes and this is the case for the $x$ and $y$ moments of inertia of the lamina Figure 24. Using equation 55 with $y^2$ or $x^2$ as necessary and assuming uniform density of $1,\, f (x, y) = 1$ then the moments of inertia $I_y$ and $I_x$ about the y- and x-axes are respectively, $\displaystyle I_y=\int_{-1}^1 \int_{x^2}^{2-x^2} x^2 dydx,\; I_x=\int_{-1}^1 \int_{x^2}^{2-x^2} y^2 dydx$.

# In[4]:


x,y = symbols('x,y',positive=True)
Ix=integrate( integrate( y**2, (y,x**2,2-x**2)  ),(x,-1,1)   )
Iy=integrate( integrate( x**2, (y,x**2,2-x**2)  ),(x,-1,1)   )
print(Ix,';',Iy)


# The moment of inertia about the x-axis is greater (by approximately $6$ times) than that about $y$ because the rotation about $x$ involves the whole body rotating around this axis whereas the body is symmetrically disposed about the y-axis and therefore the moment of inertia is smaller. The moments of inertia of molecules are described in Chapter 7.15.
# 
# ## 10.2 Triple integrals
# 
# Triple integrals are calculated in a similar way to double ones, but instead of producing a volume a density function is produced. This might literally be density if the mass/unit volume of a solid is known, but generally, density is taken to mean the amount of 'stuff' in a given volume such as electron density or probability of being at a certain position in an atomic orbital.
# 
# One commonly met triple integral is the normalization of a wavefunction $\psi(r, \theta, \phi)$; when integrated over all space the product $\psi^* \psi$ must be unity. This means that
# 
# $$\displaystyle \int_0^{2\pi}\int_0^\pi\int_0^\infty \psi(r, \theta, \phi)^* \psi(r, \theta, \phi) r^2\sin(\theta)drd\theta d\phi =1 \qquad\tag{56}$$
# 
# where the superscript * indicates a complex conjugate. The $r^2\sin(\theta)drd\theta d\phi$ term comes from converting the volume element $dxdydz$ in Cartesian to spherical polar coordinates; see Section 11 where the calculation of this conversion is described. The limits are almost invariably the same for quantum problems. The polar angle $\theta$ ranges from north to south and can only have values from $0 \to \pi$. The azimuthal (equatorial) angle $\phi$ moves around the equator so ranges from $0 \to 2\pi$.
# 
# One of the $3$p atomic orbitals has quantum numbers $n = 3,\, s = 1, \, m = 1$, and is
# 
# $$\displaystyle \psi_{311}=N\frac{r}{a_0}\left( 6-\frac{r}{a_0} \right)e^{-r/3a_0}\sin(\theta)e^{i\phi} \qquad\tag{57}$$
# 
# where $N$ is the normalization constant we want to find. Because $r, \,\theta, \, \phi$ are separate there being no term in a product such as  $\theta\phi$ when calculating equation (4.56), the integrals in $r,\, \theta$, and $\phi$ can be treated separately. The integral of $\psi^*\psi$ just in $r$ is
# 
# $$\displaystyle  \frac{N^2}{a_0^2}\int_0^\infty r^2\left( 6-\frac{r}{a_0} \right)^2e^{-2r/3a_0}r^2= \frac{19683}{8}a_0N^4 \qquad\tag{58}$$
# 
# where the second $r^2$ comes from the volume element, eqn 56. This integral has a standard form, (two terms of the type $x^ne^{-ax}$) which can be integrated by parts; see (2.13). Using Sympy, because the calculation while straightforward is involved, gives for the radial part equation 58

# In[5]:


a0,r,N=symbols('a0,r,N',positive=True)
eq= r**4*(6-r/a0)**2*exp(-2*r/(3*a0))
(N/a0)**2**2*integrate(eq,(r,0,oo))


# The angular part of the integral 
# 
# $$\displaystyle \int_0^{2\pi}d\phi\int_0^\pi \sin(\theta)e^{i\phi} \sin(\theta)e^{-i\phi} \sin(\theta) d\theta d\phi$$
# 
# are simplified by separating the $\theta$ and $\phi$ integrals into two then evaluating the complex conjugate first, because $e^{-i\phi}e^{i\phi} = 1$. The remaining integral is
# 
# $$\displaystyle \int_0^{2\pi}d\phi\int_0^\pi\sin^3(\theta)d\theta = 2\pi\int_0^\pi\sin^3(\theta)d\theta =\frac{8}{3}\pi$$
# 
# and the sine integral was worked out by converting to the exponential form. Multiplying the two results and rearranging gives the normalization as $\displaystyle N=\frac{1}{81\sqrt{\pi}}\sqrt{\frac{1}{a_0^3}}$
# 
# ## 11 Change of variables in integrals: Jacobians
# 
# In Section 3 the method of simplifying an integration by a change of variable was described. A commonly used change of variables in multiple integrals is from Cartesian either to plane polar or spherical polar coordinates. The (plane) polar coordinates are two dimensional and spherical polar are three dimensional; see Chapter 1.6.1. They are used only to simplify a calculation by using those coordinates that reflect the underlying symmetry of the problem being studied, thus the shapes of the s, p, d and other atomic wavefunctions (orbitals) are naturally described in terms of three-dimensional spherical polar coordinates with a radius $r$, a polar $\theta$, and an equatorial (azimuthal) angle $\phi$. However, many two or three or higher dimensional integrations can be simplified by a suitable algebraic substitution, which may also be thought of as a change of coordinates. Fortunately, there is a systematic way of doing this using a determinant of derivatives, called the Jacobian and these are described in this section. Determinants are described in Chapter 7.
# 
# A one-dimensional example is considered first. An apparently hard integral such as
# $\displaystyle \int_0^b x\sqrt{ a^2 -x^2} dx$ can be simplified by substituting $u = a^2 - x^2$, calculating the differential $du=-2xdx$ and changing the limits. The result is 
# 
# $$\displaystyle -\frac{1}{2}\int_{a^2}^{a^2-b^2}\sqrt{u}du = \frac{1}{3}(a^3-(a^2-b^2)^{3/2} )$$
# 
# Ignoring the limits for clarity, a general integral of a function $f(x)$ and its substitution
# can be written as
# 
# $$\displaystyle \int f(x)dx= \int F(u)\frac{dx}{du}=\int F(u)\,J(x,u)du$$ 
# 
# where $F$ is the function $f$ in the new variable $u$ and the new function $J$ contains the terms needed to 'distort' $dx$ into $du$. This is done by using the differential $dx = J(x, u)du$  where 
# 
# $$\displaystyle J(x,u)=\frac{dx}{du}=-\frac{1}{2x}=-\frac{2}{2\sqrt{a^2-u}}$$
# 
# A two-dimensional integral in its general form with a change of coordinates is
# 
# $$\displaystyle \int \int f(x,y)dxdy=\int\int F(u,v) \,J(x,y,u,v)\,dudv$$
# 
# where $f$ is some normal function of $x$ and $y$, perhaps $\sin(y)/\sin(x)$ and $u$ and $v$ are functions of $x$ and $y$. What these are depends on the particular calculation. In three dimensions, the general equation for the transformation is similar but rather formidable,
# 
# $$\displaystyle \int \int\int   f(x,y,z)dxdydz=\int\int\int F(r,\theta,\phi) \,J(x,y,z,r,\theta,\phi )\,dr d\theta d\phi$$
# 
# ![Drawing](integration-fig25.png)
# 
# Figure 25. An area in $dxdy$ and morphed to an equal value $rdrd\theta$ in Cartesian and plane polar coordinates. Infinitesimal lengths $dx$ etc. are greatly exaggerated relative to the axes. $\theta$ is the angle formed by moving anticlockwise from zero degrees starting at the horizontal line.
# _____
# 
# The change of coordinates means that the volume $dxdydz$ has to be distorted or morphed into an equivalent volume in the new coordinates $drd\theta d\phi$. Therefore the new function, the _Jacobian_ $J(x, y, z, r, \theta, \phi)$, has to be found.
# 
# Any coordinate change or substitution has three parts
# 
# **(a)** Calculating the Jacobian,
# 
# **(b)** Substituting the new variables into the function $f$,
# 
# **(c)** Changing any limits on the integration to the new coordinates.
# 
# These are best illustrated with examples. A point ($x, y, z$) is equivalently ($r, \theta, \varphi$) in _spherical polar_ coordinates, the connection between the two sets of coordinates is described by geometry and is
# 
# $$x=r\sin(\theta)\cos(\varphi),\qquad y=r\sin(\theta)\sin(\varphi),\qquad z=r\cos(\theta)$$ 
# 
# In _plane polar_ or just _polar_ coordinates the point ($x, y$) is represented as ($r, \theta$) with
# 
# $$\displaystyle x=r\cos(\theta), \qquad y=r\sin(\theta)$$
# 
# In this case the area element $dxdy$ becomes $rdrdθ$ and these are shown in Fig. 4.25.
# 
# In the case of the polar coordinates the area is relatively easily calculated. The circumference of a circle is $2\pi r$, which is the radius times the angle rotated which is $2\pi$ radians. The length of the arc for a small angle is therefore $rd\theta$ for angular change $d\theta$. The radius extends from $r \to r + dr$ making the area $rdrd\theta$. For other coordinates, the geometrical calculation is complex and an algebraic method is therefore preferred. This method, presented without proof, is to form the _Jacobian_, which is the determinant of the derivatives of the equation converting one set coordinates into the other.
# 
# Consider now the spherical polar coordinates, the function $J(x, y, z, r, \theta, \varphi)$ is needed
# and changing to the conventional notation this is the determinant of the partial derivatives of $x, y, z$ with $r, \theta, \varphi$ and is defined as
# 
# $$\displaystyle  \qquad\qquad J\left( \frac{x,y,z}{r,\theta,\varphi} \right) \equiv \frac{\partial(x,y,z)}{\partial(r,\theta,\varphi)} = 
# \begin{vmatrix}
#    \displaystyle \frac{\partial x}{\partial r} & \displaystyle\frac{\partial x}{\partial \theta} &\displaystyle \frac{\partial x}{\partial \varphi} \\
#     \displaystyle\frac{\partial y}{\partial r} & \displaystyle\frac{\partial y}{\partial \theta} & \displaystyle \frac{\partial y}{\partial \varphi} \\
#     \displaystyle\frac{\partial z}{\partial r} & \displaystyle\frac{\partial x}{\partial \theta} & \displaystyle \frac{\partial z}{\partial \varphi} \\
#     \end{vmatrix}  \qquad\qquad\qquad\qquad\qquad\qquad\text{(59)}$$
#     
# Notice the ordering; the old coordinates $x, y, z$ are on the top of each differentiation. Note also the notation in the brackets with $J$. Using equation 59 the determinant is  
# 
# $$J\left( \frac{x,y,z}{r,\theta,\varphi} \right)= \begin{vmatrix}
#    \sin(\theta)\cos(\varphi) & r\cos(\theta)\cos(\varphi) & -r\sin(\theta)\sin(\varphi) \\
#    \sin(\theta)\sin(\varphi) & r\cos(\theta)\sin(\varphi) & r\sin(\theta)\cos(\varphi) \\
#     \cos(\theta) & -r\sin(\theta) & 0 \\
#     \end{vmatrix}  =r^2\sin(\theta)$$
#     
# The volume element conversion is then written as
# 
# $$\displaystyle dxdydx=r^2\sin(\theta)dr d\theta d\varphi \qquad\tag{60}$$
# 
# In some cases, the determinant may produce a negative answer depending on the order of calculating the derivatives; however, the Jacobian represents an area or volume element so the positive result may legitimately be taken in such cases.

# A few examples are now worked through.
# 
# ### **(i) A coordinate change may simplify**
# A coordinate change will simplify the integral 
# 
# $$\displaystyle \int\int \frac{1}{\sqrt{(x-y)^2+2(x+y)+1}}dxdy$$
# 
# The changes are $x=u(1+v),\, y=v(1+u)$. The general form of the equation is 
# 
# $$\displaystyle \int\int f(x,y)dxdy=\int\int F(u,v)\frac{\partial(x,y)}{\partial (u,v)}dudv$$
# 
# The first two steps in the calculation are necessary because no limits are given. In step (a) the Jacobian is calculated and is
# 
# $$\displaystyle \frac{\partial(x,y)}{\partial (u,v)}= \begin{vmatrix} 1+v & u\\v& 1+u \end{vmatrix}=1+u+v$$
# 
# Step (b) is substituting into the function to find $F(u, v)$ and this produces
# 
# $$\displaystyle \frac{1}{\sqrt{(x-y)^2+2(x+y)+1} }= \frac{1}{\sqrt{(u-v)^2+2(u+v)+4uv+1}}=\frac{1}{1+u+v}$$
# 
# and multiplying this with the Jacobian makes the integral rather simple:
# 
# $$\displaystyle  \int\int \frac{1}{\sqrt{(x-y)^2+2(x+y)+1}}dxdy=\int\int1 dudv=uv +c$$
# 
# ### **(ii) Covert to plane polar coordinates**
# The integral 
# 
# $$\displaystyle I=\int_0^1\int_0^\sqrt{1-x^2} x(1-xy^3)dydx$$
# 
# can be solved by converting to (plane) polar coordinates. The limits are converted first. As $x$ extends from $0 \to 1$ so does $r$ as this is the radius in polar coordinates. As $x = r\cos(\theta)$ and $y = r\sin(\theta)$ then $r^2 = x^2 + y^2$, which represents a circle. The maximum value $r = 1$; therefore $x^2 + y^2 = 1$ and integration is around the first quadrant of a circle of unit radius, in the new coordinates the integration area is a rectangle where $r$ ranges from $0 \to 1$ and $\theta$ from $0 \to \pi/2$. The angle $\theta$ varies from $0 \to \pi/2$ and $r$ from $0 \to 1$, Figure 26, making, 
# 
# $$\displaystyle \int_0^1\int_0^\sqrt{1-x^2} x(1-xy^3)dydx =\int_0^1\int_0^{\pi/2} \left(r\cos(\theta)-r^2\cos^2(\theta)r^3\sin^3(\theta)\right)r\,d\theta dr$$
# 
# ![Drawing](integration-fig26.png)
# 
# Figure 26. The integration area in the $x-y$ and $r-\theta$ planes, $r$ ranges from $0 \to 1$ and $\theta$ from $0 \to \pi/2$.
# ______
# 
# Evaluating the integral in $\theta$ first produces $\displaystyle r^2-\frac{2}{15}r^6$ which is easily integrated to five $\displaystyle I=\frac{11}{35}$. The angular integral using Sympy is

# In[6]:


x,y,r,theta=symbols('x,y,r,theta',positive =True)

eq= r**2*cos(theta)-r**6*cos(theta)**2*sin(theta)**3

integrate( eq,theta)


# ### **(iii) Change to polar coordinates**
# The following integral will be solved by transforming to polar coordinates 
# 
# $$\displaystyle I= \int_0^\infty\int_0^x x^2e^{-x^2-y^2}dxdy$$
# 
# Using equation 59 gives,
# 
# $$\displaystyle \frac{\partial(x,y)}{\partial(r,\theta)} = \begin{vmatrix}\cos(\theta) & -r\sin(\theta) \\\sin(\theta)  & r\cos(\theta) \\ \end{vmatrix} =r$$
# 
# as determined also by the geometrical argument in Figure 25. A limit of $x$ means that this varies as $y = x$ a line with a gradient of one or at $45^\mathrm{o}$ to the x-axis which is the same as $\theta = \pi/4$. The integration is therefore in the area from $\theta = 0 \to \pi/4$ and with $r$ extending from $0 \to \infty$.
# 
# Substituting into the integral gives 
# 
# $$\displaystyle \int_0^\infty \int_0^x x^2e^{-x^2-y^2}dxdy =\int_0^\infty \int_0^{\pi/4} r^2\cos^2(\theta)e^{-r^2}rdrd\theta$$
# 
# which can be separated into integrals in $r$ and $\theta$, since there is no term in both variables, and the limits of the integration are constants. The integrals are standard ones, see Section 2.13. Using Sympy gives $\displaystyle I=\frac{1}{8}\left( \frac{\pi}{2}-1  \right)$. The calculation is shown below without limits.

# In[7]:


r,theta=symbols('r,theta',positive=True)
eqr = r**3*exp(-r**2)
ans_r = integrate(eqr,r )
eqt = cos(theta)**2
ans_theta = integrate(eqt,theta )
trigsimp( expand(ans_r*ans_theta) )


# ![Drawing](integration-fig27.png)
# 
# Figure 27 Example (iv). Integration limits in the $\theta - r$ plane. 
# _____
# ### **(iv) A second polar coordinate example**
# The integral $\displaystyle I = \int_0^3\int_0^1\frac{x^2}{(x^2+y^2)^{5/2}}dxdy$
# 
# can be solved by converting to polar coordinates first then changing the integration limits. Converting produces  
# 
# $$\displaystyle I = \int\int\frac{\cos^2(\theta)}{r^2}dr d\theta$$
# 
# Changing the limits is more involved in this example. The initial values are a rectangular shaped area bounded by $x=0 \to 1$ and $y=1 \to 3$. When $x$ is zero $\theta=\pi/2$,and the boundary line $y = 1$ becomes $r = 1/\sin(\theta)$, the line $y = 3, r = 3/\sin(\theta)$ and $x = 1, r = 1/\cos(\theta)$ and the integration area in the $r  - \theta$ is that shape enclosed by these curves as shown in Figure 27. (See Dence 1975, p. 109.)
# 
# The integration has to be split into two parts because CD is sloping: the areas ABCE and CDE. The coordinates of the points are the intersections of their respective curves, except E, which is at point $\left(\tan^{-1}(3), 1/\sin(\,\tan^{-1}(3)\,)\right )$. The integration limits for $r$ are $1/\sin(\theta) \to 3/\sin(\theta)$ and for area ABCE, $\theta = \tan^{-1}(3) \to \pi/2$ and for CDE, $\theta = \pi/4 \to \tan^{-1}(3)$. 
# 
# The calculation in Sympy is shown below and is in two parts The result for the first part is 
# 
# $$\displaystyle \int_{\pi/4}^{\tan^{-1}(3)} \int_{1/\sin(\theta)}^{1/\cos(\theta)} \frac{\cos^2(\theta)}{r^2} dr d\theta =\frac{1}{\sqrt{2}}-\frac{16\sqrt{10}}{75}$$

# In[8]:


r, theta = symbols('r, theta', positive = True)
eq = cos(theta)**2/r**2
integrate(integrate( eq, (r, 1/sin(theta), 1/cos(theta) )), (theta, pi/4, atan(3)) )  # Double integral


# and for the second part $\displaystyle \int_{\tan^{-1}(3)}^{\pi/2} \int_{1/\sin(\theta)}^{3/\sin(\theta)} \frac{\cos^2(\theta)}{r^2} dr d\theta =\frac{\sqrt{10}}{450}$ making the result $\displaystyle\frac{1}{\sqrt{2}}-\frac{19\sqrt{10}}{90}$.

# In[9]:


integrate(integrate( eq, (r,1/sin(theta),3/sin(theta)) ), (theta,atan(3),pi/2) )  

