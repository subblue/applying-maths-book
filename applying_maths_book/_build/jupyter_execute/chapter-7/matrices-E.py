#!/usr/bin/env python
# coding: utf-8

# # 9 Matrices in optics and designing laser cavities

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
sp.init_printing()                      # allows printing of SymPy results in typeset maths format
plt.rcParams.update({'font.size': 14})  # set font size for plots


# ## 9 ABCD matrices.
# Chemists are among the largest group of laser users and it is not often appreciated that, nowadays, not only is nearly all spectroscopy performed with lasers, but also that most chemical kinetics measurements use lasers to initiate and follow reactions. Lasers are also widely used to produce fluorescence from dye molecules bound to DNA or proteins and to generate intrinsic fluorescence or Raman scattering from tryptophan and tyrosine residues in proteins. Ultra-short femtosecond duration lasers are used to probe the excited state reactions of the chromophores in photosynthetic antennas and reaction centres, to study the photo-isomerization in vision and in bacteriorhodopsin (a light-driven proton pump), and in other chromophore containing proteins such as photoactive yellow and green proteins. Fluorescence excited by femtosecond and picosecond duration lasers in single and two photon events is used to image living cells. Some basic knowledge of optics is now definitely required to do research in many areas of the chemical and bio-sciences. It is essential to be able to understand for example, how to build a stable laser, how much light to deliver as watts/cm$^2$ into a sample, and to be able to work out what lenses or mirrors are needed to collect as much fluorescence as possible from a sample. And to this end, a very simple matrix method of analysing the behaviour of simple or complex optical systems has been developed and is described in this section. The books, Introduction to Matrix Methods in Optics (Gerrard & Burch 1975) and Lasers (Siegman 1986) provide a detailed description of these methods and the latter book considerably more than this.
# 
# The matrix method is based on following a two-component vector through an optical system. The first element of this vector represents the height of a light ray above the (optic) axis and the other element is the angle the ray makes to this axis. The optic axis is the direction the ray takes through the centre of all the optical elements. The angle the ray makes must be small enough, $\lt 6^\text{o}$, to allow the approximations $\tan(\theta) \approx \theta$ and $\sin(\theta) \approx \theta$ to be true, and this is good enough for most applications. This is known as the paraxial approximation. The basis set for the vector is $(y, \theta)$}. The calculation is always contained between pairs of reference planes and you have to decide where to put these based on the problem at hand.
# 
# ![Drawing](matrices-fig36.png)
# 
# Figure 36. Definitions of input and output rays and reference planes.
# ______
# The input ray enters at a height $y_1$ above the optic axis at reference plane 1 and moves at angle $\theta_1$ to the optic axis as shown in figure 36. Using the paraxial approximation, the output ray at reference plane 2 having passed through one or more optical elements is related by a linear transformation, hence a matrix, to the input ray at the input reference plane as
# 
# $$\displaystyle \qquad\qquad\begin{bmatrix}y_2 \\ \theta_2 \end{bmatrix}=\begin{bmatrix} A & B \\C & D  \end{bmatrix}\begin{bmatrix}y_1 \\ \theta_1 \end{bmatrix}\qquad\qquad\qquad\qquad \text{(18)}$$
# 
# where the $ABCD$ matrix is defined always to have a unit determinant
# 
# $$\displaystyle \qquad\qquad\begin{vmatrix} A & B \\C & D  \end{vmatrix} =1 \qquad\qquad\qquad\qquad\text{(19)}$$
# 
# The values of the elements A, B, C, and D determine the optical properties of a single element, such as a thin lens, an air gap, or a curved surface. Although two reference planes determine where to start and end the calculation, when several optical elements are stacked together, additional reference planes can be added to determine where one ABCD matrix starts and the other finishes. When many elements are present, the ABCD matrices are multiplied together to produce a final 2 × 2 matrix whose ABCD values are those of the whole optical assembly. The form of the individual ABCD matrix is tabulated for various individual optical elements; see Gerrard & Burch (1975) or Siegman (1986). The ABCD matrices for three optical elements are:
# 
# $$\displaystyle \begin{array}{ccc} 
# \begin{bmatrix} 1 & 0   \\ -1/f & 1  \end{bmatrix} & 
# \begin{bmatrix} 1 & d/n \\  0 & 1    \end{bmatrix} & 
# \begin{bmatrix} 1 & 0   \\ -2n/r & 1  \end{bmatrix} \\ 
# \text{thin lens} & \text{gap d} & \text{+ve mirror} \end{array}$$
# 
# The gap is just that: an air gap or a piece of glass of length $d$ with refractive index $n$. The refractive index of air is practically 1, and can be taken to be 1 for the purpose of almost all calculations. The refractive index of different glasses depends on their composition but $\approx 1.52$ is a typical value for crown glass. Quartz has a smaller value of $\approx 1.48$ and glasses containing rare earth elements, values of $\approx 1.8$. Water has a refractive index of $1.33$ at room temperature, and liquids with more polarizable atoms such as CCl$_4$ have higher values $\approx 1.6$. The reflection matrix shown is for a positive, concave, or focusing mirror with radius of curvature $-r$. A positive radius of curvature $+r$, by convention, applies to a _negative_ mirror, which is defined when the ray approaches from the left and meets a convex mirror surface. This is the convention shown in figure 37, where the surface is convex to the incoming ray starting at point Q. The radius $+r$ is shown as the line SR, see also  figure 43. The focal length $f$ of a mirror is half the radius of curvature and it is often easier to use this rather than $r$.
# 
# ![Drawing](matrices-fig37.png)
# 
# Figure 37. Ray diagram showing refraction at a lens surface with a radius $r$ made of glass with refractive index $n_2$; the medium outside the lens, say air, has a refractive index $n_1$. The angles and surface curvature are greatly exaggerated in the diagram. (Based on a figure in Gerrard & Burch 1975.)
# _____
# The derivation of each ABCD matrix is not difficult, but it is unnecessary to know how this is done for every optical element. Therefore the principle is illustrated by calculating the matrix for a single lens surface, as shown in figure 37. For each optical element, such as a surface or air gap, one reference plane at some position on the x-axis is chosen at which the input beam must be defined, and a second reference plane positioned where the properties of the output beam are required. In the example below, these two planes, $rp_1, rp_2$ are coincident at the lens surface.
# 
# We aim to find equations describing $y_2$ and angle $\theta_2$ in terms of $y_1$ and $\theta_1$ and do so with a reference plane where the beam intersects the lens and follow the derivation in Gerrard & Burch (1975). 
# 
# By Snell's law of refraction 
# 
# $$\displaystyle n_1\sin(i_1) = n_2\sin(i_2)$$
# 
# where $i_1$ and $i_2$ are, respectively, the angles of incidence and refraction measured from the normal to the surface. At small angles, $n_1i_1 = n_2i_2$. The refractive indices are $n_1$ outside the lens, and $n_2$ for the lens' glass. The angles of a triangle add up to $180^\text{o}$; therefore, by the external angle theorem on triangle $QRS$,
# 
# $$\displaystyle i_1=\theta_1+\alpha =\theta_1 +y_1/r$$
# 
# and on triangle $PRS$ 
# 
# $$\displaystyle i_2=\theta_2+\alpha =\theta_2 +y_2/r$$
# 
# Combining equations gives the equation for $y_1$,
# 
# $$\displaystyle n_1\theta_1+\frac{n_1y_1}{r} =n_2\theta_2+\frac{n_2y_1}{r} \qquad \text{or} \qquad y_1\frac{n_1-n_2}{r}=n_2\theta_2-n_1\theta_1$$
# 
# 
# and using the fact that at the surface  $y_2 = y_1$ the matrix equation for the surface is
# 
# 
# $$\displaystyle \qquad\qquad\begin{bmatrix}y_2 \\n_2\theta_2 \end{bmatrix} = \begin{bmatrix}1 & 0 \\ \displaystyle-\frac{(n_2-n_1)}{r}&1  \end{bmatrix} \begin{bmatrix}y_1 \\n_1\theta_1 \end{bmatrix} \qquad \qquad\qquad\qquad\text{(21)}$$
# 
# Multiplying the angle by the refractive index $n\theta$ in the vector makes all matrices unitary; often the notation $V = n\theta$ is used and this is done because, by Snell's law, $V$ remains unchanged as the ray crosses a boundary from one medium to another. The $ABCD$ matrix for a surface of radius $r$ is
# 
# $$\displaystyle \begin{bmatrix}1 & 0 \\ \displaystyle-\frac{(n_2-n_1)}{r}&1  \end{bmatrix}$$
# 
# with refractive index $n_1 $ on the left of the surface and $n_2$ on the right.
# 
# If there are several optical elements, the effect of a ray passing through several of them is calculated by multiplying their $ABCD$ matrices in order, as shown in figure 38. The reference planes are placed around each optical element but the first and last, which are often gaps, are determined by where the object (ray input) distance and image (ray output) distances are required to be. If a thick lens is present, three thin lens matrices would have to be used, which are those of two surfaces separated by a gap containing the glass. The multiplication order is always that in which the light passes through the elements; for example if $M_1$ is the matrix next to the input reference plane (plane 1), it is the right-hand matrix in the multiplication; similarly, the output matrix is always placed on the left-hand side. In figure 38 the right-hand matrix is an air gap and the properties of the beam are those that exist at the reference plane, in this case some distance from the last lens. This might be the focal point of the lens arrangement if the light entering the first lens is parallel.
# 
# The different properties of the $ABCD$ elements, determine what optical properties we
# observe. This is explained in the next section.
# 
# ![Drawing](matrices-fig38.png)
# 
# Figure 38. Matrices for lens-gap-lens-gap arrangement. The values $(y_1,V_1)$ are measured on the left of the diagram.
# ______
# 
# ## 9.1 Understanding the $ABCD$ matrix
# 
# The elements of the $ABCD$ matrix have the following meaning when each in turn is made
# equal to zero. The other elements are unchanged.
# 
# **(a)** Making $D = 0$ means that the first (left-most) reference plane must be at the focal point so that rays leave the optical elements parallel to one another.
# 
# **(b)** Making $A = 0$ puts the final reference plane at the focal point so the rays enter the optical arrangement parallel to one another.
# 
# **(c)** If $B=0$, this produces the _object-image_ relationship, figure 39. The object is magnified by the amount of element $A$ and brought to a focus at the final reference plane. The number $1/D$ is also the magnification.
# 
# **(d)** Making $C = 0$ ensures that both input and output rays are parallel but with angular magnification $D$. This is the telescopic relationship and the system is called *afocal*.
# 
# As an example of using the matrix methods, the Lens Maker's formula 
# 
# $$\displaystyle \frac{1}{v}+\frac{1}{u}=\frac{1}{f}$$
# 
# will be derived, where $u$ is the object distance from a lens, $v$ the image distance, and $f$ the focal length.
# 
# The $ABCD$ matrix for a thin lens of focal length f is 
# 
# $$\displaystyle \begin{bmatrix} 1 & 0 \\-1/f & 1\end{bmatrix}$$
# 
# and for an air gap of  length $d$ is 
# 
# $$\displaystyle \begin{bmatrix} 1 & d \\ 0 & 1\end{bmatrix}$$
# 
# The total $ABCD$ matrix is calculated and, by making element $B = 0$, produces the thin lens (or lens maker's) formula. A diagram of the arrangement is shown below.
# 
# The matrices are written down in the order of rays passing through the elements and are in the reverse order to that drawn on the diagram; place the matrix of the first gap on the right, to its left that for the lens then final gap $v$ making the $ABCD$ matrix. The matrix with reference plane 1 is always placed on the right.
# 
# $$\displaystyle \begin{align}
# \pmb{ABCD} & =\begin{bmatrix} 1& v\\ 0 & 1\end{bmatrix} \begin{bmatrix} 1& 0\\ -1/f & 1\end{bmatrix}\begin{bmatrix} 1& u\\ 0 & 1\end{bmatrix} \\
#  &\quad\Tiny{\text{   gap after lens}\qquad\text{lens } \qquad \text{   gap before lens} } 
# \end{align}$$
# 
# ![Drawing](matrices-fig39.png)
# 
# Figure 39. Arrangement for calculating object-image relationship for a _thin_ lens.
# _________
# Multiplying by hand gives
# 
# $$\displaystyle \pmb{ABCD}  =\begin{bmatrix} 1& v\\ 0 & 1\end{bmatrix} \begin{bmatrix} 1& u\\ -1/f & -u/f+1\end{bmatrix}= \begin{bmatrix} 1-v/f& u+v(1-u/f)\\ -1/f & 1-u/f\end{bmatrix} $$
# 
# As a check, the determinant should be $1$. The calculation is 
# 
# $$\displaystyle (1 - u/f )(1 - v/f ) + u/f + v/f (1 - u/f ) = 1$$
# 
# When $B = 0$, the object-image relationship applies, then $u + v - uv/f = 0$ and rearranging produces the Lens Maker's formula; 
# 
# $$\displaystyle \frac{1}{v}+\frac{1}{u}=\frac{1}{f}$$ 

# ## 10 The laser cavity
# 
# A laser cavity consists of two or more mirrors with a gain medium sandwiched between them, Figure 40. Usually one of the end mirrors is $100$% reflecting and the other mirror, the output coupler, is partially transmitting. This is typically in the range of $1 \to 10$% transmitting depending on the type of laser. The gain medium can be a gas, a liquid containing a highly fluorescent dye, such as Rhodamine-6, or a solid, usually a glass doped with a small percentage of luminescent ions, such as Nd$^{3+}$ or Ti$^+$. The gain medium is excited either electrically, forming a gas discharge, or from another laser or flash lamp. Most of the fluorescence is lost because it is emitted over all angles, but the laser's cavity mirrors capture some of it, which is then fed back into the gain medium where amplification caused by stimulated emission occurs. For the laser to work, the photons have to pass back and forth in the cavity indefinitely and can only do so if they follow the same path. This is equivalent to saying that the wave-front reproduces itself at the mirrors after each round trip. The wave-front is the shape of the wave inside the cavity and it has the same curvature as that of the mirror as it reaches it because light must be reflected normally off any mirror if it is to return along the same path. The curvature changes inside the cavity to satisfy this normality condition at the end mirrors. As the beam waist is approached, the curved wavefront becomes a plane wave and then changes to the opposite curvature as it moves towards the other mirror. When the laser beam leaves the cavity, its longitudinal profile is that determined by the cavity and it continuously diverges. The transverse profile of the laser intensity is ideally Gaussian, although lasers with other profiles can be made.
# 
# If the cavity is represented by an $ABCD$ matrix $\pmb{M}$, and we will define exactly what this will be shortly, then, as the input and output rays must reproduce one another, we can suppose that the input and output can only be related by some constant factor. Furthermore, this value must be the same for one round trip as for $N$ round trips of the cavity. For one round trip, from equation 18 
# 
# $$\displaystyle \begin{bmatrix} y_2 \\ \theta_2\end{bmatrix}=\pmb{M}\begin{bmatrix} y_1 \\ \theta_1\end{bmatrix}$$
# 
# then for $N$ round trips it is true that
# 
# $$\displaystyle \begin{bmatrix} y_N \\ \theta_N\end{bmatrix}=\pmb{M}^N\begin{bmatrix} y_1 \\ \theta_1\end{bmatrix}\qquad\qquad\qquad\qquad\text{(22)}$$
# 
# therefore some way of finding the $N^{th}$ power of the matrix has to be used. The method is to use a similarity matrix, which is described in Section 13.4 but before doing this a more physical method rather than a purely mathematical one is used. 
# 
# If the wavefront reproduces itself after one pass, then it is some multiple  $\lambda$ of the input, therefore
# 
# $$\displaystyle \begin{bmatrix} y_2 \\ \theta_2\end{bmatrix}=\lambda\begin{bmatrix} y_1 \\ \theta_1\end{bmatrix}$$
# 
# where $\lambda$ is a number. For $N$ round-trips, the number will be $\lambda^N$. Some conditions will have to be placed on $\lambda$, but first its values will be calculated. Combining the last equation with equation 18, gives
# 
# 
# $$\displaystyle \pmb{M}\begin{bmatrix} y_1 \\ \theta_1\end{bmatrix}=\lambda\begin{bmatrix} y_1 \\ \theta_1\end{bmatrix}$$
# 
# which has the form of an eigenvalue - eigenvector equation see section 12.13. Rearranging gives
# 
# $$\displaystyle (\pmb{M}-\lambda\pmb{I})\begin{bmatrix} y_1 \\ \theta_1\end{bmatrix}=0$$
# 
# where $\pmb{I}$ is the $2\times 2$ unit or identity matrix. To find the $\lambda's$, which are called an eigenvalues, the determinant has to be solved;
# 
# $$\displaystyle |\pmb{M}-\lambda\pmb{I}|=0 \qquad \text{or} \qquad \begin{vmatrix} A-\lambda & B \\C& D-\lambda \end{vmatrix}=0$$
# 
# the characteristic equation $(A-\lambda)(D-\lambda)-BC=0$ or $\lambda^2 -(A+D)\lambda+1=0$ because $AD-BC=1$ therefore the eigenvalues are,
# 
# $$\displaystyle \lambda = \frac{A+D \pm\sqrt{(A+D)^2-4} }{2}\qquad\tag{23}$$
# 
# Because the wavefronts must replicate themselves, $\lambda$ cannot be a real number unless it is unity, which is a trivial result; therefore it follows that $\lambda$ is purely imaginary. In this case $(A + D)^2 \lt 4$, which means that
# 
# $$\displaystyle \left| \frac{A+D}{2} \right| \lt 2 \qquad\tag{24}$$
# 
# Now this is surprising; we do not need to know the values of $\lambda$ because only equation 24 needs to be obeyed. If it is obeyed, the wavefronts in the cavity replicate themselves and the cavity is stable; if not it is unstable. To design a laser cavity, the first step is to calculate the values of $A$ and $D$ that make a cavity stable with the mirror radius of curvature that have been chosen and to plot a graph of the stable region. The next step is therefore to work out an $ABCD$ matrix for a cavity.
# 
# ![Drawing](matrices-fig40.png)
# 
# Figure 40. Basic laser cavity
# ______
# 
# The simplest cavity has two mirrors, as shown in Figures 40 and 41. The cavity is split into several regions by defining a reference plane at each mirror, at the edges of each gap and in the gain medium, which is assumed to have flat surfaces, and can be treated as a gap with a different refractive index. The matrices must allow the photons to travel from the edge of the output coupler to the end mirror and back again, but note that the output
# mirror is only counted once. The scheme is shown in Figure 41. It is clear from this that the laser beam is not a parallel beam of radiation but is always diverging or converging depending where it is measured. The same beam profile is observed after focusing by a lens, only expanded or compressed depending on the type of lens. 
# 
# 
# ![Drawing](matrices-fig41.png)
# 
# Figure 41 $ABCD$ matrices for a two-mirror cavity.
# ______
# 
# The $ABCD$ matrix $\pmb{M}$ is made by following a ray from the output mirror and back again;
# the sequence of matrices is
# 
# $$\displaystyle \pmb{M} = G_1G_2G_3M_2G_3G_2G_1M_1\tag{25}$$
# 
# Notice that mirror 1 is on the right and that each mirror is included only once. The gaps are indicated by the $G$ matrices. $G_2$ is the gap produced by the gain medium. To make the calculation simpler, the three gaps can be combined into one. If their optical lengths are $d_1, d_2/n_2$, and $d_3$, the total optical length, assuming the refractive index of air is $1$, is $L = d + d /n + d$ and the combined gap matrix  $\displaystyle \begin{bmatrix} 1 & L \\ 0 & 1 \end{bmatrix}$. This is the same as multiplying the three gap matrices together The matrices for a gap also commute so that $G_1G_2G_3 = G_3G_2G_1$. The overall matrix for the cavity is, therefore,
# 
# $$\pmb{M}= \begin{bmatrix} 1 & L \\0 & 1\end{bmatrix}\begin{bmatrix} 1 & 0 \\-1/f_2 & 1\end{bmatrix}\begin{bmatrix} 1 & L \\0 & 1\end{bmatrix} \begin{bmatrix} 1 & 0 \\-1/f_1 & 1\end{bmatrix} $$
# 
# where the focal length of the output coupler mirror is $f_1$ and that of the high reflector $f_2$. This can easily be multiplied out by hand, but we will use python/SymPy.

# In[2]:


L, f1, f2 = sp.symbols('L, f1, f2')

G = sp.Matrix( [ [1,L],[0,    1] ] )
M1= sp.Matrix( [ [1,0],[-1/f1,1] ] )
M2= sp.Matrix( [ [1,0],[-1/f2,1] ] )

M = G * M2 * G * M1
sp.simplify(M)


# In[3]:


AplusD = sp.simplify( M[0,0] + M[1,1] )
AplusD


# The stability occurs when $|A + D| \lt 2$ and this, obviously, can only be calculated when specific values are used. If the optical length of the cavity is $L = 0.5$ m and the mirrors have focal lengths of $f_1 = 0.1$ and $f_2 = 0.2$  m, then $A+D = -0.5$, which is clearly going to be a stable cavity. This can be appreciated if a sketch is made. If instead, the cavity is lengthened to 1$ $m then $A+D=22$, which will prove not to be a stable cavity as a ray will work its way out after a few round trips, thereby limiting any feedback. The boundary between the stable and unstable cavity occurs in the two-mirror cavity when the cavity length is the sum of the mirrors' radii of curvature, which is twice the sum of their focal lengths. This can be seen from equation (7.25). A quick way to see if a cavity is stable and to work out where the beam-waist is going to be is to draw two circles each with each mirror's radius of curvature and with the centres of the circles separated by $L - r_1 - r_2$. The beam-waist is where the two circles cross.
# 

# ### **10.1 Gaussian Beams and the Complex Beam Parameter $q$**
# 
# The shape of a Gaussian beam determined by diffraction. This can be understood by looking at the diffraction of a plane wave passing through a gap as in figure 41a. The mirrors in a laser act as if they are equal gaps by only reflecting part of the initial plane wave, which is initially supposed to be bigger than the mirror's diameter. Initially the plane wave travels in the $z$ direction and diffracts as it passes through the gap shown on the left in fig. 41a. Its amplitude and shape is successively changed by diffraction at each reflection until the beam has a stable shape. After many of these reflections a Gaussian beam is formed which means that its transverse intensity (in the $x,y$ direction) is $\sim\exp(-r^2/w^2)$ where $r=\sqrt{x^2+y^2}$ is the radial distance from the centre of the beam and $w$ is the beam's radius when the amplitude has fallen to $1/e$. (This is chosen in much the same way as the lifetime $\tau$ is used to define concentration in a first order reaction as $\exp(-t/\tau)$). 
# 
# ![Drawing](matrices-fig41a.png)
# 
# Figure 41a. Diffraction of a plane wave travelling left to right ($z$ direction) and being successively diffracted at a series of slits which are equivalent to plane mirrors separated equally by $d$. (Figure based on fig 5.7 in Demtr&ouml;der (1996).
# ___________________________
# 
# The equation describing the Gaussian beam can be derived from the paraxial wave equation, Demtr&ouml;oder (1996), Siegman(1986). The resulting equation describes the amplitude distribution in the transverse $r$ and longitudinal $z$ directions,
# 
# $$\displaystyle A(r,z)= A_0\left(\frac{w_0}{w}\right)\exp\left(i\left(\frac{2\pi z}{\lambda} +\varphi\right)\right) \exp\left(-r^2\left(\frac{1}{w^2}-\frac{2\pi i}{2\lambda R}\right)\right)\tag{25a}$$
# 
# where $i=\sqrt{-1}$ making the exponential terns complex quantities and in this equation $\lambda$ is the wavelength. The first exponential describes the phase change which occurs along the $z$ axis, the second exponential term which has both real and imaginary parts, shows that in the radial direction the beam decays away as $\exp(-r^2/w^2)$. 
# 
# The intensity of the Gaussian beam is $I(r,z)=A^*(r,z)A(r,z)$ where the * indicates the complex conjugate (replace $i$ with $-i$) and the result is 
# 
# $$\displaystyle I(r,z)= I_0\left(\frac{w_0}{w_z}\right)^2\exp(-2r^2/w_z^2)$$
# 
# where $I_0$ is the intensity at the beam waist. The beam radius $w_z$ is a function of position $z$ and is given by
# 
# $$\displaystyle w_z^2= w_0^2 \left( 1+\left( \frac{\lambda z}{\pi w_0^2}\right)^2\right)  $$
# 
# where $w_0$ is the minimum beam waist, $\lambda$ the wavelength and $z$ the distance along the beam. To calculate $E$ or $I$  the ABCD parameters must be known as explained above. The equations relating these to the beam waist and confocal length are given in the summary in section 10.5 below and in fig. 41b.

# ### **Example.  Volume of laser excitation**
# In a spectroscopic experiment, such as measuring the fluorescence from a probe molecule, we might wish to know the number of molecules that are in the the confocal volume and so work out if, for example, fluorescence can be detected. If the beam waist $w_0$ is known then we can achieve this by working out the volume occupied by the laser form $-z_o$ to $z_0$, or any other distance $\pm z$, and the concentration of the target molecule. As the beam has cylindrical symmetry it is easier to workout the volume using *Cylindrical Coordinates*. This means changing the differential volume element from $dV=dxdydz$ to 
# 
# $$\displaystyle dV = r\,dr\,d\theta\, dz$$
# 
# where $\theta$ is the angle around the beam perpendicular to $z$. Using the intensity $I(r,z)$ given above the volume integral is;
# 
# $$\displaystyle V_{z}=\int_{-z}^z\int_0^{2\pi}\int_0^\infty I(r,z) r\;dr \;d\theta \;dz$$
# 
# the limits on $\theta$ cover the complete circle $0\to 2\pi$ and on $r$ from $0\to\infty$, i.e the entire width of the beam. The integral in $\theta$ is simply $2\pi$, therefore substituting for $I(r,z)$ gives
# 
# $$\displaystyle V_z=2\pi I_0 \int_z^z\int_0^\infty \left(\frac{w_0}{w_z}\right)^2r\exp(-2r^2/w_z^2)\;dr \;dz$$
# 
# and the beam intensity now depends on $r$ and $z$. The integral in $r$ is
# 
# $$\displaystyle \int_0^\infty r\exp(-2r^2/w_z^2)dr$$
# 
# and as $w_z$ does not depend on $r$ it is, in effect, a constant. Substituting  $u = r^2, du=2r dr$ gives 
# 
# $$\displaystyle \frac{1}{2}\int_0^\infty \exp(-2u/w_z^2) du= -\frac{w_z^2}{4}\exp(-2u/w_z^2)\bigg|_0^\infty=\frac{w_z^2}{4}$$
# 
# the remaining integral in $z$ is 
# 
# $$\displaystyle \begin{align}V_z&= \frac{\pi}{2}I_0 \int_{-z}^z\left(\frac{w_0}{w_z}\right)^2 w_z^2 dz\\
# &=2\frac{\pi}{2} \int_0^z\left(\frac{w_0}{w_z}\right)^2 w_z^2 dz\\&= \pi w_0^2z\end{align}$$
# 
# which as it should has the dimensions of volume, or equivalently area $\times$ length. 
# 
# A solution of $1$ mmol has $10^{-3}\cdot 6.023\cdot 10^{23}/ 10^{-3}$ molecules/m$^3$. A typical beam radius $w_0$ is $5\cdot 10^{-6}$ m and as the confocal length $z_0=\pi w_0^2/\lambda$ where $\lambda$ is the laser's wavelength, say $500$ nm, then $z_0\approx 160\;\mu m$ thus in the confocal volume there are $ 6.023\cdot 10^{23}\times 25\cdot 10^{-12} \cdot 160\cdot 10^{-6}\pi\approx 7.5\cdot 10^9$ molecules. 
# 
# Only photons from the region $\pm z_0$ will reach reach the detector as those from any other position will be blocked by not having the correct trajectory to pass through any apertures on the pathway to the detector. As the optical density over the length $\pm z_0$ will be very small, not many molecules are excited and for those that are their fluorescence is emitted over $4\pi$ radians. As only a small fraction of these will be collected, you can imagine that the number of photons detected could easily be no more than one ten thousandth of the number of molecules in the laser volume, nevertheless this number is sufficient and confocal fluorescence microscopy is now a well established technique. If a $1$ mw laser at $500$ nm was used, there would be $\approx 2.5\cdot 10^{15}$ photons/s far more than the number of molecules.

# ### **10.2 Complex beam parameter**
# Equation 25a describes how a Gaussian beam propagates through free space. However, when a thin lens or a mirror is encountered the beam focus position and spot size will change but the beam remains as Gaussian. The ABCD matrix is used to quantify these changes. It is very convenient to replace the terms in the second exponential with what is now called a Complex Beam Parameter $q$. This is done by defining
# 
# $$\displaystyle \frac{1}{w^2}-\frac{2\pi i}{2\lambda R} = \frac{-2\pi i}{2\lambda}\frac{1}{q}$$
# 
# and simplifying gives 
# 
# $$\displaystyle \frac{1}{q}=\frac{1}{R}+i\frac{\lambda}{\pi w^2}\tag{25b}$$
# 
# Specialized texts such as Siegman (1986); Gerrard & Burch (1975); Svelto (1982), Demtr&ouml;der (1996) and Yariv (1975) give the detailed derivations of $q$. At the beam waist where $R=\infty$,
# 
# $$\displaystyle q_0=\frac{\pi w_0^2}{i\lambda}=-i\frac{\pi w_0^2}{\lambda}$$
# 
# and at some position $z$,
# 
# $$\displaystyle q_z = q_0 +z$$
# 
# The complex curvature parameter $q$ can be decomposed into its real and imaginary components producing, see Gerrard & Birch (1975) and the summary below,
# 
# $$\displaystyle q = z- iz_0 = \frac{A-D}{2C}+i\frac{\sin(\theta)}{C}\tag{25c}$$
# 
# $$\displaystyle \frac{1}{q} = \frac{1}{R}+ i\frac{\lambda}{\pi w^2}= \frac{D-A}{2B}+i\frac{\sin(\theta)}{B}\tag{25d}$$
# 
# where $\theta$ is given by 
# 
# $$\displaystyle \cos(\theta) = (A + D)/2, \qquad \sin(\theta)=\sqrt{1-\cos^2(\theta)}\tag{25d}$$
# 
# These angles are obtained from the two eigenvalues $\lambda$ (eqn. 23) which are also given by
# 
# $$\displaystyle \lambda_{1,2}= e^{\pm i \theta}=\cos(\theta)\pm i\sin(\theta) \qquad\tag{25e}$$
# 
# ### **10.3 Using the complex beam parameter**
# 
# When the optical system is described by the ABCD matrix if the complex beam parameter has a value $q_1$ on entering and a value $q_2$ on leaving, $q_2$ is described by,
# 
# $$\displaystyle q_2=\frac{Aq_1+B}{Cq_1+D}\tag{26}$$
# 
# and this equation is used, for example, to calculate the beam profile inside a laser cavity which contains several mirrors. Once a value for $q$ has been obtained the changed beam radius $w$, for example, can be found by using the imaginary part of $q$ eqn. 25b i.e. $\text{Im}(q)=\lambda/(\pi w^2)$
# 
# ![Drawing](matrices-fig41b.png)
# 
# Figure 41b. The Gaussian beam profile showing the beam waist, or neck, $w_0$ and the confocal length, which defines the limit of the focal region, $z_0$. The dotted arcs show typical phase fronts which have a radius $R$  and the dotted straight line shows the asymptotic beam divergence. The equations describing the Gaussian beam are given in the text. The confocal distance $z_0=\pi w_0^2/\lambda$.
# ___________________________

# ### **10.4 Example. The complex beam parameter in a 3 mirror cavity**
# 
# As an example of using the complex beam parameter the beam profile in a three mirror laser cavity is calculated. Initially we might want to place the gain medium at the beam's waist to stand the best chance of getting the laser to work by maximizing the gain. However, this may not always be the best place because if the beam waist is too small the high intensity inside the cavity, which, for example, might be a titanium sapphire crystal may cause this to 'burn'. A radiation intensity of $\gg 10^{10}$ w cm$^{-2}$ can cause breakdown in the crystal or even drill a hole right through it, thus calculating what the beam waist's size and position are likely to be is important.  
# 
# The beam profile can only be calculated when the mirrors focal length and laser length has been chosen. To choose the right mirrors and their positions a stability diagram is needed. This means calculating $(A+D)/2$ until suitable focal lengths and mirror separations are found. This is assumed to have been done and the values used were obtained in this way. 
# 
# The shorter arm of the cavity contains the gain material, perhaps a Ti sapphire crystal or a dye jet which is excited by a pump laser. The matrices for the cavity mirrors are $M_1, M_2, M_3$, and for the gaps $G_1, G_2$.  Using fig. 41c the ABCD matrix $ \pmb M$ is  
# 
# $$\displaystyle \pmb M = G_1M_2G_2M_3G_2M_2G_1M_1$$
# 
# which produces values A,B,C and D. The initial $q$ value is from eqn. 25c,  
# 
# $$\displaystyle q_1= \frac{A-D}{2C}+i\frac{\sin(\theta)}{2C}$$
# 
# where $\theta$ is given by eqn. 25d. The next step is to calculate the radius $w$ over the first gap using $z=0\cdots g_1$. To do this the matrix for a gap is calculated for the first point along $z$ and multiplied by matrix $M_1$, i.e. $G_1(z_1)M_1$. Next a new $q$ calculated using eqn. 26 and from this $w_1$ at position $z_1$ is calculated using just the imaginary part of eqn. 25b. The profile in the second gap is calculated starting at the second mirror with values $z_i - g_1$ thus the matrix is $\pmb M=G_2(z_i-g_1)M_2G_1M_1$, where $G_2(z_i-g_1)$ means calculate the gap matrix with value $z_i-g_1$.  This process is repeated for all $z$ values. As the beam repeats itself the calculation need only go to the cavity length.
# 
# The algorithm is;
# 
# Calculate $q_1$ from the ABCD matrix for the whole cavity using eqn. 25c.
# 
# Loop over $i = 0, 1, 2,\cdots$ for points $z_i$ up to the total length $g_1+g_2$.
# 
# > if $z_i \le g_1$ calculate $G_1$ matrix at position $z_i$ then $\pmb M$ matrix as  $\pmb M_i = G_1(z_i)M_1$
# 
# > else if $z_i \le g_1+g_2$ calculate $G_2$ matrix at position $z_i-g_1.\; \pmb M_i = G_2(z_i-g_1)M_2G_1M_1$
# 
# > calculate new A,B,C,D from $M_i$
# 
# > calculate $q_2$ using eqn. 26. The same $q_1$ is used.
# 
# > calculate $w$ from imaginary part of $q_2$, eqn. 25b
# 
# repeat 
# 
# ![Drawing](matrices-fig41c.png)
# 
# Figure 41c. Left. Labelling for a $3$ mirror laser cavity calculation. Right. Calculated profile displayed as a linear cavity with mirror focal lengths of $25$ mm for $M_3$, $90$ mm for $M_2$ and $\infty$ for $M_1$ with gaps of $150$ and $400$ mm. The beam waist between $M_2$ and $M_3$ is $\approx 0.09$ mm and at the plane mirror $\approx 0.43 $ mm. The vertical red line shows the position of mirror 2.
# __________________________
# 

# In[4]:


# Three mirror cavity. Calculation of beam profile as shown in Figure 41c.
# if you chqnge parameters an unstable cavity may be produced. 
# To find stable regions you will need to plot one mirror radii vs the other
# (keeping the third fixed) or vs cavity length with fixed mirror focal length etc.

#----------------------------------------
def abcd_transform(q_in, M):              #Transforms the q-parameter using an ABCD matrix.
    A = M[0, 0]
    B = M[0, 1]
    C = M[1, 0]
    D = M[1, 1]
    q_out = (A * q_in + B) / (C * q_in + D)
    return q_out
#----------------------------------------

def Cavity_beam_profile(f1, g1, f2, g2, f3, wavelength, num_points):

    # Calculate the round-trip matrix from M1, 
    M_g1 = Gap(g1)
    M_g2 = Gap(g2)
    
    M_f1 = Mirror(f1)
    M_f2 = Mirror(f2)
    M_f3 = Mirror(f3)

    M_round_trip = M_g1 @ M_f2 @ M_g2 @ M_f3 @ M_g2 @ M_f2 @ M_g1 @ M_f1      # matrix 
    
    A = M_round_trip[0, 0] 
    B = M_round_trip[0, 1] 
    C = M_round_trip[1, 0] 
    D = M_round_trip[1, 1]
    
    # Check for stability
    stability_param = (A + D) / 2
    if np.abs(stability_param) > 1:
        print('{:s} {:6.3f}'.format('Cavity is unstable. (A+D)/2 = ', stability_param) )
        return None, None
    
    # if cavity stable continue from here
    print('{:s} {:6.3f}'.format('Cavity is stable. (A+D)/2 = ', stability_param) )
    sinetheta = np.sqrt(1.0 - (A+D)**2/4 )              # sine theta
    q_at_M1 = (A - D)/(2*C) + 1j*sinetheta /C           # complex number
    
    # Calculate the beam profile along the cavity
    
    total_length = g1 + g2                               # only need go half way along cavity to get profile
    z_points = np.linspace(0, total_length, num_points)  # z position in cavity
    w_values = np.zeros(num_points,dtype=float)          # to hold beam radius w
    
    for i in range(num_points):
        z = z_points[i]
        if z <= g1:
            M = Gap(z) @ M_f1                            # Gap_z propagates across gap to position z. 
        elif z <= g1 + g2:
            M = Gap(z - g1) @ M_f2 @ M_g1 @ M_f1
        q_z = abcd_transform(q_at_M1, M)                 # at each z value
        
        #  as 1/q = 1/R + i * lambda / (pi * w^2), solve for w but only use the imaginary part
        inv_q = np.abs(np.imag(1 / q_z ) )               # abs is to ensure sqrt is positive
        w_values[i] = np.sqrt( wavelength / (np.pi * inv_q))
        
    return z_points, w_values
#-------------------------------------

Gap    = lambda d: np.array([[1, d], [0, 1]])            # gap d matrix
Mirror = lambda f: np.array([[1, 0], [-1/f, 1]])         # mirror or lens focal length f matrix

#----------------------------------------
# Cavity parameters 
g1 = 400           # gap  mm
g2 = 150           # mm

f1 = 1e6           # focal length in metres. (This is in effect a flat mirror)
f2 = 90            # mm
f3 = 25            # mm 
wavelength = 500e-9*1000   # mm 
num_points = 500

# Calculate and plot the beam profile
z, w = Cavity_beam_profile(f1, g1, f2, g2, f3, wavelength, num_points)

if z is not None:                                         # stable cavity so plot results
    plt.figure(figsize=(8, 2))
    plt.plot(z,  w , color='blue')
    plt.plot(z, -w , color='blue')
    
plt.axvline(0,color='grey')
plt.axvline(g1,color='grey')
plt.axvline(g1+g2,color='grey')
plt.show()


# ### **10.5 Summary of Gaussian beam equations**
# 
# 
# From a study of the Gaussian beam properties of laser cavities (Gerrard & Burch 1975), the following properties are obtained. The reference plane is labelled RP.
# 
# $$\displaystyle \begin{array}{lll}
# \text{Radius of curvature of the Gaussian beam at RP: }&\displaystyle R=\frac{2B}{D-A}&\qquad\qquad\qquad\\
# \text{Transverse beam radius \omega at RP}  &\displaystyle w=\sqrt{\frac{\lambda B} {\pi\sin(\theta)} }\\
# \text{Location of neck to left of RP} & \displaystyle z = \frac{(A - D)}{2C}\\
# \text{The beam waist at neck} &\displaystyle w_0=\sqrt{\pm\frac{\lambda\sin(\theta)} {\pi C} }\\
# \text{The confocal beam length or Rayleigh range:}&\displaystyle z_0=\frac{\pi w_0^2}{\lambda}\\
# \end{array}$$
# 
# $\lambda$  here is wavelength and $w$ is defined as transverse radius at $1/e$ amplitude. The angle $\theta$ is defined via $\cos(\theta)=(A+D)/2$ and $\sin^2(\theta)+\cos^2(\theta)=1$. 
# 
# The last parameter $z_0$, the confocal beam length, is the length of the region over which the laser is focused in the sense that it is the length either side of the beam waist where the beam increases by $\approx\sqrt{2}$. This is also called the 'near field'. The beam waist at $z$, a position either side of the focus, is given by
# 
# $$\displaystyle w_z^2= w_0^2 \left( 1+\left( \frac{\lambda z}{\pi w_0^2}\right)^2\right)  $$
# 
# This equation is not used to calculate the profile in a laser cavity, the method above should be used, but once $w_0$ or $z_0$ are known $w_z$ can be found. In calculating the beam radii $\omega_z$ and $\omega_0$ the $\pm$ sign is chosen to ensure that the number is real. When cavities that are more complicated are used, it is necessary to use the complex beam parameter to determine the properties of the cavity. 
# 

# In[ ]:




