#!/usr/bin/env python
# coding: utf-8

# ## Solutions Q 93-114

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
init_printing()                         # allows printing of SymPy results in typeset maths format
plt.rcParams.update({'font.size': 14})  # set font size for plots


# ### Q93 answer
# Differentiating with respect to $x$ first 
# 
# $$\displaystyle \frac{\partial z}{\partial x}=2x\sin(y/x)-y\frac{(x^2+y^2)}{x^2}\cos(y/x)$$
# 
# and then with $y$ gives 
# 
# $$\displaystyle \frac{\partial z}{\partial x}=2y\sin(y/x)+ y\frac{(x^2+y^2)}{x}\cos(y/x)$$
# 
# and where brackets are not around the derivatives, for example with $\displaystyle \frac{\partial z}{\partial x}$, there is no ambiguity because $y$ must be constant.
# 
# The mixed derivative does need us to specify which variable is held constant,
# 
# $$\displaystyle \frac{\partial ^2z}{\partial x \partial y}=\left(\frac{\partial z}{\partial x }  \right)_y\left(\frac{\partial z}{\partial y }  \right)_x= \left( 2x\sin(y/x)-y\frac{(x^2+y^2)}{x^2}\cos(y/x) \right)\left( 2y\sin(y/x)+ y\frac{(x^2+y^2)}{x}\cos(y/x) \right) $$
# 
# which can be simplified somewhat. Using SymPy the mixed calculation can be done in one step 

# In[2]:


z, x, y = symbols('z, x, y')
z = (x**2+y**2)*sin(y/x)
simplify(diff(z,x,y) )


# ### Q94 answer
# (a) Differentiating with $x$ twice produces $\displaystyle \frac{\partial z}{\partial x}=e^{x+cy}+\frac{1}{x-cy} \quad \text{ and }\quad \frac{\partial^2 z}{\partial x^2}=e^{x+cy}-\frac{1}{(x-cy)^2} $
# 
# and with $y$ also $\displaystyle \frac{\partial z}{\partial y}=ce^{x+cy}-\frac{1}{x-cy} \quad \text{ and }\quad \frac{\partial^2 z}{\partial x^2}=c^2e^{x+cy}-\frac{1}{(x-cy)^2} $
# 
# Multiplying the second derivative in $x$ by $c^2$ shows that $z$ is a solution to $\displaystyle \frac{\partial^2z}{\partial y^2}=c^2\frac{\partial^2z}{\partial x^2}$
# 
# (b) By 'arbitrary' is meant only that we need not specify exactly how $z$ depends on $x$ and $y$ except that they contain terms in $x + cy$. Taking derivatives;
# 
# $$\displaystyle \frac{\partial z}{\partial x}=\frac{\partial f}{\partial x}+\frac{\partial \phi}{\partial x} \quad \text{ and } \quad \frac{\partial^2 z}{\partial x^2}=\frac{\partial^2 f}{\partial x^2}+\frac{\partial^2 \phi}{\partial x^2}$$
# 
# and
# 
# $$\displaystyle \frac{\partial z}{\partial y}=c\frac{\partial f}{\partial y}-c\frac{\partial \phi}{\partial y} \quad \text{ and } \quad \frac{\partial^2 z}{\partial y^2}=c^2\frac{\partial^2 f}{\partial y^2}+c^2\frac{\partial^2 \phi}{\partial y^2}$$
# 
# then by comparing the second derivative in $y$ with that in $x$ multiplied by $c^2$ shows again that $\displaystyle \frac{\partial^2z}{\partial y^2}=c^2\frac{\partial^2z}{\partial x^2}$.
# 
# ### Q95 answer
# (a) By substitution $\displaystyle \frac{\partial c}{\partial t}=-\frac{\partial J}{\partial x}=-\frac{\partial}{\partial x}\left(-D  \frac{\partial c}{\partial x} \right)$, therefore $\displaystyle \frac{\partial c}{\partial t}=D\frac{\partial^2 c}{\partial x^2}$ which is Fick's second law.
# 
# (b) While the complete expression for $c$ can be differentiated, it is easier to remove the constants and simplify the expression first. Doing this produces 
# 
# $$\displaystyle \frac{\partial c}{\partial t} = \frac{c_0}{\sqrt{4\pi D}}\frac{\partial}{\partial t}\left( \frac{e^{-x^2/4Dt}}{\sqrt{t}} \right)$$
# 
# differentiating and simplifying gives 
# 
# $$\displaystyle  \frac{\partial c}{\partial t}=\frac{c_0}{4\sqrt{\pi Dt^3}}\left( -1+\frac{x^2}{2Dt} \right)e^{-x^2/4Dt}$$
# 
# To differentiate twice with $x$ gives
# 
# $$\displaystyle \begin{align}
# D\frac{\partial^2 c}{\partial x^2}=&\frac{Dc_0}{\sqrt{4\pi Dt}}\frac{\partial^2}{\partial x^2}e^{-x^2/4Dt} \\=& \frac{Dc_0}{\sqrt{4\pi Dt}}\frac{\partial}{\partial x}\left( -\frac{2x}{4Dt} e^{-x^2/4Dt}\right)\\=&\frac{c_0}{4\sqrt{\pi Dt^3}}\left( -1+\frac{x^2}{2Dt} \right) e^{-x^2/4Dt}\end{align}$$
# 
# which proves that the equation for $c$ is a solution to Fick's Second Law as this result is the same as the previous one. Using SymPy we can check the result easily. (it is necessary to use the simplify command to force the simplification.)

# In[3]:


x, D, c0,t = symbols('x, D, c0, t')

f= c0/sqrt(4*pi*D*t)*exp(-x**2/(4*D*t))
if simplify((D*diff(f,x,x)  - diff( f,t))) == 0:
    print('true')
else:
    print('false')


# (c) The concentration equation does not apply at $t = 0$ because here $c$ is infinity and this is not possible as the total concentration is $c_0$ unless the disc of atoms is made infinitesimally thin. In solving the diffusion equations, the complete answer has been approximated and so simplified to obtain the concentration equation quoted above. At $t = 0$ the total concentration must be $c_0$ which is the same at all times because no material is lost.
# 
# (d) As $x= \sqrt{2Dt}$ and if $t = 1\,\mathrm{\mu s}$ then $x \approx 4.5 \cdot 10^{-7}$ m so a scale of microseconds and tens to hundreds of nanometres seems sensible to try. A concentration of $1\,\mathrm{ mol\,dm^{-3}} =0.602$ in  molecules / nm$^3$ .  Plotting produced the following graph, which shows how the molecules all initially at $x = 0$ at $t = 0$, spread out into the solution as time progresses and shows that diffusion really is a slow process. When you notice the smell of coffee or perfume in a room this is probably due to convection of the air rather than diffusion. 
# 
# ![Drawing](differen-fig63.png)
# 
# Figure 63 One-dimensional diffusion of molecules initially placed at $x = 0$ at $t = 0$ is shown at different times in microseconds. The diffusion constant is the similar to that of water, $\approx 2.5\cdot 10^{-9}\,\mathrm{ m^2\,s^{-1}}$. At time zero, the molecules form a $\delta$ function at $x = 0$.
# _______
# 
# ### Q96 answer
# (a) Expanding $\displaystyle \left(p+\frac{a}{V^2}  \right)(V-b)=RT$ produces $pV+a/V-bp-ab/V^2=RT$. 
# 
# At constant $T$ and because $p$ is a function of $V$ 
# 
# $$\displaystyle \frac{\partial}{\partial V}pV=V\frac{\partial p}{\partial V}+p$$
# 
# the first derivative wrt $V$ is
# 
# $$\displaystyle V\left(\frac{\partial p}{\partial V}   \right)_T + p-\frac{a}{V^2} - b\left( \frac{\partial p}{\partial V} \right)_T +\frac{2ab}{V^3}= 0$$
# 
# Simplifying produces $\displaystyle (V-b)\left( \frac{\partial p}{\partial V}\right)_T + p -\frac{a}{V^2}+\frac{2ab}{V^3} = 0$.
# 
# Labelling with a subscript $c$ to indicate the critical points, and because the derivatives are zero here, produces $\displaystyle p_c-\frac{a}{V_c^2}+\frac{2ab}{V_c^3}=0$.
# 
# (b) Starting with the first derivative the second is, at constant $T$,
# 
# $$\displaystyle V\left( \frac{\partial^2 p}{\partial V^2} \right)_T+\left( \frac{\partial p}{\partial V} \right)_T+\left( \frac{\partial p}{\partial V} \right)_T+2\frac{a}{V^3}-b\left( \frac{\partial^2 p}{\partial V^2} \right)_T-\frac{6ab}{V^4}=0$$
# 
# and as these derivatives are also zero at the critical point $V_c=3b$. Substituting into the equations for $p_c$ produces $\displaystyle p_c=\frac{a}{27b^2}$ and $T_c$ can be found by substitution into the van-der Waals equation and is $\displaystyle T_c=\frac{8a}{27bR}$. 
# 
# Doing the same calculation in a different way with SymPy gives

# In[4]:


a,b,R, T , p, V=symbols(' a b R T p V') 

pvdw = R*T/(V-b)-a/V**2   # pressure in vdw equation
dpdv = diff(pvdw,V)       # dpdV
dpdv


# In[5]:


dpdv2 = diff(pvdw,V,V)   # d^2pdV^2
dpdv2


# In[6]:


ans = solve( (dpdv, dpdv2),(T, V) )  # answer solving simultanoeus eqns gives T and V in that order
ans


# ### Q97 answer
# Differentiating $H$ with temperature at constant pressure $p$ produces $\displaystyle \left( \frac{\partial H}{\partial T}\right)_p =  \left( \frac{\partial U}{\partial T}\right)_p +p \left( \frac{\partial V}{\partial T}\right)_p$
# 
# and from the definition of $C_p$ and rearranging gives the required expression $\displaystyle \left( \frac{\partial U}{\partial T}\right)_p=C_p - p \left( \frac{\partial V}{\partial T}\right)_p$
# 
# ### Q98 answer
# At constant pressure, differentiating $S$ with $T$ produces $\displaystyle \left( \frac{\partial S}{\partial T}\right)_p= n\frac{C_p}{T}$.
# 
# Because volume is not explicitly in the equation, the ideal gas law is used to substitute for pressure and so obtain an equation containing volume; 
# 
# $$\displaystyle  S=S+0+nC_p\ln(T)-nR\ln\left( \frac{RT}{V} \right)$$ 
# 
# then 
# 
# $$\displaystyle \left( \frac{\partial S}{\partial T}\right)_V =n\frac{C_p}{T}-n\frac{R}{T}$$
# 
# which by substitution of the first result shows that the two partial derivatives are different; 
# 
# $$\displaystyle \left( \frac{\partial S}{\partial T}\right)_V =\left( \frac{\partial S}{\partial T}\right)_p-\frac{nR}{T}$$
# 
# The plot shows this also as the partial derivatives are horizontal or vertical lines on this plot and are clearly different. For example at constant pressure, say $2.5$ bar the slope at low temperatures is greater than at higher ones as the contours become more widely spaced at the temperature increases, this is in accord with $\displaystyle \left( \frac{\partial S}{\partial T}\right)_p= n\frac{C_p}{T}$. In the plot $C_p= 5R/2$ which is that for a diatomic molecule; $3R/3$ for translation and $R/2$ each for vibrational kinetic and potential energy.
# 
# ![Drawing](differen-fig64.png)
# 
# Figure 64. Contour plot of the entropy with pressure and temperature. ($C_p=5R/2$ and $S_0=0$). The contours are at constant entropy with values labelled in J/mol/K. The gradient $\partial S/\partial p$ at constant $T$ would be a vertical line, $\partial S/\partial T$ at constant $p$ is a horizontal line.
# ______

# ### Q99 answer
# (a) Using the ideal gas law $pV=RT$ therefore  $\displaystyle \left( \frac{\partial p}{\partial T}\right)_V=\frac{R}{V}$ and therefore $\displaystyle \left( \frac{\partial U}{\partial V}\right)_T=\frac{RT}{V}-p=0$. This result is expected because the ideal gas is defined as consisting of hard sphere point molecules with no interaction energy between them; therefore, at constant temperature, the internal energy does not depend upon the volume of the gas.
# 
# (b) The van der Waals gas is defined by $\displaystyle \left(p+\frac{a}{V^2}  \right)(V-b)=RT$. With $V$ as a constant a  $\displaystyle \left( \frac{\partial p}{\partial T} \right)_V =\frac{R}{V-b}$ . Using the equation in the question
# 
# $$\displaystyle \left( \frac{\partial U}{\partial V}\right)_T=\frac{RT}{V-b}-p =\frac{a}{V^2}$$
# 
# and the van der Waals equation was used to simplify the derivative. This confirms that the constant $a$ is related to the interaction energy between molecules; the internal energy becomes large when the volume is small. Conversely, when the volume is large, 
# 
# $$\displaystyle \left( \frac{\partial U}{\partial V}\right)_T \to 0$$
# 
# because the molecules rarely meet one another and the gas becomes 'ideal'.
# 
# ### Q100 answer
# (a) As $pV=nRT$ and $T$ is held constant in an isothermal process then $\displaystyle \left( \frac{\partial V}{\partial p} \right)_T =-\frac{nRT}{p^2}$. 
# 
# Substituting for $RT$ and dividing by $-V$ gives 
# 
# $$\displaystyle \kappa =-\frac{1}{V}\left( \frac{\partial V}{\partial p} \right)_T=\frac{1}{p}$$
# 
# (b) For the van der Waals gas where $\displaystyle \left(p+\frac{a}{V^2}  \right)(V-b)=RT$ expanding the terms to make it easier to differentiate gives 
# 
# $$\displaystyle pV+\frac{a}{V}-pb-\frac{ab}{V^2}-RT=0$$
# 
# Differentiating $V$ wrt $p$ at constant $T$ gives 
# 
# $$\displaystyle V+p\left(\frac{\partial V}{\partial p} \right)_T-\frac{a}{V^2}\left(\frac{\partial V}{\partial p} \right)_T -b+2\frac{ab}{V^2}\left( \frac{\partial V}{\partial p} \right)_T=0$$
# 
# and after some serious rearranging 
# 
# $$\displaystyle \kappa = \frac{(V-b)V^2}{pV^3-aV+2ab}$$
# 
# As a check if $a$ and $b$ are both zero then the result for an ideal gas results; $\kappa = 1/p$.
# 
# ### Q101 answer
# (a) When $t = 0$ the concentration is $\displaystyle c=c_0e^{v(x-x_0)/2D}$. 
# 
# If also $x = x_0$ then the concentration is $c_0$ at zero time, and this is therefore the initial concentration and  $x_0$ is therefore the initial position. Clearly, the units of each term in the equation must be the same. On the left-hand side they are $\mathrm{dm^3\, mol^{-1}\,s^{-1}}$, and the first term on the right is $D\partial^2 c/\partial x^2$ which has units $D \cdot\,\mathrm{ dm^3\, mol^{-1}\, dm^{-2}}$. To make this $\mathrm{dm^3\, mol^{-1}\,s^{-1}}$, $D$ must have units of $\mathrm{dm^2\, s^{-1}}$ or distance$\mathrm{^2\, time^{-1}}$ (or area/time). In solution, large molecules typically have diffusion coefficients of $\approx 10^{-9}\,\mathrm{ m^2\, s^{-1}}$. 
# 
# (b) To show that the equation in the question is a solution, differentiate with respect to $t$ and $x$. Differentiation with respect to $t$ gives 
# 
# $$\displaystyle \frac{\partial c}{\partial t}=-\frac{v^2}{4D}e^{v(x-x_0)/2D} e^{-v^2/4D}$$
# 
# and wrt $x$; 
# 
# $$\displaystyle \frac{\partial c}{\partial x}=\frac{v}{2D}e^{v(x-x_0)/2D}c_0 e^{-v^2t/4D}$$
# 
# and 
# 
# $$\displaystyle \frac{\partial^2 c}{\partial x^2}=\left(\frac{v}{2D}\right)^2c_0e^{v(x-x_0)/2D} e^{-v^2t/4D}$$
# 
# Looking at these three derivatives the exponential terms are the same in each and when the diffusion equation is formed they will all cancel out. Therefore 
# 
# $$\displaystyle -\frac{v^2}{4D}=D\frac{v^2}{4D^2}-v\frac{v}{2D}$$
# 
# which is zero, proving what was required.

# ### Q102 answer
# (a) The first law of thermodynamics states that,for an infinitesimal quasi-statical change of state, a condition often called 'reversible', the change in the internal energy $U$ of an object is the sum of the heat transferred to the object $Q$ and the work done on the object $W$,
# 
# $$\displaystyle dU=dQ+dW$$
# 
# The work done on a gas is $-pdV$ and because $U=U(T)$ is a function of temperature only for an _ideal gas_, then we can state that $\displaystyle dU=\frac{dU}{dT}dT$ and write $dU$ rather than $\partial U$, because $U$ only depends on $T$ and therefore cannot be a partial derivative of anything else. 
# 
# Do not be tempted to differentiate the first law; it is already in differential form. Substituting for $dU$ into the first law equation and rearranging produces 
# 
# $$\displaystyle dQ=\frac{dU}{dT}dT+pdV$$
# 
# The heat capacity of a substance is defined as the heat absorbed per unit change in temperature, thus 
# 
# $$\displaystyle C = \frac{dQ}{dT}$$
# 
# and working under conditions of constant volume means that $dV$ is zero and 
# 
# $$\displaystyle dQ =\frac{dU}{dT} dT$$
# 
# The constant volume heat capacity is therefore $\displaystyle C_V=\frac{dU}{dT}$.
# 
# Next, to calculate the constant pressure heat capacity $C_p$, the change in pressure must be zero, therefore $dp \to 0$. To do this the first law must be recast in terms of $dp$, rather than $dV$. From the ideal gas law $pV = RT$ and for changes in pressure, temperature and volume; $pdV+Vdp=RdT$. Substituting for $pdV$ into the first law produces
# 
# $$\displaystyle dQ=dU+RdT-Vdp = \left(\frac{dU}{dT}+R  \right)dT-Vdp$$
# 
# and so it follows that at constant pressure when $dp \to 0$, 
# 
# $$\displaystyle \frac{dQ}{dT}=\left(\frac{dU}{dT}+R  \right)$$
# 
# and therefore as $\displaystyle \frac{dQ}{dT}$ defines heat capacity $\displaystyle C_p=\frac{dU}{dT}+R$ and $C_p=C_V+R$.
# 
# The heat capacity at constant pressure is larger than at constant volume because the volume must change if the pressure is constant and therefore work has to be done on the gas to make it expand.
# 
# (b) A new function called the enthalpy is always used to take account of any change in volume (and hence work) that occurs at constant pressure and is defined as
# 
# $$\displaystyle H=U+pV = U+nRT$$
# 
# For 1 mole of gas, $n = 1$; differentiating $H$ with respect to $T$ at constant $p$ produces 
# 
# $$\displaystyle \left( \frac{\partial H}{\partial T} \right)_p=\left(  \frac{\partial U}{\partial T}\right)_p+R$$
# 
# and by comparison with the previous definition it follows that 
# 
# $$\displaystyle C_P=\left(\frac{\partial H}{\partial T}  \right)_p$$
# 
# This is usually written as $\displaystyle C_P=\frac{d H}{d T} $ because the constant $p$ is implied with $C_p$.
# 
# ### Q103 answer
# (a) The quantities $dU$ and $dS$ are both functions of $T$ and $V$, and starting with $U$ the total derivative is,
# 
# $$\displaystyle dU=\left( \frac{\partial U}{\partial V}\right)_TdV+\left( \frac{\partial U}{\partial T}\right)_VdT$$
# 
# and similarly 
# 
# $$\displaystyle dS=\left( \frac{\partial S}{\partial V}\right)_TdV+\left( \frac{\partial S}{\partial T}\right)_VdT$$
# 
# Substituting for $dU$ into the equation given in the question to obtain, 
# 
# $$\displaystyle \left( \frac{\partial U}{\partial T}\right)_VdT+\left( \frac{\partial U}{\partial V}\right)_TdV=TdS-pdV$$
# 
# and rearranging for $dS$ gives 
# 
# $$\displaystyle dS= \frac{1}{T}\left[\left( \frac{\partial U}{\partial V}\right)_T +p \right]dV +\frac{1}{T}\left( \frac{\partial U}{\partial T}\right)_VdT$$
# 
# Comparing with the definition of $dS$ in the total derivative, the $dV$ term is 
# 
# $$\displaystyle \left( \frac{\partial S}{\partial V}\right)_T=\frac{1}{T}\left[\left( \frac{\partial U}{\partial V}\right)_T +p \right]$$
# 
# and the $dT$ term 
# 
# $$\displaystyle \left( \frac{\partial S}{\partial T}\right)_V=\frac{1}{T}\left( \frac{\partial U}{\partial T}\right)_V$$
# 
# from which, using the definition of $C_V$ in the previous question, we obtain  
# 
# $$\displaystyle \left( \frac{\partial S}{\partial T}\right)_V=\frac{C_V}{T}$$
# 
# (b) As $H$ is by definition $H=U+pV $ and therefore a function of $U$, $p$, and $V$, then by definition 
# 
# $$dH = dU + pdV + Vdp$$
# 
# Substituting for $dU + pdV$ into the equation given in the question, $dU + pdV = TdS$, gives $dH = TdS + Vdp$. As $H$ and $S$ are each a function of $T$ and $p$, the total derivatives are 
# 
# $$\displaystyle  dH=\left( \frac{\partial H}{\partial p}\right)_Tdp+\left( \frac{\partial H}{\partial T}\right)_pdT \quad \text{ and }\quad  dS=\left( \frac{\partial S}{\partial p}\right)_Tdp+\left( \frac{\partial S}{\partial T}\right)_pdT$$
# 
# Next substituting $dH$ into $dH = TdS + Vdp$ and rearranging gives
# 
# $$\displaystyle dS=\frac{1}{T}\left[\left( \frac{\partial H}{\partial p}\right)_T -V \right]dp+ \frac{1}{T}\left( \frac{\partial H}{\partial T}\right)_pdT$$
# 
# and comparing terms with $dP$ and then $dT$ and using 
# 
# $$\displaystyle dS=\left( \frac{\partial S}{\partial p}\right)_T dp + \left( \frac{\partial S}{\partial T}\right)_pdT$$
# 
# therefore  $\displaystyle \left( \frac{\partial S}{\partial T}\right)_p=\frac{C_p}{T}$. 
# 
# To find the entropy this equation has to be integrated,
# 
# $$\displaystyle \int dS =\int \frac{C_p}{T}dT$$
# 
# which produces $S=S_0+C_p\ln(T)$ and assuming that $C_p$ is independent of temperature over the range of temperatures studied and $S_0$ is an integration constant.
# 
# ### Q104 answer
# Equation(50) states that  $\displaystyle\left( \frac{\partial U}{\partial V}\right)_T=T\left( \frac{\partial p}{\partial T}\right)_V-p$ and as the van der Waals equation is $\displaystyle \left(p+\frac{a}{V^2}\right)(V-b)=RT$, differentiating pressure wrt $T$ gives $\displaystyle T\left( \frac{\partial p}{\partial T}\right)_V= \frac{RT}{V-b}= p+\frac{a}{V^2}$, therefore $\displaystyle\left( \frac{\partial U}{\partial V}\right)_T=\frac{a}{V^2}$.

# ### Q105 answer
# Volume has the general form $V=f(p,V,E)$ and because $E$ is stated to be constant, its derivative is zero, giving 
# 
# $$\displaystyle  dV=\left( \frac{\partial V}{\partial p}\right)_Tdp+\left( \frac{\partial V}{\partial T}\right)_pdT$$
# 
# see equation 42. Dividing this through by $dT$ at constant $E$ produces 
# 
# $$\displaystyle \left( \frac{\partial V}{\partial T}\right)_E=\left( \frac{\partial V}{\partial p}\right)_T\left( \frac{\partial p}{\partial T}\right)_E+\left( \frac{\partial V}{\partial T}\right)_p$$
# 
# ### Q106 answer
# Rewriting to isolate $V$ gives $V=RT/p+B_T$ and differentiating produces 
# 
# $$\displaystyle \left( \frac{\partial V}{\partial p}\right)_T=-\frac{RT}{p^2}$ and $\displaystyle \left( \frac{\partial V}{\partial T}\right)_p=\frac{R}{p}+\frac{dB_T}{dT}$$ 
# 
# The mixed derivatives are the same 
# 
# $$\displaystyle \left( \frac{\partial V}{\partial T}  \left( \frac{\partial V}{\partial p} \right)_T \right)_p = \left( \frac{\partial V}{\partial p} \left( \frac{\partial V}{\partial T}\right)_p\right)_T =-\frac{R}{p^2}$$
# 
# showing that they are perfect differentials.
# 
# ### Q107 answer
# By integrating the Maxwell equation, the entropy change between any two pressures $p_0$ and $p_1$ is
# 
# $$\displaystyle \Delta S_{p_0 \to p_1} =-\int_{p_0}^{P_1} \left(\frac{\partial V}{\partial T} \right)_pdp$$
# 
# (a) Expanding the van der Waals equation of state $(p+a/V^2)(V-b)=RT$ gives $pV+a/V - pb-ab/V^2 = RT$. Differentiating $V$ with respect to $T$ with $p$ constant and rearranging gives $\displaystyle \left(\frac{\partial V}{\partial T} \right)_p\left( p-a/V^2 +2ab/V^3 \right)=R$. The entropy is obtained by substituting for the derivative, forming an integral with a standard logarithmic form; for example, see Integration 2.3.
# 
# $$\displaystyle \Delta S_{0\to p}=-R\int_0^p \frac{1}{p-a/V^2 +2ab/V^3}dp=-R\ln\left(p-\frac{a}{V^2} +\frac{2ab}{V^3}  \right)\bigg | _0^p$$
# 
# The similar calculation for the ideal gas gives 
# 
# $$\displaystyle \Delta S_{0\to 1} =-R\int_0^1 \frac{dp}{p}=-R\ln(p)\bigg | _0^1$$
# 
# and by subtracting this entropy from that of the vdW gas makes the change in entropy from 1 to $p$ bar 
# 
# $$\displaystyle \Delta S_{1\to p}=-R\ln\left(p-\frac{a}{V^2} +\frac{2ab}{V^3}  \right)$$
# 
# This reduces to that of the ideal gas when both $a$ and $b$ are zero.  This result can be written in terms of volume alone by substitution 
# 
# $$\displaystyle \Delta S = -R\ln\left(\frac{RT}{V-b}-\frac{2ab}{V^3}  \right)$$
# 
# The units inside the log have dimensions of pressure. These must be made dimensionless by dividing by 1 bar, although this is almost never shown in a formula.
# 
# (b) The Bertholet equation can be simplified  
# 
# $$\displaystyle V= \frac{RT}{p}+c_0\left(1-\frac{c_1}{T^2}\right)$$
# 
# where $c_0, c_1$ are constants as may be seen by comparison with the equation in the question. The derivative is 
# 
# $$\displaystyle \left( \frac{\partial V}{\partial T} \right)_p= \frac{R}{p}-\frac{2c_0c_1}{T^3}$$
# 
# The change in entropy is, following the method in (a) 
# 
# $$\displaystyle \Delta S_{0\to p}=-\int_0^p \frac{R}{T}+\frac{2c_0c_1}{T^3}dp=-r\ln(p)\bigg |_0^p-\frac{2c_0c_1R}{T^3}p$$
# 
# making 
# 
# $$\displaystyle \Delta S_{1\to p}=-R\ln(p) -\frac{27RT_c^2}{32p_c}\frac{p}{T^3}$$
# 
# as the difference in entropy from 1 to $p$ bar. The equation is dimensionally correct; in the second term the temperature units cancel as do the pressure, leaving only $R$. The pressure should be represented as $p$/bar.
# 
# ### Q108 answer
# (a) Differentiating $dH=TdS+Vdp$  with $p$ at constant $T$ changes the differentials into partial ones and gives 
# 
# $$\displaystyle \left( \frac{\partial H }{\partial p} \right)_T =T \left( \frac{\partial S}{\partial p} \right)_T+V$$
# 
# Substituting for the entropy term produces the required equation $\displaystyle \left( \frac{\partial H }{\partial p} \right)_T =-T \left( \frac{\partial V}{\partial T} \right)_p+V$.
# 
# (b) Differentiating each term with temperature at constant pressure gives 
# 
# $$\displaystyle \frac{\partial}{\partial T}\left( \left( \frac{\partial H }{\partial p} \right)_T \right)_p =-T\left( \frac{\partial^2 V }{\partial T^2} \right)_p-\left( \frac{\partial V }{\partial T} \right)_p+\left( \frac{\partial V }{\partial T} \right)_p$$
# 
# Using the definition $C_p = dH/dT \equiv (\partial H/\partial T)_p$ the left-hand side is simplified and also cancelling terms on the right gives
# 
# $$\left( \frac{\partial C_p }{\partial p} \right)_T = -T \left( \frac{\partial^2 V }{\partial T^2} \right)_p$$
# 
# Integrating this equation produces 
# 
# $$\displaystyle (C_{p_1} - C_{p_0})_T=-T\int_0^{p_1}\left(\frac{\partial^2 V}{\partial T^2} \right)_pdp$$
# 
# which means that the heat capacity relative to a standard state at $p$ = 0 and at temperature $T$ is found by integrating the second derivative of the change in volume of the gas with temperature over a range of pressures from 0 to $p_1$. This change is zero for an ideal or van der Waals gas but not for the Berthelot gas where from the previous question 
# 
# $$\displaystyle \left(\frac{\partial V}{\partial T} \right)_p=\frac{R}{p}+\frac{27RT_c^2}{32p_c}\frac{1}{T^3}$$
# 
# and 
# 
# $$\displaystyle \left(\frac{\partial^2 V}{\partial T^2} \right)_p=-\frac{81RT_c^2}{32p_c}\frac{1}{T^4}$$
# 
# making 
# 
# $$\displaystyle (C_{p_1} - C_{p_0})_T=\frac{81RT_c^2}{32p_c}\frac{p_1}{T^3}$$
# 
# Note that the right-hand side of this equation has dimensions of energy/temperature, as does the heat capacity.
# 
# ### Q109 answer
# (a) by definition, $\displaystyle C_V=\left( \frac{\partial U}{\partial T} \right)_V$  and also as $H=U+pV$, and by definition 
# 
# $$\displaystyle C_p=\left( \frac{\partial H}{\partial T} \right)_p=\left( \frac{\partial U}{\partial T} \right)_p+ \left( \frac{\partial pV}{\partial T} \right)_p$$
# 
# it follows that 
# 
# $$\displaystyle C_p=\left( \frac{\partial U}{\partial T} \right)_p + p\left( \frac{\partial V}{\partial T} \right)_p$$
# 
# at constant pressure.
# 
# (b) Comparing the equations for $C_p$ in parts (a) and (b) it seems that the derivative $\partial V/\partial T$ at constant pressure will be needed, so we start with this. The internal energy is a function $U = f(p, V, T)$, and therefore can produce the differential 
# 
# $$\displaystyle dU=\left( \frac{\partial U}{\partial T} \right)_VdT+\left( \frac{\partial U}{\partial V} \right)_TdV$$
# 
# which by dividing by $dT$ at constant pressure gives
# 
# $$ \left( \frac{\partial U}{\partial T} \right)_p= \left( \frac{\partial U}{\partial T} \right)_V+\left( \frac{\partial U}{\partial V} \right)_T\left( \frac{\partial V}{\partial T} \right)_p$$
# 
# The first term on the right is $C_V$ which leaves $(\partial U/\partial T)_p$ to be found. As $U=H-pV$ then 
# 
# $$\displaystyle dU=\left( \frac{\partial H}{\partial T} \right)_pdT-p\left( \frac{\partial V}{\partial T} \right)_pdT$$
# 
# and by dividing  by $dT$ at constant $p$ produces, 
# 
# $$\displaystyle \left(\frac{\partial U}{\partial T}\right)_p=\left( \frac{\partial H}{\partial T} \right)_p-p\left( \frac{\partial V}{\partial T} \right)_p$$
# 
# Substituting for $(\partial U/\partial T)_P$ gives the final expression
# 
# $$C_p=C_V+\left[p+\left( \frac{\partial U}{\partial V} \right)_T  \right]\left( \frac{\partial V}{\partial T} \right)_p$$
# 
# which reduces to $C_p=C_V+R$ for an ideal gas.

# ### Q110 answer
# Differentiating the entropy and multiplying by $T$ produces $\displaystyle TdS=T\left( \frac{\partial S }{\partial T} \right)_VdT+T\left( \frac{\partial S}{\partial V} \right)_TdV$. 
# 
# By substituting Maxwell's equation 
# 
# $$\displaystyle  TdS=T\left( \frac{\partial S }{\partial T} \right)_VdT+T\left( \frac{\partial p}{\partial V} \right)_VdV$$
# 
# The first term has now to be examined. The heat capacity $C_V$ is the rate of change of internal energy with temperature $C_V = (\partial U/\partial T)_V$ and entropy $S = U/T$ giving $C_V = T(\partial S/\partial T)_V$ which by substitution gives the final result, 
# 
# $$\displaystyle TdS=C_VdT+T\left( \frac{\partial p}{\partial V} \right)_VdV $$
# 
# ### Q 111 answer
# (a) Differentiating the Helmholtz energy gives $\displaystyle \left( \frac{\partial A}{\partial V} \right)_T=\left( \frac{\partial U}{\partial V} \right)_T-T\left( \frac{\partial S}{\partial V} \right)_T$.
# 
# Next the derivatives need to be changed into terms that can be measured, which means those involving $T$ and $p$. With this in mind using the differential form $dA = \cdots$ The first derivative is 
# 
# $$\displaystyle \left( \frac{\partial A}{\partial V} \right)_T=-p$$ 
# 
# and because $dA$ is a perfect differential, the derivative in $S$ becomes 
# 
# $$\displaystyle \left( \frac{\partial S}{\partial V} \right)_T=\left( \frac{\partial p}{\partial T} \right)_V$$
# 
# and the rate of change of internal energy 
# 
# $$\displaystyle \left( \frac{\partial U}{\partial V} \right)_T=T\left( \frac{\partial p}{\partial T} \right)_V-p$$
# 
# The van der Waals equation can now be used directly giving 
# 
# $$\displaystyle \left( \frac{\partial p}{\partial T} \right)_V=\frac{nR}{V-b}$$ 
# 
# and the final result is $\displaystyle \left( \frac{\partial U}{\partial V} \right)_T=a\frac{n^2}{V^2}$.
# 
# (b) The constant $a$ for CO$_2$ is $3.66\,\mathrm{ bar\, dm^6\,mol^{-2}}$. Integrating the last result gives 
# 
# $$\displaystyle \Delta U=\int_{V_1}^{V_2} \frac{an^2}{V^2}dV=-n^2a\left( \frac{1}{V_2}-\frac{1}{V_1} \right)$$
# 
# which has units $\mathrm{bar dm^3\,mol^{-1}}$. As pressure (bar) is force/area therefore the units are $\displaystyle \mathrm{\frac{kg\cdot m\cdot s^{-2}}{m^2}m^3=\frac{kg\cdot m^2}{s^2}=J}$.  The change in internal energy of $1$ mole of CO$_2$ expanding from $10 \to 20\,\mathrm{ dm^3}$ at constant temperature is therefore $0.183$ J.
# 
# ### Q112 answer
# (a) The integral is $\displaystyle \int\left(\frac{\partial U}{\partial V} \right)_TdV=\int\left[ T\left( \frac{\partial S}{\partial V} \right)_T-p \right]dV$ 
# 
# and this can be put in a more useful form with the Maxwell equation for $S$ and $V$ given in Q110 and then 
# 
# $$\displaystyle \int\left[ T\left( \frac{\partial p}{\partial T} \right)_V-p \right]dV$$ 
# 
# Differentiating the (van der Waals) pressure wrt $T$, simplifying the result and integrating gives 
# 
# $$\displaystyle \int\left(\frac{\partial U}{\partial V} \right)_TdV = an^2\int\frac{dV}{V^2}=-\frac{an^2}{V}$$
# 
# (b) The total energy is 
# 
# $$\displaystyle U_{total}=\frac{3}{2}nk_BT-\frac{an^2}{V}$$
# 
# where the first term is the kinetic energy obtained by using the eqi-partition  theorem and is in effect the internal energy if chlorine is assumed to be an ideal gas.
# 
# ### Q113 answer
# Rewriting gives $\displaystyle \ln(k)=\ln(A)-\frac{\Delta H}{RT}+\frac{\Delta S}{R} $ and differentiating at constant pressure 
# 
# $$\displaystyle \frac{1}{k}\left(\frac{\partial k }{\partial T}  \right)_p=\frac{1}{A}\left(\frac{\partial A }{\partial T}  \right)_p+\frac{\Delta H}{RT^2}-\frac{1}{RT}\left(\frac{\partial \Delta H}{\partial T }  \right)_p+\frac{1}{R}\left(\frac{\partial \Delta S}{\partial T}  \right)_p$$
# 
# Substituting the identity in the question and rearranging produces $\displaystyle \left(\frac{\partial k }{\partial T}  \right)_p=k\left[ \frac{1}{A}\left(\frac{\partial A }{\partial T}  \right)_p+\frac{\Delta H}{RT^2} \right]$.

# ### Q114 answer
# (a) The partial derivative can be expressed directly as 
# 
# $$\displaystyle \frac{\partial z}{\partial x}=\frac{\partial z}{\partial r}\frac{\partial r}{\partial x}+\frac{\partial z}{\partial \theta }\frac{\partial \theta}{\partial x}$$
# 
# The derivatives in $x$ are calculated using a right-angled triangle to obtain angles (see Q12 & fig 38),
# 
# $$\displaystyle \frac{\partial r}{\partial x}=\frac{x}{\sqrt{x^2+y^2}}=\cos(\theta)$$
# 
# and as $\theta =\tan^{-1}(y/x)$
# 
# $$\displaystyle \frac{\partial \theta}{\partial x}=-\frac{y}{x^2+y^2}=-\frac{\sin(\theta)}{r}$$
# 
# and these can be substituted into the result if required.
# 
# (b) The second derivative is harder to evaluate. Start by differentiating the last result as if the whole of it were $z$, i.e.
# 
# $$\displaystyle  \frac{\partial^2 z}{\partial x^2}=\frac{\partial}{\partial r}\left[\frac{\partial z}{\partial r}\frac{\partial r}{\partial x}+\frac{\partial z}{\partial \theta }\frac{\partial \theta}{\partial x} \right]\frac{\partial r}{\partial x}+\frac{\partial}{\partial \theta}\left[\frac{\partial z}{\partial r}\frac{\partial r}{\partial x}+\frac{\partial z}{\partial \theta }\frac{\partial \theta}{\partial x} \right]\frac{\partial \theta}{\partial x}$$
# 
# Using the product rule and operating to the right using $\partial \partial r$ the first term becomes 
# 
# $$\displaystyle \begin{align}
# \frac{\partial}{\partial r}\left[\frac{\partial z}{\partial r}\frac{\partial r}{\partial x}+\frac{\partial z}{\partial \theta }\frac{\partial \theta}{\partial x} \right]\frac{\partial r}{\partial x}
# &= \frac{\partial^2 z }{\partial r^2} \left( \frac{\partial r}{\partial x} \right)^2
#   +\frac{\partial z}{\partial r} \left( \frac{\partial } {\partial x}  \frac{\partial r}{\partial x} \right)    \frac{\partial r}{\partial x} +
#   \left(\frac{\partial }{\partial r}\frac{\partial z}{\partial \theta}  \right)\frac{\partial \theta}{\partial x}\frac{\partial r}{\partial x} +\frac{\partial z}{\partial \theta}\left(\frac{\partial }{\partial r}\frac{\partial \theta}{\partial x}  \right)\frac{\partial r}{\partial x}\\
#  &= \frac{\partial^2 z }{\partial r^2} \left( \frac{\partial r}{\partial x} \right)^2 +
#   \frac{\partial z }{\partial x} \frac{\partial^2 r }{\partial x^2}+
#   \frac{\partial^2 z }{\partial r\partial \theta}\frac{\partial \theta }{\partial x}\frac{\partial r }{\partial x}+
#   \frac{\partial z }{\partial \theta}  \frac{\partial^2 \theta }{\partial r\partial x} \frac{\partial r }{\partial x}
# \end{align}$$
# 
# and we used 
# 
# $$\displaystyle \frac{\partial}{\partial r}\frac{\partial r}{\partial x}\equiv \frac{\partial }{\partial x}$$
# 
# in the second term. The second term is
# 
# $$\displaystyle \begin{align}
# \frac{\partial}{\partial \theta}\left[\frac{\partial z}{\partial r}\frac{\partial r}{\partial x}+\frac{\partial z}{\partial \theta }\frac{\partial \theta}{\partial x} \right]\frac{\partial \theta}{\partial x}& =
# \frac{\partial^2 z }{\partial \theta \partial r }  \frac{\partial r}{\partial x} \frac{\partial \theta }{\partial x} +
# \frac{\partial z}{\partial r}\left(  \frac{\partial }{\partial \theta}
# \frac{\partial r }{\partial x} \right)
# \frac{\partial \theta}{\partial x}  +
# \frac{\partial^2 z }{\partial \theta^2}\frac{\partial \theta}{\partial x}\frac{\partial \theta }{\partial x}+
# \frac{\partial z}{\partial \theta}\left( \frac{\partial }{\partial \theta}\frac{\partial \theta}{\partial x} \right)\frac{\partial \theta}{\partial x}\\ &=
# \frac{\partial^2 z }{\partial \theta \partial r }  \frac{\partial r}{\partial x} \frac{\partial \theta }{\partial x}+
# \frac{\partial z}{\partial r}  \frac{\partial^2 r }{\partial \theta \partial x} 
# \frac{\partial \theta}{\partial x}+\frac{\partial^2 z }{\partial \theta^2}\left( \frac{\partial \theta}{\partial x}\right)^2+\frac{\partial z}{\partial \theta}\frac{\partial^2 \theta}{\partial x^2}
# \end{align}$$
# 
# Finally, notice that all the terms are derivatives between the two coordinates except two of them, which have $\partial ^2\theta/\partial r\partial x$ and $\partial^2r/\partial \theta \partial x$ and these are zero because $r$ and $\theta$ are only functions of $x$ and $y$ and a derivative with respect to anything else would be zero. The result is
# 
# $$\frac{\partial^2 z }{\partial x^2}=\frac{\partial^2 z }{\partial r^2}\left(\frac{\partial r}{\partial x}\right)^2 +\frac{\partial^2 z }{\partial \theta^2}\left( \frac{\partial \theta}{\partial x} \right)^2+\frac{\partial z}{\partial x}\frac{\partial^2 r }{\partial x^2 }+\frac{\partial z }{\partial \theta}\frac{\partial^2 \theta}{\partial x^2}+2\frac{\partial^2 z }{\partial r\partial \theta }\frac{\partial \theta}{\partial x}\frac{\partial r}{\partial x }$$
# 
# The second derivatives in $r, x \theta$ are
# 
# $$\displaystyle  \frac{\partial^2 r}{\partial x^2}=\frac{y^2}{(x^2+y^2)^{3/2}}=\frac{\sin^2(\theta)}{r}\\
#    \frac{\partial^2 r}{\partial \theta^2}=\frac{2xy}{(x^2+y^2)^2}=\frac{2\cos(\theta)\sin(\theta)}{r^2}$$
#    
# substituting and rearranging gives the result
# 
# $$\displaystyle  \frac{\partial ^2 z}{\partial x^2}+\frac{\partial^2 z }{\partial y^2}=\frac{\partial^2 z }{\partial r^2 }+\frac{1}{r }\frac{\partial z}{\partial r}+\frac{1}{r^2 }\frac{\partial^2 z}{\partial \theta^2}$$

# In[ ]:




