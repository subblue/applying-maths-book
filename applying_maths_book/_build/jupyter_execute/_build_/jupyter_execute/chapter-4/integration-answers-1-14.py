#!/usr/bin/env python
# coding: utf-8

# ## Solutions Q1 - 14

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
init_printing()                      # allows printing of SymPy results in typeset maths format
plt.rcParams.update({'font.size': 16})  # set font size for plots


# ### Q1 answer
# The kinetic energy is zero and the total energy $E$ is the same as the potential energy at the turning point, figure 38.
# 
# (a) By definition the energy is $E=\int k(r-r_e)dr$ and because $k$ and $r_e$ are constants, these can be taken outside the integral
# 
# $$\displaystyle E=k\int rdr -kr_e\int dr=\frac{kr^2}{2}-kr_er+c$$
# 
# where the integral $\int dr = r$. Note that only one constant $c$ is needed even though there are two integrals.
# 
# To solve the specific problem, the definite integral with limits $1.146r_e$ and $r_e$ (in nm) is
# 
# $$\displaystyle E=k\int_{r_e}^{1.146r_e}(r-r_e)dr=kr_e\left[ \frac{r^2}{2}-r\right]_{r_e}^{1.146r_e} = 0.0107kr_e^2$$
# 
# ![Drawing](integration-fig38.png)
# 
# Figure 38. Potential energy (blue) and kinetic energy (red) of a diatomic molecule as a vibrating harmonic oscillator. The turning points,where the kinetic energy is zero are marked with black dots.n The total energy is $E$.
# ____
# 
# (b) Using values for $k$ and $r_e$ this energy is $0.89 \cdot 10^{-19}$ J / molecule or $53.6\,\mathrm{ kJ \, mole^{-1}}$, which is $\approx 12$% of the dissociation energy - a surprisingly large amount.
# 
# (c) The frequency in $\mathrm{s^{-1}}$ is calculated using $\displaystyle \nu = \frac{1}{2\pi}\sqrt{ \frac{k}{\mu} } =0.904\cdot 10^{14}\,\mathrm{s^{-1}}$ where $\mu$ is the reduced mass $35/36 \times 1.667 \cdot 10^{-27}$ kg and as $n = 1$, the total energy of this level is $0.89 \cdot 10^{-19}$ J which is the same as calculated above.
# 
# ### Q2 answer
# (a) Taking logs of both sides of the Arrhenius equation produces $\displaystyle \ln(k) = \ln(k_0) - \frac{E_a}{RT}$. 
# 
# Differentiating with respect to $T$ gives $\displaystyle \frac{d\ln(k)}{dT}=\frac{E_a}{RT^2}$ as in the question.
# 
# (b) To integrate this differential equation, separate the equation into parts in $k$ and in $T$ as 
# 
# $$\displaystyle \int_{k_1}^{k_2}d\ln(k) = \int_{T_1}^{T_2} \frac{E_a}{RT^2}dT$$
# 
# and integrate both sides separately. Because the term $d\ln(k)$ integrates to $\ln(k)$ the result is 
# 
# $$\displaystyle \ln(k_2)-\ln(k_1)=-\frac{E_a}{RT}\bigg|_{T_1}^{T_2}$$
# 
# This can be simplified to $\displaystyle \ln\left( \frac{k_2}{k_1} \right) = \frac{E_a}{R}\left(\frac{1}{T_1}-\frac{1}{T_2} \right)$ which can be further rearranged into the equation in the question.
# 
# ### Q3 answer
# Rearranging to separate variables produces $\displaystyle \int\frac{dc}{c}=-k\int dt$. 
# 
# Integrating both  sides separately gives $\ln(c)=-kt+q$ where $q$ is a constant. 
# 
# The constant must be included and is undefined until the problem is specified exactly. Because the concentration is $c_0$ at $t = 0$ the calculation can be continued in two ways. 
# 
# First taking the result $\ln(c) = -kt + q$, when $t = 0,\, c = c_0$ then $q = \ln(c_0)$ and therefore $\ln(c) = -kt + \ln(c_0)$ or 
# 
# $$\displaystyle\ln\left(\frac{c}{c_0}\right)=-kt$$
# 
# and secondly by adding the limits in the integration initially as 
# 
# $$\displaystyle  \int_{c_0}^c\frac{dc}{c}=-k\int_0^t dt, \qquad  \to \qquad\ln(c)\bigg|_{c_0}^c=-kt\bigg|_0^t$$
# 
# which produces the same result.
# 
# ### Q4 answer
# (a) Rearranging the equation and adding limits gives $\displaystyle \int_{v_0}^v\frac{dv}{v}=-\frac{3\pi\delta\eta}{m}\int_0^t dt$ 
# 
# and integrating gives 
# 
# $$\displaystyle \ln(v)\bigg|_{v_0}^v=-\frac{3\pi\delta\eta}{m}t\bigg|_0^t$$
# 
# which can be rearranged to give $v=v_0\exp\left(-\frac{3\pi\delta\eta }{m}t\right)$.
# 
# The exponential must be dimensionless therefore the ratio, $\displaystyle \tau = \frac{m}{3\pi\delta\eta}$ represents a time. To prove that this is a time, add dimensions to the constants. Viscosity is usually measured in centipoise which is not an SI unit. The SI units of viscosity are $\mathrm{ kg\, m^{-1} \,s^{-1}}$ and $1 cP = 0.001\,\mathrm{ kg\, m^{-1}\, s^{-1} }$. The dimensions are then 
# 
# $$\displaystyle \frac{m(\mathrm{kg}) }{\delta(\mathrm{m}) \eta(\mathrm{ kg\, m^{-1}\, s^{-1} })  }\equiv s$$
# 
# (b) Estimating the time means knowing the mass and size of the particles. Benzene and similar molecules may be estimated to be $\approx 0.3$ nm in diameter, and of mass $\approx 100$ amu where $1\, \mathrm{amu} = 1.66 \cdot 10^{-27}$ kg. The viscosity of water is 1 cP. The relaxation time for benzene is therefore approximately
# 
# $$\displaystyle \frac{100\times 1.66\cdot 10^{-27}}{3\pi \times 0.3\cdot 10^{-9} \times 0.001}\approx 6\cdot 10^{-14} \text{s}$$
# 
# or 60 fs, and this must be about the time between intermolecular collisions.
# 
# The calculation for the protein is just as straightforward, except that the mass of the protein is not known. One way round this is to use a typical density $\rho$ and $100 \,\mathrm{kg\, m^{-3}}$ is typical, the mass is $
# \rho\delta^3$. The time is therefore 
# 
# $$\displaystyle \tau =\frac{\rho\delta^3  }{3\pi\eta}\approx 10^{-12}\text{s}$$
# 
# which means that the direction of the initial movement is lost within about a picosecond, which is a very short time considering the relatively large size of a protein.
# 
# ### Q5 answer
# (a) In this case,the answer can be guessed to be something like $\ln(3x - 2)$ because of the reciprocal in $x$ in the function. Differentiating this guess gives $3/(3x-2) $ and therefore $\displaystyle \int\frac{dx}{3x-2}=\frac{1}{3}\ln(3x-2)$.
# 
# (b) Converting to exponentials gives 
# 
# $$\displaystyle I=\int\cosh^2(x)dx=\frac{1}{4}\int\left(e^{2x}+2+e^{-2x}\right)dx =\frac{1}{8}\left(e^{2x}+4x-e^{-2x} \right)+c$$
# 
# With some manipulation this can be returned to a trig form which is $x/2 + \sinh(2x)/4$.
# 
# (c) This can be converted to exponentials first giving $I=\cosh^2(x)/2$
# 
# (d) Expanding the integral gives $\displaystyle I=\int 2\ln(2x)-\ln(1-x)dx$ which has a standard form, see 2.13.
# 
# (e) The function in the integral is odd so the integral is zero because the limits are symmetrical about zero.
# 
# (f) The integral becomes a standard one the type $x^{n+1}/(n+1)$ by letting $u-x-3$. Thus this is an example of using substitution to simplify and then solve the integral. 
# 
# $$\displaystyle I=\int u^{1/2}du =\frac{2}{3}u^{3/2} \to \frac{2}{3}(x-3)^{3/2}$$
# 
# Adding limits produces 
# 
# $$\displaystyle I=\frac{2}{3}(x-3)^{3/2}\bigg|_{-1}^1=\frac{2}{3}\left((-2)^{3/2}-(-4)^{3/2}\right)=\frac{4i}{3}\left(\sqrt{2}-4\right)$$
# 
# ### Q6 answer
# Converting to the exponential form gives  
# 
# $$\displaystyle \int_0^L \frac{1}{4}\left( 2-e^{2iLx}-e^{-2iLx} \right)dx = \frac{L}{2} -\frac{\sin(2L^2)}{4L} $$
# 
# The sine cannot be greater than $\pm 1$ so that when $L$ is large, for example $\gt 100$, this term becomes small because it is divided by $L$ and the limit approaches $L/2$.

# ### Q7 answer
# Integrating to obtain the velocity gives 
# 
# $$\displaystyle \begin{align} \boldsymbol v &=\int_0^t (2\sin(\omega_0 t)\boldsymbol{i}+\cos(\omega_0 t)\boldsymbol{j}+t\boldsymbol{k})\; dt
# = \left( -2\frac{\cos(\omega_0 t)}{\omega_0}\boldsymbol{i}+\frac{\sin(\omega_0 t)}{\omega_0}\boldsymbol{j} +\frac{t^2}{2}  \boldsymbol{k}\right)\Bigg|_0^t \\
# &= \frac{2-2\cos(\omega_0 t)}{\omega_0}\boldsymbol{i}+\frac{\sin(\omega_0 t)}{\omega_0}\boldsymbol{j}+\frac{t^2}{2}\boldsymbol{k} \end{align}$$
# 
# As a check at t = 0, the particle is stationary and the vector must be zero which it is; $\boldsymbol v = 0\boldsymbol i + 0\boldsymbol j + 0\boldsymbol k$. The displacement vector $\boldsymbol r$ is obtained by a further integration since $\boldsymbol v = d\boldsymbol r/dt$ or $\boldsymbol r=\int\boldsymbol v dt$. Integrating the last  results gives 
# 
# $$\displaystyle \boldsymbol{r} = \frac{2}{\omega_0^2}[\omega_0 t-\sin(\omega_0 t)]\boldsymbol{i}-\frac{1}{\omega_0^2}[\cos(\omega_0 t)-1]\boldsymbol{j}+\frac{t^3}{6}\boldsymbol{k}$$
# 
# and at $t=0$, $\boldsymbol r = 0\boldsymbol i + 0\boldsymbol j + 0\boldsymbol k$.
# 
# ### Q8 answer
# Because of the term in $1/V$ this is a logarithmic integration; 
# 
# $$\displaystyle \int_{V_1}^{kV_1} pdV=\int_{V_1}^{kV_1}\frac{nRT}{V}dV=nRT\ln(V)\bigg|_{V_1}^{kV_1}=nRT\ln(k)$$
# 
# and in the last step $\ln(kV_1)-\ln(V_1)=\ln(kV_1/V_1)$ was used.
# 
# ### Q9 answer
# (a) If $V$ is the initial volume which is reduced to 10%, then the work done is calculated using equation 6 ($\displaystyle \int x^n dx$) with power $n=-\gamma$.
# 
# $$\displaystyle w=\int_V^{V/10}pdV=-\int_V^{V/10}\frac{k}{V^\gamma}dV=-\frac{kV^{1-\gamma}}{1-\gamma}\bigg|_V^{V/10}=\frac{kV^{1-\gamma}}{1-\gamma}(1-10^{\gamma-1})$$
# 
# Substituting values in $pV^\gamma = k$ makes $k = 101325 \times 5.0^{1.404}
# = 9.707 \cdot 10^5$ J and therefore $w = +1.96 \cdot 10^6$ J.
# 
# (b) The first law states that the work done on the gas plus the heat transferred to the gas must be equal to the change in internal energy of the gas, $dU$ where $dU = \bar dq + \bar dw$. As no heat enters or leaves, the change in heat $\bar dq = 0$, and the work done is the same as the change in internal energy. This depends only on the temperature and at constant volume $dU = C_V dT$ therefore, for $n$ moles of gas, the work done is
# 
# $$w=n\int_{T_1}^{T_2}C_V dT =\frac{5nR}{2}\int_{T_1}^{T_2}dT=\frac{5nR}{2}(T_2-300)$$
# 
# Because the volume and pressure are known, $n = 203.12$ moles and solving for $T_2$ produces $T_2 = 756$ K which is the temperature of the gas immediately after compression.
# 
# The bar notation $\bar dq$ and $\bar dw$ means that in the language of thermodynamics when integrated, they form _path integrals_, i.e. normal integrals to you or me, and their value depends on the way the heat changes or the work is done. Some authors instead use $\delta q$ and $\delta w$ to indicate a path integral. The internal energy $U$ is a state function and when integrated, its value depends only on the starting and ending values not on how the internal energy was obtained by the molecules for example $\displaystyle \int_{U_1}^{U_2} dU=U_2-U_1$
# 
# ### Q10 answer
# (a) If $V_0$ is the initial, $V_1$ the final volume then the work done is 
# 
# $$\displaystyle w=-\int_{V_0}^{V_1}pdV\quad\text{ where }\quad \displaystyle p=\frac{nRT}{V-nb}-\frac{an^2}{V^2}$$ 
# 
# Integrating gives
# 
# $$\displaystyle w=-\int_{V_0}^{V_1} \frac{nRT}{V-nb}-\frac{an^2}{V} dV=-nRT\ln\left(\frac{V_1-nb}{V_0-nb}  \right) +an^2\left( \frac{1}{V_0}-\frac{1}{V_1} \right)$$
# 
# (b) The units of the second term need some care; they are 
# 
# $$\displaystyle a \,(\mathrm{ bar \,dm^6 \,mol^{-2})\,n\,(\,mol^2)/V(\,dm^3)  = bar\, dm^3} = 100 \text{ J}$$
# 
# Since pressure is force/area and $1 \mathrm{ bar = 10^5}$ Pa, multiplying by $1 \times \mathrm{ m^3}$ would convert this into $10^5$ joules; (energy = force $\times$ distance) but the volume is in dm$^3$ or $10^{-3} \mathrm{m^3}$ so the conversion is 100. Substituting the constants gives the work as 18.3 kJ. The ideal gas by comparison needs more work at 22.4 kJ to compress it.
# 
# (c) If the gas were O$_2$ then the van der Waals equation suggests that $21.9$ kJ are needed and if H$_2$ then $23.6$ kJ.
# 
# The energy need to compress the Cl$_2$ is less than that for ideal gas, that for O$_2$ about the same, and H$_2$ slightly greater. The difference is due to the interplay of the attractive potential term $a$ and the repulsive one $b$. If the attractive forces dominate as the gas is compressed the molecules will remain closer to one another for longer than if they were hard spheres. This reduces the effective pressure and so less work is needed. This is the case for the polarizable chlorine molecules. If repulsion dominates,then the molecules avoid one another effectively increasing the pressure and this is the case for H$_2$, which is weakly polarizable, compared to Cl$_2$. The energy to compress is larger. Oxygen molecules seem to have a balance of repulsion and attraction that makes them appear to behave as it they were ideal, but this is an accidental cancelling of two effects.
# 
# ### Q11 answer
# (a) The enthalpy change is 
# 
# $$\displaystyle H_T^\mathrm{o}-H_{298}^\mathrm{o}=\int_{298}^T a+bT+\frac{c}{T^2}dT=a(T-298)+\frac{b}{2}\left(T^2-298^2\right)-c\left(\frac{1}{T}-\frac{1}{298}  \right)$$
# 
# and entropy change 
# 
# $$S_T^\mathrm{o}-S_{298}^\mathrm{o}=\int_{298}^T \frac{a}{T}+b+\frac{c}{T^3}dT=a\ln\left(\frac{T}{298}\right)+b(T-298)-\frac{c}{2}\left(\frac{1}{T^2}-\frac{1}{298^2}  \right)$$
# 
# (b) Using the values for the constants produce $H_T^\mathrm{o} - H_{298}^\mathrm{o} = 2.95$ kJ/mol and $S_T^\mathrm{o} - S_{298}^\mathrm{o} = 9.14$ J/mol/K.
# 
# ### Q12 answer
# (a) The Clapeyron equation for a change of phase such as melting or evaporation is 
# 
# $$\displaystyle \frac{dp}{dT}=\frac{\Delta S}{\Delta V}$$ 
# 
# Because $ \Delta S=\Delta H/T$ for the phase change, $\displaystyle \frac{dp}{dT}=\frac{\Delta H}{T\Delta V}$. 
# 
# For a liquid to vapour transition this is written as $\displaystyle \frac{dp}{dT}=\frac{\Delta_{vap} H}{T\Delta_{vap} V}$. Separating terms in pressure and temperature produces 
# 
# $$\displaystyle \int_{p_1}^{p_2}dp= \frac{\Delta H}{\Delta V}\int_{T_1}^{T_2} \frac{dT}{T}$$
# 
# and therefore 
# 
# $$\displaystyle p_2-p_1=\frac{\Delta H}{\Delta V}\ln\left( \frac{T_2}{T_1} \right)$$
# 
# provided $\Delta V$ remains constant with a change in pressure and temperature. This is the Clapeyron equation.
# 
# (b) The Clausius-Clapeyron equation is obtained by allowing the change in volume (per mole) on forming the vapour, to be far larger than that of the original liquid making $\Delta_{vap}V \approx V_{vap}$ and therefore 
# 
# $$\Delta_{vap}V=RT/p\quad\text{ and }\quad\displaystyle \frac{dp}{dT}=p\frac{\Delta_{vap}H}{RT^2}$$
# 
# This is the Clausius-Clapyron equation equation that is usually written as
# 
# $$\frac{d\ln(p)}{dT}=\frac{\Delta_{vap}H}{RT^2}$$
# 
# Integrating 
# 
# $$\displaystyle \int_{p_1}^{p_2}d\ln(p)= \frac{\Delta_{vap} H}{R}\int_{T_1}^{T_2} \frac{dT}{T^2}$$
# 
# produces  $\displaystyle \ln\left(\frac{p_2}{p_1} \right) = -\frac{\Delta_{vap}H}{R}\left( \frac{1}{T_2}-\frac{1}{T_1} \right) $
# 
# ### Q13 answer
# The Clapeyron equation 
# 
# $$\displaystyle p_2-p_1=\frac{\Delta H}{\Delta V}\ln\left( \frac{T_2}{T_1} \right)$$
# 
# is used for a solid-liquid transition. The changes in enthalpy and volume relate therefore to changes occurring in fusion.
# 
# The Clausius-Clapeyron equation describes solid - vapour and liquid - vapour changes because the final volume is far greater than the initial one, and is 
# 
# $$\displaystyle \frac{dp}{dT}=p\frac{\Delta H}{RT^2}$$
# 
# where $\Delta H$ the enthalpy change at the liquid - vapour or sublimation transition. Integrating this last equation from pressure p1 to p2 and temperature $T_1 \to T_2$ gives 
# 
# $$\displaystyle \ln\left(\frac{p_2}{p_1} \right) = -\frac{\Delta_{vap}H}{R}\left( \frac{1}{T_2}-\frac{1}{T_1} \right) $$
# 
# as discovered in the previous question. The change in the volume during fusion is 
# 
# $$\displaystyle \Delta_{fus}V = m\left(\frac{1}{d_l}-\frac{1}{d_s} \right)$$
# 
# where $m$ is the molar mass and $d_l$ and $d_s$  the densities of the liquid and solid. The pressure variation for the solid to liquid (melting or fusion) change is
# 
# $$\displaystyle p_2=p_1+\frac{\Delta_{fus}H}{\Delta_{fus}V}\ln\left(\frac{T_2}{T_1} \right)$$
# 
# and for evaporation and sublimation
# 
# $$\displaystyle p_2=p_1\exp\left( -\frac{\Delta_{vap}H}{R}\left(\frac{1}{T_2}-\frac{1}{T_1} \right) \right)$$
# 
# with the appropriate $\Delta H$. This is $\Delta_{vap}H$ for evaporation and $\Delta_{fus}H + \Delta_{vap}H$ for sublimation. Sublimation is treated as two steps merged into one; melting and instantaneous evaporation.  One way of calculating the phase diagram is shown below.

# In[2]:


# Algorithm 1. Solid-liquid-gas Phase Diagram. Data for benzene

R = 8.314              # J/mol/K
dens_sol = 981.0       # kg/m^3
dens_liq = 879.0       # kg/m^3
mol_mass = 78.0/1000.0 # kg/mol
DH_vap   = 30.8*1000   # J/mol
DH_fus   = 10.6*1000   # J/mol

p3 = 36.0/760*101325 # triple point pressure Pa
T3 = 5.5 + 273.16    # triple point temperture K

DV_fus = mol_mass*(1/dens_liq-1/dens_sol)                          # delta volume fusion

p_liq_vap= lambda T: p3*np.exp( (DH_vap/R)*(1/T3-1/T))             # pressure

p_sol_vap= lambda T: p3*np.exp( ((DH_fus+DH_vap)/R )*(1/T3-1/T) )

p_sol_liq= lambda T: p3 + DH_fus/DV_fus*(np.log(T)-np.log(T3))

# plot each function vs temperature. Use limits to restrict range and produce the figure.


# ![Drawing](integration-fig39.png)
# 
# Figure 39. Calculated phase diagram for benzene using the Clapeyron and Clausius-Clapeyron equations. Pressure is in pascal, temperature in kelvin. 
# ____
# 
# Notice the form of the phase diagram. At low temperatures $\approx 260$ K and pressures $\approx 1000$ Pa, only benzene vapour exists. As the pressure is increased, a vertical movement in the diagram, the vapour condenses to the solid phase. Not until the temperature reaches $\approx 280$ K is there enough energy for the vapour to form the liquid phase when the pressure is increased. Below this temperature, the solid sublimes. Notice also how vertical the phase transition line between solid and liquid is. In many textbooks, it is misrepresented as a somewhat sloping line. The solid-liquid boundary will still appear as an almost vertical line if the graph is plotted with the log of the pressure up to $10^6$ Pa.
# 
# **Exercise:** Now that you can produce any gas-liquid-solid phase diagram, calculate these for water and carbon dioxide  using the data given below or choose some molecules of your own.
# 
# Some data is in the table
# 
# $$\displaystyle \begin{array}{lll}
# \hline
#  &\mathrm{water} &  \mathrm{carbon \,dioxide} & & \\
# \hline
# \mathrm{molec \,weight, g/mol} & 18.01528 & 44.012 \\
# \mathrm{Triple \,point, K} & 273.16 & 216.55 \\ 
# \mathrm{Triple \,point}& 611.29 \,\mathrm{ Pa} & 5.185\,\mathrm{ bar}\\ 
# \mathrm{H_{fus},\, kJ/mol} & 6.01 & 196.10/44.012 &\text{at triple point} \\
# \mathrm{H_{vap},\, kJ/mol} & 43.990 & 571.08/44.012&\text{at triple point}\\
# \mathrm{liquid \,density, g/cm^3 } & 0.99978 & 1.562\\
# \mathrm{solid\, density, g/cm^3 } & 0.917 & 1.032 \\
# \hline \end{array}$$
# 
# ### Q14 answer
# (a) $\displaystyle \left( \frac{\partial q}{\partial p} \right)_S =V$ integrates, at constant entropy, to become 
# 
# $$\displaystyle \int_{q_0}^{q}dq=\int_{p_0}^pVdp=\int_{p_0}^p\left( \frac{C}{p} \right)^{1/\gamma}dp$$
# 
# This integration is of the $x^{n+1}/(n + 1)$ form, equation 6, therefore,
# 
# $$\displaystyle \begin{align}q-q_0 =&\int_{p_0}^p\left( \frac{C}{p} \right)^{1/\gamma}dp \\ &=\frac{C^{1/\gamma} p^{1-1/\gamma}}{1-1/\gamma} \bigg|_{p_0}^p \\&=\frac{\gamma}{\gamma-1}C^{1/\gamma}\left( p^{(\gamma-1)/\gamma}-p_0^{(\gamma-1)/\gamma} \right) \end{align}$$
# 
# Dividing by $p_0$ and converting the change in enthalpy into velocity via $m.u^2/2=q_0 -q$ gives,
# 
# $$\displaystyle -m.u^2=\frac{2\gamma}{\gamma-1}C^{1/\gamma} p_0^{(\gamma -1)/\gamma} \left( \left(\frac{p}{p_0} \right)^{(\gamma-1)/\gamma}-1 \right)$$
# 
# and finally substituting $p_0V_0^\gamma = C$ and rearranging produces
# 
# $$\displaystyle u=\sqrt{\frac{2\gamma}{\gamma-1}\frac{p_0V_0}{m} \left( 1-\left(\frac{p}{p_0} \right)^{(\gamma-1)/\gamma} \right)  }$$
# 
# which is the equation describing the speed of a gas in a rocket or jet engine at pressure $p$. This equation can also be related to $T_0$, the inlet temperature, via $p_0V_0 = RT_0$.
# 
# (b) The volume of gas $V$ is given by $\displaystyle V=\left(\frac{p_0}{p}\right)^{1/\gamma}$  and because $\sigma m .u=V\mu$ the gas and hence nozzle cross section $\sigma$ varies as 
# 
# $$\displaystyle \sigma= \frac{\mu}{m}\left(\frac{p_0}{p}\right)^{1/\gamma}V_0 \left( \frac{2\gamma}{\gamma-1}\frac{p_0V_0}{m} \left( 1-\left(\frac{p}{p_0} \right)^{(\gamma-1)/\gamma} \right) \right)^{-1/2}$$
# 
# The equation $\sigma m .u = V\mu$ follows because the mass entering / sec must be the same as that leaving and this is why the gas speeds up at the throat. ($ \mu $ is the mass flow rate in kg sec$^{-1}$). The cross section vs reduced pressure $p/p_0$ is shown in figure 40 with all the constants set to one except $\gamma$ which is 1.22.
# 
# ![Drawing](integration-fig40.png)
# 
# Figure 40 Left: Universal calculated shape of gas cross-section in a jet or rocket (Laval) nozzle, vs reduced pressure $p/p_0$. Right: Gas velocity relative to the speed of sound us. Note that the inlet side is on the right in both figures where the pressure is high. The dashed lines show the position of the minimum nozzle width which is where the gas is at Mach 1.
# ______
# 
# (c) The minimum nozzle cross section vs pressure is the derivative of $\sigma$ vs $p$. This is not difficult to evaluate but messy and is easily performed by Sympy. The constants need not be included because the derivative is set to zero at the minimum and they will cancel out. (note g is used instead of $\gamma$ )

# In[3]:


p0, p, g = symbols('p0, p, g')
eq = (p0/p)**(1/g)/sqrt( 1 - (p/p0)**((g - 1)/g) )
simplify(diff(eq,p) )


# This equation must be zero at the minimum nozzle diameter and can be solved by factoring the bracket in the numerator to give the pressure ratio in the throat of  
# 
# $$\displaystyle \frac{p}{p_0}=\left( \frac{2}{\gamma+1} \right)^{\gamma/(\gamma-1)}$$
# 
# which, somewhat surprisingly, depends only on the ratio of heat capacities $\gamma$. The gas velocity in the throat is obtained by substituting this pressure into the equation for the speed, giving 
# 
# $$\displaystyle U_s=\sqrt{\frac{2\gamma}{\gamma+1}\frac{RT_0}{m}}$$
# 
# which is the speed of sound in the gas under the prevailing conditions. 
# 
# It is interesting to note, and was pointed out by Reynolds a long time ago (1886), that the rate of gas discharge depends on the cross-sectional area and not on the backing pressure. The gas cannot move faster than the speed of sound through the nozzle, so increasing the input pressure has no effect once the gas is moving at this speed. The gas velocity is given by
# 
# $$\displaystyle u=u_s\sqrt{\frac{\gamma+1}{\gamma-1}\left( 1-\left( \frac{p}{p_0}\right)^{(\gamma-1)/\gamma} \right)}$$
# 
# which is shown in the figure. 
# 
# The gas first reaches supersonic speed Mach 1 at the throttle point, called 'choked flow', and increases thereafter as it expands into a vacuum. In a real molecular beam experiment in the lab, and presumably for a rocket, the exhaust gas is slowed down by the residual background gas present but, if the pressure is low, this occurs only at some distance from the nozzle; this stationary shock-wave (relative to nozzle) is called the Mach disc.

# In[ ]:




