#!/usr/bin/env python
# coding: utf-8

# ## Solutions Q1 - 21

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sympy import *
init_printing()                      # allows printing of SymPy results in typeset maths format
plt.rcParams.update({'font.size': 14})  # set font size for plots


# ### Q1 answer
# Following the example of the harmonic oscillator, the equation becomes $v^2 - \omega^2x^2 = c$. This is the equation of hyperbolae and has a saddle point at $\omega\cdot x$ = 0. The plot is of $v\equiv dx/dt$ vs $\omega x$ with arbitrary values of $\omega$ and $c$ that we can choose. The figure shows the plot by choosing $\omega = c = 1$. Different values alters the 'look' of the plot, extending in one direction or the other but it always has the same form.
# 
# ![Drawing](diffeqn-fig31.png)
# 
# Fig. 31 Phase portrait for $d^2x/dt^2 - \omega^2x = 0. \quad \omega = c = 1$.
# 
# ### Q2 answer
# (a) $\displaystyle \frac{dy}{dx}=\frac{xy}{1-x^2}$ or $\displaystyle \int\frac{dy}{y}=\int\frac{x}{1-x^2}dx$. 
# 
# Integrating produces $\ln(y)=-\ln(x^2-1)/2+c$. When $y_0=c,\; x = x_0$ 
# 
# and then $\displaystyle \ln\left(\frac{y}{y_0}\right)= \ln\left(\frac{x_0^2-1}{x^2-1}\right)/2$ 
# 
# which can be simplified further to $\displaystyle \frac{y}{y_0}=\sqrt{\frac{x_0^2-1}{x^2-1}}$.
# 
# (b) $\displaystyle \frac{dy}{dx}=e^x\tan(y)$ or $\displaystyle \int \frac{dy}{\tan(y)}=\int e^xdx$. integrating produces $\displaystyle \ln(\sin(y))=e^x+c$. 
# 
# When $y_0=c, \; x=x_0$ then $\displaystyle \ln \left( \frac{\sin(y)}{\sin(y_0)}  \right)=e^x-e^x_0$
# 
# (c) Rearranging and integrating gives $\displaystyle \int_0^ye^ydy=\int_0^x e^x\sin(x)dx$, where the limits are included in the integral. The right-hand side can be integrated by parts; the final result is
# 
# $$\displaystyle e^y-1=\left[\sin(x)-\cos(x)\right]\frac{e^x}{2}+\frac{1}{2}$$
# 
# ### Q3 answer
# In a bimolecular reaction,when both species have the same initial concentration or the scheme is 2$A$ $\to$ product, then the rate equation is 
# 
# $$\displaystyle dA/dt = -k_2A^2$$
# 
# where $A$ is the concentration at time $t$, and the initial conditions $A(0) = A_0$. Integrating produces 
# 
# $$\displaystyle \int\frac{dA}{A^2}=-k_2t+c$$
# 
# which produces 
# 
# $$\displaystyle \frac{1}{A}-\frac{1}{A_0}=k_2t$$
# 
# and a plot of reciprocal concentration vs. time is linear. The reaction can also be solved by considering that an amount $x$ reacts at time $t$ if $a$ is the initial concentration then 
# 
# $$\displaystyle \frac{d(a-x)}{dt}=-k_2(a-x)^2,\quad\text{or}\quad \displaystyle \int\frac{dx}{(a-x)^2}=k_2t+c$$
# 
# producing $\displaystyle \qquad\frac{1}{a-x}-\frac{1}{a}=k_2t$.
# 
# ### Q4 answer
# (a) The integration produces 
# 
# $$\displaystyle \int\frac{dx}{x(N-x)}=kt, \quad \ln\left( \frac{x}{x-N} \right) =NkT+c$$
# 
# and with the initial condition and some rearranging $\displaystyle x=\frac{N}{1+(N-1)e^{-Nkt}}$.
# 
# As a check, at zero time $x = 1$ and at long times $x \to N$. The reason that $x = 1$ at time zero is that at least one individual must be infected. At long time all are infected. The same calculation using SymPy is a little involved as the initial conditions have to be treated separately. 

# In[2]:


x, t, k, N, C1 = symbols('x, t, k, N, C1' )       # use SymPy , define symbolic variables
x = Function('x')
f01 = Derivative( x(t),t) -k*x(t)*( N - x(t) )    # eqn 7 
ans = dsolve(f01 )
ans


# In[3]:


ans1 = ans.subs(t,0).subs(x(0),1)   # substitute with initial conditions
ans1


# In[4]:


c1_s = solve(ans1,C1)               # solve for constant
c1_s


# In[5]:


simplify( ans.rhs.subs(C1, c1_s[0] ) )  # substitute and simplify


# (b) The time for half the population to be infected occurs when $x = N/2$ giving $t_{1/2} = \ln(N − 1)/Nk$. This makes sense if $k$ is small then the spreading of the disease is slow and $t_{1/2}$ large, and if the population is large it also takes some time for half of them to become infected.
# 
# ![Drawing](diffeqn-fig31b.png)
# 
# Fig 31-B Example plots of how the number of infected individuals increases with time for different rate constants, $k$.
# _____
# 
# ### Q5 answer
# Scattering takes the place of absorption in the case of X-rays. According to the text, Section 1.11, the scattering cross section for each of $n$ electrons is $\sigma = 4\pi e^2/mc^2$. Following reasoning in the text leading to the Beer-Lambert law, 
# 
# $$\displaystyle I_s = I_0(1 - e^{-\sigma nL})$$
# 
# and when the scattering is small the exponential can be approximated giving $I_s/I_0 = \sigma nL$. For one electron, the amount scattered is $nL$ times less giving the required answer.
# 
# ### Q6 answer
# The force downwards is $mg - kv^2$ and the equation of motion is 
# 
# $$\displaystyle  m\frac{dv}{dt} = mg - kv^2, \quad v(t_0) = v_0, \quad t_0 = 0 $$
# 
# which using the abbreviation $b^2=k/mg$ gives $\displaystyle m\frac{dv}{dt} = g(1 - b^2v^2)$. The solution to this equation is given by 
# 
# $$\displaystyle \int \frac{dv}{1-b^2v^2}=gt$$
# 
# which produces $\tanh^{-1}(bv)= b(t+c)$.
# 
# The hyperbolic tan can be be converted to log form with the standard identity $\displaystyle 2\tanh^{-1}(ax)=\ln\left( \frac{1+ax}{1-ax}\right)$, hence the result is 
# 
# $$\displaystyle \ln \left(\frac{1+bv}{1-bv}  \right) = 2b(t+c)$$ 
# 
# To evaluate the constant when the initial velocity $t_0$ is not zero produces a complicated expression, which only evaluates to a number and so it is left as $c$ for clarity. It is effectively a displacement in time. Rearranging to find the velocity gives
# 
# $$\displaystyle v=\frac{1}{b}\left(\frac{e^{2b(t+c)}-1}{e^{2b(t+c)}+1} \right) $$
# 
# At long times, the limiting velocity is $v_{term} = 1/b = \sqrt{gm/k}$. The larger $k$ is, the smaller the velocity on reaching the ground; a typical value would be $1000$ kg/m making a terminal velocity for a $70$ kg skydiver $\approx $ 0.8 m/s.
# 
# ### Q7 answer
# Differentiating both sides with respect to $t$ gives, $\displaystyle \frac{dx}{dt}\left(\frac{1}{a-x}+\frac{1}{(a-x)^2}  \right)=ak$, 
# 
# which simplifies to $\displaystyle \frac{dx}{dt} = k(a - x)^2$. This means that the rate equation has the form of $da/dt=ka^2$ because a moles of A and B are present and $x$ moles react at time $t$. As a check, if the scheme is
# 
# $$\displaystyle \begin{array}{ccc}\\
#   A & + & B &\to &C\\
# (a-x)& & (a-x)&  &x
# \end{array}$$
# 
# the rate equation is $d(a - x)/dt = -k(a - x)^2$ which is the same equation as just obtained. The rate constant $k$ has second order units, $\mathrm{dm^3\, mol^{-1}\, s^{-1}}$.
# 
# ### Q8 answer
# Newton's law of cooling is 
# 
# $$\displaystyle \frac{d\theta }{dt}=k(\theta_s-\theta)$$
# 
# if $\theta$ is the temperature of the water and $\theta_s$ the surroundings. Integrating with the initial condition that at $t = 0, \,\theta = \theta_0$ gives 
# 
# $$\displaystyle \theta_T - \theta_s = (\theta_0 - \theta_s )e^{-kt}$$ 
# 
# The initial temperature is $15^\mathrm{o}$, the surroundings $-12^\mathrm{o}$, and the fall is $5^\mathrm{o}$ in $8$ minutes, then $\theta_T = 10$. The rate constant $k$ is 
# 
# $$\displaystyle k=\frac{1}{t}\ln\left(\frac{\theta_0-\theta_s}{\theta_T-\theta_s} \right) = 0.0256\,\mathrm{ min^{-1}}$$
# 
# To form ice the temperature must be zero ($\theta_T = 0$) or lower, so that $\displaystyle t=\ln(27/12)/k = 31.7$  mins.
# 
# ### Q9 answer
# The process is solid $\to$ vapour and is first order because it is proportional to the amount of solid remaining at any time. If $s$ is the amount of solid I$_2$ then the rate equation is $\displaystyle \frac{ds}{dt} = -ks$ which integrates to $\displaystyle s = s_0e^{-kt}$ if $s_0$ is the initial amount of iodine. The rate equation for the amount of iodine vapour is 
# 
# $$\displaystyle \frac{ds_V}{dt}=+ks=ks_0e^{-kt}$$
# 
# and integrating with $s_V=0$ initially produces $\displaystyle s_V=s_0(1-e^{-kt})$.
# 
# The pressure is related to the concentration and using the ideal gas law gives $s_V = pV/RT, \; s_0 = p_\infty V/RT$  and therefore the pressure increases as $\displaystyle p = p_\infty(1 - e^{-kt})$.
# 
# ### Q10 answer
# (a) The fluorescence yield is the rate of emission divided by the rate of absorption. This is $\varphi = k_f /(k_f + k_S)$, equation (5). With quencher the scheme is, 
# 
# $$\displaystyle \begin{array}{ccc}\\
#  G       & \stackrel {k_a}\longrightarrow S_1 &   & \\
#  S_1     & \stackrel {k_f}\longrightarrow G   & + & h\nu\\
#  S_1     & \stackrel {k_S}\longrightarrow T   &   &\\
#  S_1 + Q & \stackrel {k_q}\longrightarrow G   & + & Q 
# \end{array}$$
# 
# Using $S_1$ to represent the concentration $[S_1]$ etc., at steady state,
# 
# $$\displaystyle \frac{dS_1}{dt}=k_aG-(k_f+k_S+k_q)S_1=0$$
# 
# therefore as the fluorescence yield $\varphi$ is the rate of emission ($k_f S_1$) divided by rate of absorption ($k_aG$),
# 
# $$\displaystyle \varphi_Q = \frac{k_fS_1}{ (k_f+k_S+k_QQ)S_1 } =\frac{k_f}{k_f + k_S + k_QQ} $$
# 
# Forming the ratio $\varphi /\varphi_Q$ produces the Stern - Volmer equation with $k_{\mathrm sv} = k_q /(k_f + k_S)$.
# 
# **(b)** The fluorescence lifetime is by definition the reciprocal of the sum all rate processes destroying the excited state, therefore $\tau_q=(k_f+k_S+k_qQ)^{-1}$ and in the absence of quencher, $\tau=(k_f+k_S)^{-1}$. The ratio of these two lifetimes gives a second Stern - Volmer equation 
# 
# $$\displaystyle  \frac{\tau}{\tau_q}=\frac{k_f+k_S+k_qQ}{k_f+k_S}=1+k_{\mathrm{sv}}Q$$
# 
# ### Q11 answer
# The rate equations are 
# 
# $$\displaystyle \frac{dA}{dt}=-k_AA, \; \frac{dB}{dt}=+k_1A-k_0$$
# 
# if $A\equiv [A],\; B\equiv [B]$. The concentration of A is $\displaystyle A=A_0e^{-k_1t}$ and putting this back into the second rate equation gives 
# 
# $$\displaystyle \frac{dB}{dt}=k_1A_0e^{-k_1t}-k_0$$
# 
# Integrating produces $\displaystyle B=-A_0e^{-k_1t}-k_0t+c$. 
# 
# The initial condition is that $[B]_0=0$ therefore,
# 
# $$\displaystyle  B=A_0(1-e^{-k_1t})-k_0t$$
# 
# The maximum occurs when 
# 
# $$\displaystyle \frac{dB}{dt}=0= k_1A_0e^{-k_1t} -k_0 $$
# 
# which gives $\displaystyle t_{max}=\frac{1}{k_1} \ln \left(\frac{k_1A_0}{k_0} \right)$. 
# 
# The maximum B concentration is at $\displaystyle B_{max}=A_0-\frac{k_0}{k_1}-k_0t_{max}$.
# 
# (b) Using the values in the question, $t_{max} = 0.68$ hour and $B_{max} = 0.99\,\mathrm{ g\, dm^{-3}}$. The time to reach a blood alcohol level of $B_c = 0.8\,\mathrm{ g\, dm^{-3}}$ is found from $\displaystyle B_c = A_0(1 - e^{-k_1t}) - k_0t$, which is transcendental and has to be solved numerically. However, the time can be obtained approximately, when $k_1t$ is large and is then $B_c \approx A_0 - k_0t$, and is $\approx 1.9$ hours. Using the Newton - Raphson method the more accurate result is $1.93$ hours.
# 
# The calculation using Sympy is shown below and the generic figure shows the change with time.

# In[6]:


# define function to solve numerically
#---------------------
def f(t):
    return Bc - A0*(1.0 - np.exp( -k1*t)) + k0*t
#----------------------

Bc = 0.8              # parameters
k1 = 5.0
k0 = 0.19
A0 = 70.0/60.0
t0 = [0.1 ,2.5]             # initial guesses
roots = fsolve(f, t0)
print('{:s} {:f} {:s} {:f} {:f} '.format('times at conc', Bc, 'are ',roots[0],roots[1] ) )


# ![Drawing](diffeqn-fig32.png)
# 
# Fig32. The equation $\displaystyle B=A_0(1-e^{-k_1t})-k_0t$ plotted with values given in the question. The concentration is in mg/dm$^3$ thus a line at $0.8$ is equivalent to $80$ mg/ml blood alcohol (dashed horizontal line). In Scotland the legal limit is lower than England and is $50$ mg/ml blood alcohol (dotted line) , meaning that one would be 'over the limit' by far more and for far longer than in England.
# ____
# 
# When no alcohol remains the residual level is zero, and this occurs $6$ hours $8$ minutes after starting to drink. The graph of the time profile of alcohol in the blood is shown in Fig. 32. Because the amount of alcohol decreases linearly with time, at least when the amount is moderate, this can easily be used to back calculate how much alcohol was present at an earlier time. You can see how the model fails when B is small; it predicts a negative B will occur. At very low alcohol levels the reaction is no longer zero order but bimolecular and the concentration does not become negative.
# 
# ### Q12 answer
# The total flux into the pellet is, $\displaystyle J=+4\pi r^2D\frac{dc}{dr}$ 
# 
# and is positive as diffusion is into the pellet with a surface area of $4\pi r^2$. Defined as total flux, $J$ has units of mole s<sup>-1</sup>. As the catalyst poisons to radius $r_p$, then its concentration is given by
# 
# $$\displaystyle \int_{c_p}^{c_0}dc=\frac{J}{4\pi D}\int_{r_p}^{r_0}\frac{dr}{r^2},\quad c_0-c_p=-\frac{J}{4\pi D}\left(\frac{1}{r_0}-\frac{1}{r_p}  \right) $$
# 
# If the flux into the pellet is equal to the rate of reaction then $J = kc_p$, which after rearranging gives,
# 
# $$\displaystyle c_p=c_0\left[1- \frac{k}{4\pi D}\left(\frac{1}{r_0}-\frac{1}{r_p}\right) \right]^{-1} $$
# 
# To see what this function looks like, it can be rearranged to 
# 
# $$\displaystyle \frac{c_p}{c_0}=\left[ 1+\frac{\alpha x}{1-x}  \right]^{-1}$$
# 
# where $x$ is a fraction of the radius $r_p=(1-x)r_0$ and $\alpha= k/(4\pi Dr_0)$. Two curves are shown in Fig. 33 when $\alpha \gt 1$ and when $\lt 1$. When $\alpha \lt  1$, this means that the diffusion coefficient is large or the rate constant $k$ small. The concentration of the poison in the pellet is more than 50% almost over the whole volume of the pellet. When diffusion is slow, or the rate constant is large, the concentration of the poison decreases rapidly away from the surface and more than $50$% poisoning only occurs within about 10% of the surface.
# 
# ![Drawing](diffeqn-fig33.png)
# 
# Fig. 33 Relative or fractional concentration of the poison vs. the fraction of the pellet's radius at two different values of $\alpha$. The value $x$ = 0 means that $r_p = r_0$ and $x$ = 1 means that $r_p$ = 0.
# _____
# 
# ### Q13 answer
# As M and Q react, the rate is $\displaystyle k_2MQ = JM = +4πr^2DM \frac{dQ}{dr}$. 
# 
# To find $J$ the last two terms are integrated from $r = R$ to infinity,
# 
# $$\displaystyle  \frac{J}{4\pi D}\int_R^\infty\frac{dr}{r^2}=\int_0^QdQ$$
# 
# which gives $J=4\pi DRQ$. Equating this to the rate produces the diffusion controlled rate constant $k_2=4\pi DR$.
# 
# ### Q14 answer
# The rate equation is $\displaystyle \frac{dR}{dt} = k_IM + (\alpha - 1)k_BR + k_pR - k_pR - k_TR$  
# 
# and the propagation step cancels out because in this step, one radical is produced for each one consumed. The branching step has the term $\alpha$ - 1 because one radical is lost accounting for propagation. The integral equation from a radical concentration of zero at zero time is
# 
# $$\displaystyle \int_0^R\frac{1}{k_IM+\gamma R}dR=\int_0^tdt$$
# 
# where for clarity $\gamma = (\alpha - 1)k_B - k_T$. Integrating produces $\displaystyle \ln\left(1+\frac{\gamma R}{k_IM} \right)=\gamma t$ which is more conveniently written as $\displaystyle R = k_IM(e^{\gamma t} - 1)/\gamma$. 
# 
# When $\gamma \gt 0$, this means that the branching rate is larger than the termination rate, and an explosion may occur because the term $\gamma t$ in the exponential is positive. The radical population increases towards infinity and hence the rate of reaction increases rapidly. Reaction is stable when $\gamma \lt 0$ and reaches steady state at long times; $R_\infty = k_I M/\gamma$. When $\gamma = 0$ the reaction is on the border between stability and explosion and radical concentration increases linearly with time as $k_IMt$.
# 
# ### Q15 answer
# (a) Equilibrium is rapidly established with the complex IM and its rate of change can be taken to be at steady state, hence 
# 
# $$\displaystyle \frac{d[IM]}{dt} = k_1[I][M] - k_{-1}[IM] - k_2[IM][I] = 0$$
# 
# giving $\displaystyle [IM]=\frac{k_1[I][M]}{k_{-1}+k_2[I]}$. The appearance of I$_2$ molecules is 
# 
# $$\displaystyle \frac{d[I]}{dt}=-k_2[IM][M]=-\frac{k_2k_{-1}}{k_1}[M][I]^2$$
# 
# Integrating this expression with the initial condition that $[I] = [I]_0$ at zero time gives,
# 
# $$\displaystyle  \frac{1}{[I]}-\frac{1}{[I]_0}=k_2\frac{k_{-1}}{k_1}[M]t$$
# 
# which explains the linear dependence of $1/[I]$ with time.
# 
# (b) If each of the rate constants follows an Arrhenius type expression $\displaystyle k_a = k_0e^{-E_a/RT}$ with activation energy $E_a$, then, from the rate expression the term $k_2k_{-1}/k_1$ gives the overall reaction an activation energy $E_2 + E_{-1} - E_1$. If $E_1$ is greater than the other two, then the experimentally measured activation energy is negative even though each step has a positive or zero activation energy as must always be the case.
# 
# ### Q16 answer
# (a) The scheme is solvable if $x$ is the amount consumed at time $t$,
# 
# $$\displaystyle \begin{array}{ccc}
# \mathrm{A} &\underset{k_{-1}}{\stackrel {k_1}{\leftrightharpoons}} & \mathrm{B}\\
# a-x & & b+x
# \end{array} \qquad \qquad  
# \text{then} \qquad \qquad \displaystyle \frac{dx}{dt}=k_1(a-x)-k_{-1}(b-x) $$
# 
# in which the variables are separable. The initial conditions are $x = 0$ when $t = 0$ producing the integration,
# 
# $$\displaystyle \int_0^x\frac{dx}{k_1a-k_{-1}b+(k_1+k_{-1})x}=t$$
# 
# This is a standard integration (see Chapter 4) and evaluates, after rearranging, to 
# 
# $$\displaystyle x=\frac{k_1a-k_{-1}b}{k_1+k_{-1}}\left(1-e^{-(k_1+k_{-1})t}\right)$$
# 
# which demonstrates how the amount of $x$ increases from zero to a constant value as equilibrium is approached.
# 
# (b) The equilibrium amount $x_e$ is found when $t \to \infty$, or using the rate expression when the rate of change is zero, $k_1(a-x_e)-k_{-1}(b+x_e)=0$, from which 
# 
# $$\displaystyle x_e =\frac{k_1a-k_{-1}b}{k_1+k_{-1}}$$
# 
# and hence 
# 
# $$\displaystyle x=x_e\left(1-e^{-(k_1+k_{-1})t}  \right)$$
# 
# (c) The amount $x=\alpha_t -\alpha_0$ and $x_e =\alpha_\infty -\alpha_0$ then 
# 
# $$\displaystyle \frac{\alpha_t -\alpha_0}{\alpha_\infty -\alpha_0}=\left(1-e^{-(k_1+k_{-1})t}  \right)$$ 
# 
# Because the change with time is proportional to the ratio of the measured quantity, the units of this do not matter because they cancel out. This is always the case for first-order reactions.
# 
# (d) The same rate expression in (a) written in terms of the change $\Delta x=x-x_e$ is 
# 
# $$\displaystyle \frac{d} {dt}(\Delta x+x_e)=k_1(a-x_e-\Delta x)-k_{-1}(b+x_e+\Delta x)$$
# 
# which simplifies to 
# 
# $$\displaystyle \frac{d} {dt}(\Delta x)=-(k_1+k_{-1})\Delta x$$
# 
# When integrated, this produces 
# 
# $$\Delta x = \Delta x_0e^{-(k_1+_{k-1})t}$$
# 
# showing that the relaxation is first-order.
# 
# (e) Using the rate constant from the last parts of the calculation and the expression for the rate constant given in the question, produces
# 
# $$\displaystyle k=k_1+k_{-1}=k_0e^{-E_0/RT}\left( e^{-V/2RT}+e^{+V/2RT}   \right)=k_0e^{-E_0/RT}\cosh\left(\frac{V}{2RT}  \right)$$
# 
# The $\pm V/2$ arises because the potential is zero in the centre of the membrane and is therefore $-V/2$ on one side and $+V/2$ at the other. When $V = 0$, the rate constant $\displaystyle k = 2k_0e^{-E_0 /RT}$ and this is its minimum value as $\cosh(x)$ has an approximately parabolic shape. Using the value $V = 0.1$ eV given in the question, produces $\cosh(9.6/(2RT)) \approx 3.5$.
# 
# 
# ### Q17 answer
# Let $x$ be the amount of H+ and OH- ions and $a$ the concentration of water, then the rate equation is 
# 
# $$\displaystyle \frac{dx}{dt} = -k_1(a - x) - k_2x^2$$
# 
# This equation is separable and can be integrated with the initial conditions $x$ = 0 at $t$ = 0 but the result is complex. Changing $x$ to $\Delta x + x_e$ makes the equation into
# 
# $$\displaystyle \frac{d(\Delta x+x_e)}{dt}=\frac{d\Delta x}{dt}=-k_1(a-x_e-\Delta x)=k_2(x_e+\Delta x)^2 $$
# 
# As the perturbation $\Delta x$ is small, then $\Delta x^2$ is smaller and can be ignored as it is less than $2x_e\Delta x$ and $x_e^2$ and this produces 
# 
# $$\displaystyle \frac{d\Delta x}{dt} = -k_1(a - x_e - \Delta x) - k_2x_e^2 - 2k_2x_e\Delta x$$
# 
# To eliminate $a$, use the rate equation at equilibrium $dx/dt = 0$ to obtain $k_1(a - x_e) = k_2x_e^2$. The final equation is 
# 
# $$\displaystyle \frac{d\Delta x}{dt} = -(k_1 + 2k_2x_e)\Delta x$$
# 
# which is a first-order equation with solution 
# 
# $$\displaystyle \Delta x = \Delta x_0e^{-kt}$$
# 
# where $x_0$ is the initial change recorded in the experiment and $k = k_1 + k_2x_e$. It does not matter what units $\Delta x$ is measured in, because it can always be written as the ratio $\Delta x/\Delta x_0$ when solving for the rate constant.
# 
# ### Q18 answer
# (a) If $x$ is the acetone concentration (species A), the rate equation is, 
# 
# $$\displaystyle \frac{dx}{dt}=-kxb=-kx(a_0+b_0-x)\quad\text{or}\quad \displaystyle \int\frac{dx}{(a_0+b_0-x)x}=-kt+c$$
# 
# which can be integrated using partial fractions;
# 
# $$\displaystyle \frac{1}{(a_0+b_0-x)x}=\frac{1}{(a_0+b_0)}\left[\frac{1}{x}+\frac{1}{a_0+b_0-x} \right]$$
# 
# which produces two log functions when integrated. The initial conditions are $x = a_0$ when $t = 0$ then the integration constant is $\ln(a_0/b_0)$ and the rate equation,
#   
# $$\displaystyle  \ln \left(\left| \frac{xb_0}{a_0(a_0+b_0-x)}     \right|  \right) =-(a_0+b_0)kt$$
#   
# and the absolute value is added because the log cannot be negative. Rearranging produces 
# 
# $$\displaystyle x = a_0\frac{a_0+b_0}{a_0+b_0e^{-(a_0+b_0)kt}} $$
# 
# This and also the increase of $B$ are plotted in Fig. 34.
# 
# (b) The phase portrait is an inverted parabola; see Fig.34, Right. The right-hand equilibrium point is unstable, so that from any starting value, $x$ will end up at zero which means complete reaction occurs. The stability is determined by $d^2x/dt^2$ and is positive at $x = a_0 + b_0$, so this point is unstable and negative at $x = 0$ which is a stable point.
# 
# ![Drawing](diffeqn-fig34.png)
# 
# Fig. 34 Autocatalysis. Initially species A decreases slowly as the concentration of B is small. As the reaction proceeds, more B is produced and although A is reduced overall, the rate increases and A is consumed even more rapidly. At longer times, the concentration of A becomes so small that even though that of B is large the reaction rate is slow. The phase portrait is shown on the right.
# ________
# 
# ### Q19 answer
# (a) Rearranging gives 
# 
# $$\displaystyle \int\frac{dn}{n\ln(n/a)}=-kt+c$$
# 
# at first seems rather hard. Using the substitution $n/a = e^u$, produces $dn = ae^udu$ and 
# 
# $$\displaystyle \int\frac{dn}{n\ln(n/a)}=a\int\frac{e^udu}{aue^u}=\int\frac{du}{u}=\ln(u)$$
# 
# Substituting back and adding the limits gives
# 
# $$\displaystyle \ln\big(\ln(n/a)\big) = -kt + \ln\big(\ln(n_0/a)\big) \qquad \text{ or } \qquad n = ae^{\ln(n_0 /a)e^{-kt} }$$
# 
# At long times $t \to \infty$, the inner exponential becomes very small and as $\displaystyle e^{-\infty} \to 0$, then $n \to a$ and the population becomes constant. Therefore, parameter $a$ must represent the maximum and limiting population. When $t = 0$, then the population is $n_0$. 
# 
# If the initial is less than the final population, this rises gradually to a constant value and the shape of the curve is approximately sigmoidal. If the initial population is greater than $a$, then the population falls approximately exponentially towards $a$. The parameter $k$ is the rate constant (unit = 1/time) controlling the rate at which the final population is reached.
# 
# (b) The dimensionless equation is $\eta = e^{\large{e^{-\tau}} \ln(\eta_0)}$
# 
# where $\eta = n/a,\; \eta_0 = n_0/a$ and $\tau = kt$ and the equation is shown in Fig. 35
# 
# ![Drawing](diffeqn-fig35.png)
# 
# Fig. 35 Population change according to the Gompertz equation.
# _____
# 
# (c) The equation for $B$ is found directly and is $B=Be^{-k_1t}$. Substituting into the equation for $A$ gives 
# 
# $$\displaystyle \frac{dA}{dt} = k_2AB_0e^{-k_1t}$$
# 
# which can be separated and integrated using the initial conditions;
# 
# $$\displaystyle \int\limits_{A_0}^A \frac{dA}{A}= k_2B_0\int\limits_0^t e^{-k_1t}dt$$
# 
# with the solution $\displaystyle \ln\left(\frac{A}{A_0}\right)=\frac{k_2B_0}{k_1}\left( 1-e^{-k_1t} \right) $. 
# 
# This has the form of the Gompertz equation, i.e., $\displaystyle A = A_0e^{\left((k_2B_0/k_1)(1-e^{-k_1t})\right)}$.
# 
# (d) In this model of a tumour, $k_2 = 1$, if $B_0$ is the initial rate of tumour growth, $k_1$ the regression constant and $A_0$ the number of cells initially present. At long times, $A \to A_0e^{B_0 /k_1}$ and if the tumour is to shrink then, $B_0/k_1 \lt 1$ making the exponential less than $1$.
# 
# ### Q20 answer
# (a) The rate of births is $k_1N$ and of deaths $(k_2 + k_3N)N$, thus the rate equation is 
# 
# $$\displaystyle \frac{dN}{dt}=k_1N-k_2N-k_3N^2 =[k_1-k_2-k_3N]N $$ 
# 
# The steady-state solution is found when the rate of change in population is zero and is either $N_{ss} = 0$ or $N_{ss} = (k_1 - k_2)/ k_3$. These are also called the fixed, stationary, or equilibrium points in the phase portrait. If $dN/dt$ is plotted vs. $N$, it is zero at the stationary points and in this case, positive in between. For simplicity, the non-zero steady state population is put into the equation and produces,
# 
# $$\displaystyle \frac{dN}{dt}=k_3(N_{ss}-N)N$$
# 
# from which we can see that if $N \gt N_{ss}$, the population will initially decrease over time; otherwise, it will increase until the steady state is reached, and this is shown in the left-hand pane of Fig. 36.
# 
# (b) The rate equation can be integrated by separating variables after first converting into partial fractions. The method is essentially the same as in the previous question. As an illustration SymPy is used here. The calculation is performed in stages as the initial conditions option does not currently work.

# In[7]:


k3, N_ss, t, N0, C1= symbols('k3, N_ss, t, N0, C1')   # SymPy define symbols to use
N  = Function('N')
f01= Derivative(N(t),t) - k3*(N_ss - N(t))*N(t)        # define and symbolically solve equation
ans = dsolve(f01)
ans


# In[8]:


const = solve( ans.subs(t,0).subs(N(0),N0) , {C1} )   # substitute initial conditions into answer to find C1
const[0]


# In[9]:


n_t = simplify(ans.rhs.subs(C1,const[0]))       # substitute into initial answer for C1 and simplify
n_t


# which can be re-written as 
# 
# $$\displaystyle n=\frac{N_0N_{ss}}{(N_{ss}-N_0)e^{-N_{ss}k_3t}+N_0}$$
# 
# which is more stable numerically as the exponential cannot become very large. With constants $k_1 = 2,\, k_2 = 1,\, k_3 = 0.001$, and $N_{ss} = 1000$, with $N_0$ varying from $200 \to 2000$ the curves in the left-hand pane of Fig. 36 are produced. As predicted, initially the population rises or falls to the steady-state value. If the rate constant  due to over-crowding, $k_3$, is too large, the steady state population will approach zero.
# 
# ![Drawing](diffeqn-fig36.png)
# 
# Fig. 36 Left: Population vs. time under 'normal' conditions for different initial numbers $N_0$ increasing by 200 for each curve. Right: Harvesting is introduced with  $k_h=200$ and $N_0$ values differing by 100 and starting at 100. Notice how the population can crash to zero if the initial number is below the lower of the two steady states, $N_{ss} \approx 276$. The values of the parameters used are given in the text.
# ______
# 
# (c) The rate equation is 
# 
# $$\displaystyle \frac{dN}{dt} =(k_1 -k_2)N-k_3N^2 -k_h$$
# 
# and the steady state is the solution of $k_3 N^2 - (k_2 - k_1 )N + k_h = 0$ which is 
# 
# $$\displaystyle N_{ss}=\frac{k_1-k_2\pm\sqrt{(k_1-k_2)^2-4k_3k_h}}{2k_3}$$
# 
# In the question, $k_1 - k_2$ = 1 (unit: time$^{-1}$), $2\sqrt{k_3k_h}$ = 0.89 making $N_{ss}$ = 723 and 276 if $k_h$ = 200 (unit: number . time$^{-1}$). (As $N_{ss}$ must be a real positive number $(k_1 - k_2)^2 \ge 4k_3k_h$).
# 
# However, solving the rate equation and plotting the values tells another story. At a low initial population as
# time passes, harvesting may destroy the population and then it becomes negative! The phase portrait allows this to be visualized, Fig. 37. If the initial population is less than the lower critical point, at $N = 276$ the population moves towards zero because this point is unstable. If the initial population is above this then the second stable point is reached at $N = 723$ as this point is stable. Thus, to determine whether the population will crash for a given set of parameters, the rate equation does not have to be solved but only the phase portrait examined. However, if you want to know how long it will take the population to crash or to recover, then the rate equation has to be solved.
# 
# ![Drawing](diffeqn-fig37.png)
# 
# Fig. 37 Phase portraits for the two population models. The lower curve (blue) corresponds to the model with harvesting. The circles indicate the steady state values; the arrows indicate how the population changes with time depending on its initial value.
# _____
# 
# ### Q21 answer
# (a) The singlet states decay by fluorescence and crossing to the triplets and formed by triplet annihilation.
# 
# $$\displaystyle \begin{align} \frac{dS}{dt} &= -(k_f + k_s)S + \frac{k_aT^2}{2} \\
# \frac{dT}{dt} &=k_sS-k_aT^2 -k_TT \end{align} $$
# 
# In the rate equation for the singlet state, because two triplets are indistinguishable from one another their amount must be halved. Under the conditions of the experiment, the triplet state rate equation can be simplified to
# $\displaystyle \frac{dT}{dt} = -k_aT^2 - k_TT $ which can be solved by separating the variables as 
# 
# $$\displaystyle \int \frac{dT}{(k_aT+k_T)T}=-t+c$$
# 
# and is integrated by separating into partial fractions. The result is 
# 
# $$\displaystyle \ln\left( \frac{T}{k_T+k_aT}  \right) = -k_Tt +c $$
# 
# As the initial condition for the triplet population is $T(0)=T_0$ the constant can be determined producing
# 
# $$\displaystyle \frac{1}{T}=\left(\frac{1}{T_0}+\frac{k_a}{k_T}  \right) e^{k_Tt}-\frac{k_a}{k_T}$$
# 
# The plot below shows that only at long times does the log plot become linear and gives the correct decay time for the triplet excited state. Rearranging the equation when $e^{k_Tt}>> 1$ i.e. large gives, 
# 
# $$\displaystyle  T = \frac{T_0}{1+k_aT_0/k_T}e^{-k_Tt} $$
# 
# which shows that the decay becomes exponential at long times. 
# 
# At short times, expanding the exponential produces
# 
# $$\displaystyle \frac{1}{T}=\frac{1}{T_0}+\left(\frac{1}{T_0}+\frac{k_a}{k_T}  \right)k_Tt$$
# 
# which is the equation of a second-order decay process because the reciprocal of the concentration $T$ is proportional to time. When the annihilation rate constant is small so that $k_a/k_T <<1$ the decay takes its normal exponential form.
# 
# Notice that any _delayed fluorescence_ will decay as $\displaystyle \approx e^{-2k_Tt}$, which is twice as fast as the triplets do, because of the $T^2$ term in its rate expression.
# 
# ![Drawing](diffeqn-fig38.png)
# 
# Fig. 38 The decay of the triplet population $T$ (in $\mathrm{dm^3\,mol^{-1}\,s^{-1}}$) vs. time. The effect of T-T annihilation on the T population for different triplet rate constants, $k_T$ (s$^{-1}$) is shown with $k_a=10^9\,\mathrm{ dm^3\,mol^{-1}\,s^{-1}}$. The initial triplet population is $10^{-4} \,\mathrm{ dm^3\,mol^{-1}\,s^{-1}}$.

# In[ ]:




