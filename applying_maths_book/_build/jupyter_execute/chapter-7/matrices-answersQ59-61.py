#!/usr/bin/env python
# coding: utf-8

# # Solutions Q59 - 61

# ## Q59 answer
# (a) The moment about the C$_4$ axis, which projects out of the plane of the molecule, is $I =4m_F R^2$ if $m_F$ is the mass of a fluorine atom and $R$ the bond length. Its value is $4 \times 18.9984\times 1.6604\cdot 10^{-27} \times 4\cdot10^{-20}=5.04710\cdot 10^{-45}\,\mathrm{kg\,m^2}$.
# 
# (b) The moment about a diagonal is $2m_FR^2$, because only two F atoms are not on the axis.
# 
# (c) The moment about the edge, involves calculating the perpendicular distance of the Xe and F atoms from the axis. Using Pythagoras' theorem, the Xe atom is $R/\sqrt{2}$ from the axis and each F atom twice this. The moment is then $m_{Xe}R^2/2 + 2 \times 2m_FR^2$, which has a value $1.38\cdot10^{-44}\,\mathrm{ kg\,m^2}$ and is considerably larger than the other two moments because the heavy Xe atom is not on the rotation axis.
# 
# ## Q60 answer
# (a) and (b). All the masses are at a fixed distance from the z-axis and the total mass of all particles is $M$, therefore, by definition $I_z = MR^2$ we can write down the answer $MR^2$ directly. By the perpendicular axis theorem, 
# 
# $$\displaystyle I_x = \frac{MR^2}{2} = I_y$$
# 
# This value is smaller than about the z-axis because the mass is further from this axis than from either the $x$ or $y$ axes.
# 
# (c) Inertia about the edge can be calculated by the parallel axis theorem, and should be that about the centre of mass of the z-axis plus $MR^2$ , or $I = 2MR^2$ . This is greater than $I_z$ because the mass is further from the axis on average, than is motion about the centre of the loop.
# 
# ## Q61 answer
# Let $\delta$ be the distance the centre of mass is from the C atom; the molecule is labelled as shown in Figure 70. The centre of mass (or gravity) is situated at 
# 
# $$\displaystyle q_{cm}=\frac{\displaystyle \sum_{i=1}^3m_ix_i}{\displaystyle \sum_{i=1}^3}=\frac{m_H(r_{HC}+\delta)+m_C\delta+m_N(r_{CN}-\delta)}{M}$$
# 
# where $M$ is the total mass. The distance $q_{cm}$, has to be zero, so that the axis of the moment of inertia passes through this point, therefore
# 
# $$\displaystyle m_H(r_{CH}+\delta)+m_C\delta+m_N(r_{CN}-\delta)=0$$
# 
# producing $\displaystyle \delta = \frac{m_Nr_{CN}-m_Hr_{CH}}{M}$.
# 
# The moment of inertia is, by definition, 
# 
# $$\displaystyle I = m_H(r_{CH} + \delta)^2 + m_C\delta^2 + m_N(r_{CN} - \delta)^2$$
# 
# into which $\delta$ has to be substituted. The result, while complicated, has two unknowns, $r_{CH}$ and $r_{CN}$ , but from one isotope only one moment of inertia is measured. Measuring spectra from different isotopes, enables both these bond lengths to be found by making the questionable, but almost universal assumption, that the force constant for the bond is unchanged by isotopic substitution.
# 
# The bond lengths are given in the text as: $r_{CH} = 1.066, r_{CN} = 1.153 \,A $, and conveniently the atomic masses for the isotopes are given in many chemistry and physics textbooks and are, in  kg,$^{12}C=1.9926\cdot 10^{-26};\;  ^{14}N=2.2352\cdot 10^{-26};\; ^1H=1.6735\cdot 10^{-26}$. By substitution into the formulae, these values produce $\delta = 0.5579\,\overset{o}A ;\; I=1.8850\cdot 10^{-46}\,\mathrm{kg\,m^2}$.
# 
# Some calculated moments of inertia for common C, N and H isotopes are shown below. The rotational constants are $B = \hbar^2/2I$ (Joules) but are given in cm$^{-1}$ in the table.
# 
# $$\displaystyle \begin{array}{c|cc}
# \hline
# I = m\times 10^{-46} &  ^{14}N &  ^{15}N \\ 
# \hline
# ^1H^{12}C & m = 1.8850\; B = 1.4850 &  m = 1.9515\; B = 1.4418\\
# ^1H^{13}C&  m = 1.9350\; B = 1.4467 & m = 1.9955\; B = 1.4027\\
# ^2H^{12}C & m = 2.3098\; B = 1.2119 & m = 2.3781\; B = 1.1771\\
# \hline
# \end{array}$$
# 
# The lines in the rigid rotor spectrum are separated by $2B$. Changing the C and N isotopes, changes the line positions by approximately a few times $10^{-2}\,\mathrm{cm^{-1}}$, illustrating the need for high spectral resolution. The lines with deuterium substitution are far more widely separated; $\approx 0.2\,\mathrm{cm^{-1}}$, because of the approximate doubling in mass of D to H.

# In[ ]:




