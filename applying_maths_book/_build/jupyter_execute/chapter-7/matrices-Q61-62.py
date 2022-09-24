#!/usr/bin/env python
# coding: utf-8

# # Question 61 - 62

# ## Q61 Moments of inertia of HCN
# 
# (a) Calculate the moment of inertia of the linear molecule HCN, using masses $m_N, m_H$ and $m_C$ for the most common isotopes of the atoms and bond lengths $r_{CH}$ and $r_{CN}$ given in the text. Explain why isotopic substitution is necessary to obtain the bond lengths. The parameters needed are defined in Figure 70.
# 
# (b) Calculate the numerical value of the com displacement δ, Fig. 7.70, then the moment of inertia I using the values of the bond length given in the text for the isotope containing 1H, 14N and 12C atoms.
# 
# (c) Assuming rigid-rotor behaviour, calculate by how much a rotational transition will shift if $^{14}$N and then $^{13}$C isotopes are used.
# 
# **Strategy:** On the diagram of the molecule choose a point to represent the centre of mass (com) and find its $x$ coordinate. Write down the equation for the moment of inertia based on this point, and the equation for the centre of mass and assume that this is at $x = 0$.
# 
# ![Drawing](matrices-fig70.png)
# 
# Figure 70. A linear triatomic molecule.
# 
# ## Q62 Moments of inertia of molecules
# Use an X-ray database to find coordinates for CH$_3$F, CH$_2$Cl$_2$, chlororethylene (CH$_2$CHCl), and any other molecules you want. Calculate the moments of inertia and plot the inertia axes on top of the molecular structure. The values you obtain should be approximately as given in the table, which are in units of $10^{-47}\,\mathrm{ kg\,m^2}$:
# 
# $$ \begin{array}{c|ccc}
# \hline
#    & I_a & I_b & I_c\\ 
# \hline
# \mathrm{CH_3F}    & 5.3 &     & 33\\
# \mathrm{CH_2Cl_2}  & 26  & 255 & 276\\
# \mathrm{CH_2CHCl} & 15  & 139 & 154\\ 
# \hline 
# \end{array}$$

# In[ ]:




