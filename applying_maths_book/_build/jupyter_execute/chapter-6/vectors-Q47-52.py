#!/usr/bin/env python
# coding: utf-8

# #  Questions 47 - 52

# ## Q47 Cross product
# Show that $\sin(\alpha - \beta) = \sin(\alpha)\cos(\beta )- \cos(\alpha)\sin(\beta)$. See problem Q25.
# 
# **Strategy:** As the problem involves cosines we could try to solve it by calculating the cross product of a and b as two unit vectors which have an angle $\alpha - \beta$ between them. The cross product is
# $\vec a \times\vec b = |\vec a||\vec b|\sin(\alpha - \beta)\;\boldsymbol k$ and we calculate this in two ways and compare the results.
# 
# ## Q48  'Law of sines'
# If the sides of a triangle have lengths $a, b, c$ prove the 'law of sines' for plane triangles,
# 
# $$\displaystyle \frac{\sin(\alpha)}{a}=\frac{\sin(\beta)}{b}\frac{\sin(\chi)}{c}$$
# 
# **Strategy:** Half of the cross product of any two sides of a triangle, taken as vectors, is equal to half its area. Let the sides be described by vectors $\vec A, \vec B$ and $\vec C$ and calculate the cross products using different pairs of sides.
# 
# ## Q49  Cross product
# Show that $\vec A \times (\vec B \times\vec C) = \vec B(\vec A\cdot\vec C) - \vec C(\vec A\cdot\vec B)$ by defining vectors in a three-dimensional basis set. Use python/Sympy to show that one side of the equation is the same as the other as this is simpler than doing it by hand.
# 
# ## Q50  Distance point to line
# Using figure figure 34 (section 16.3) 
# 
# (a) calculate the distance of point $B$ from the line $A$p, and 
# 
# (b) the distance of $A$ from the line $pB$.
# 
# ## Q51  Equilateral triangle
# Show that 
# 
# (a) the length of any perpendicular inside an equilateral triangle, of side $r$, to any vertex is $(r/2)\sqrt{3}$ and 
# 
# (b) (b) the lengths of the perpendiculars from any point $p$ inside the triangle to the sides add to give $(r/2)\sqrt{3}$.
# 
# **strategy:** All sides have the same length because the triangle is equilateral and the internal angles are $60^\text{o}$. (a) Determine the coordinates of the vertices then calculate one perpendicular distance. (b) choose a point $p$ inside the triangle with coordinates $(x, y, 0)$ and calculate the distance from $p$ to each side.
# 
# ## Q52  Fe-O2 bond length in Haem.
# The protein haemoglobin binds molecular oxygen to the Fe atom of the porphyrin chromophore of which there are four in this protein. This chromophore gives blood its characteristic purple venous and red arterial colours in its deoxygenated and oxygenated forms respectively. The binding of O$_2$2 causes the iron atom's d-orbital electronic energy levels to shift, causing a change in the haemoglobin's visible absorption spectrum. These changes are sufficiently quantitative that they are used as a non-invasive, optical monitor of the extent of blood oxygenation of patients in hospital.
# 
# In the haem protein, the Fe is sixfold coordinated, and the sixth position is taken by a nearby histidine residue whose bonding to the Fe pulls it out of the plane of the porphyrin. When the Fe atom releases the oxygen molecule, the porphyrin changes shape which results in the Fe atom moving more into the plane of the ring. The resulting force on the histidine is sufficient to trigger a shape change in the protein allowing the O$_2$ molecule to escape.
# 
# The following data is taken from the Brookhaven protein data bank (pdb) entry 1THB recorded at $1.5$ angstrom resolution (Waller & Liddington 1990). The data below contains only some of the $\approx 4900$ atom's coordinates, in $\overset{o}A$, of the histidine and Fe porphyrin.
# 
# (a) Calculate the O-O, Fe-O, and Fe-His nitrogen bond lengths. Look up the Fe-O and Fe-N bond lengths from other compounds and comment on the bond lengths in this protein.
# 
# (b) Compared to the plane set by the NA-NB-ND atoms, calculate how much out of the plane, histidine Nitrogen, Fe, and O atoms are.
# 
# $$\displaystyle \small \begin{array}\\
# \hline
# &&&&&& x,\qquad y,\qquad z&\\
# \hline
# \text{ATOM  }& 2930&  CD2& HIS& C&  87    &   7.609, -14.537,   9.600, & 1.00, 18.83 &     1THB3084 \\
# \text{ATOM  }& 2931&  CE1& HIS& C&  87    &   8.872, -16.338,  10.152, & 1.00, 25.86 &     1THB3085 \\
# \text{ATOM  }& 2932&  NE2& HIS& C&  87    &   7.742, -15.655,  10.429, & 1.00, 20.36 &     1THB3086 \\
# \text{HETATM}& 3353& FE  & HEM& C&   1    &   6.591, -16.663,  12.091, & 1.00, 22.73 &     1THB3507 \\
# \text{HETATM}& 3358&  N A& HEM& C&   1    &   7.559, -18.441,  12.096, & 1.00, 19.96 &     1THB3512 \\
# \text{HETATM}& 3369&  N B& HEM& C&   1    &   7.832, -16.106,  13.651, & 1.00, 22.85 &     1THB3523 \\
# \text{HETATM}& 3377&  N C& HEM& C&   1    &   5.405, -15.044,  12.620, & 1.00, 17.69 &     1THB3531 \\
# \text{HETATM}& 3385&  N D& HEM& C&   1    &   5.015, -17.506,  11.076, & 1.00, 19.75 &     1THB3539 \\
# \text{HETATM}& 3396&  O1 & HEM& C&   1    &   5.558, -17.474,  13.668, & 0.40, 21.60 &     1THB3550 \\
# \text{HETATM}& 3397&  O2 & HEM& C&   1    &   4.756, -17.191,  14.691, & 1.00, 32.41 &     1THB3551 \\
# \hline\end{array}$$
# 
# **Strategy:** The atom positions should be made into vectors. The equation of the plane is given by equation 44 and the perpendicular distance to it by equation 45.
# 
# ![Drawing](vectors-fig37.png)
# 
# Figure 37. Part of .pdb 1THB showing the porphyrin, O$_2$, and histidine molecules attached to the Fe at the centre of the porphyrin's ring. The distortion of the porphyrin is clear, showing that the Fe is not in the plane of the four N atoms. The best fit plane to the N atoms is described in chapter 13.6.
