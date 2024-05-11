#!/usr/bin/env python
# coding: utf-8

# # 6 Molecular Group Theory 

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
init_printing()                      # allows printing of SymPy results in typeset maths format
plt.rcParams.update({'font.size': 14})  # set font size for plots


# ## 6.1 Motivation and Concepts
# 
# When we learn to draw molecules and molecular orbitals their inherent symmetry becomes clear; benzene or perhaps tetrahedral methane spring to mind and possibly the geometry of sp$^2$ and sp$^3$ hybridizations. To characterize exactly what such symmetry means is the role of molecular group theory. This can also be used to determine the selection rules of spectroscopic transitions, to characterize the 'shapes' of normal mode vibrations and simplify molecular orbital calculations. In chemistry, the word 'symmetry', while retaining its colloquial meaning also has a technical meaning and this might appear to be rather abstract and divorced from other topics such as quantum mechanics and spectroscopy. Group theory's jargon does not help in learning the subject mainly because it appears to be so abstract. In fact, it is quite the opposite: it is intensely practical, and expresses a complicated set of rules and ideas in a few symbols; $D_{6h}$, for example, encapsulates all the many symmetry properties of benzene. The jargon we shall have to understand will lead us to be able to distinguish between 
# 
# $\qquad$*symmetry elements, symmetry operations, irreducible* and *reducible representations,*,
# 
# $\qquad$*characters, classes, basis sets, similarity transforms*, and *Mulliken labels*.
# 
# This section can only give a brief introduction to the subject to act as a basis for further
# study. There are many books on this topic, but Vincent (2001) follows a tutorial approach; Molloy (2004) has many molecular examples; Atkins & Friedman (1997) has a chapter giving a thorough mathematical approach; and Cotton (1990) and Bishop (1993) discuss the subject fully. 
# 
# The organization of this section is as follows: first, the geometrical properties of symmetry elements and operators are introduced. These are then used to identify a point group. It is then shown how two combined operations lead to the formation of an operator multiplication table, and how a symmetry operation can be represented by a matrix. Next, the pertinent properties of a mathematical group are described, and it is demonstrated how symmetry operations can form such a group. The character table is described next, and it is shown how the symmetry operations can be represented by a row of numbers rather than as a matrix or a symmetry label. Understanding how to use the information presented in the point group to characterize molecular vibrations and orbitals is our ending point.
# 
# ## 6.2 Essential jargon
# 
# Symmetry is defined as the relationship between parts of an object or between groups of objects in space. To identify the _symmetry elements_ inherent in a molecule, we look at those geometrical operations that can be performed that will make the molecule _indistinguishable_ from its initial state. Rotations and reflections are two of five such operations and are described further in Section 6.3. Group theory shows how symmetry properties can be represented by a set of numbers, called _characters_ when collected into a table called a _point group_ or *character table*, see Fig. 15 for an example. It is quite remarkable that in the context of group theory, a symmetry operation such as rotating a molecule can be represented by a number, often $0$ or $\pm 1$ although other values, including complex numbers, are possible depending on the point group. These numbers are the _characters_ of the point group and a _row of characters_ is called an _irreducible representation_ or 'irrep'. The character table uniquely defines the symmetry of a molecule and properties, for example, how different molecular orbitals behave and whether visible, infrared, and Raman transitions can occur, and, if they do, the orientation of the transition dipoles in the molecule. All the symmetry operations belonging to each point group are listed along the top row of a point group table; see Fig. 7.15. The characters are in the body of the table. In the next paragraphs, these concepts are expanded upon.
# 
# Most molecules contain very little symmetry, cholesterol or SOCl$_2$ are examples, but water, ammonia, chlorobenzene, ferrocene, and particularly benzene each have many symmetry elements. The more 'regular' the molecular structure is, the larger the number of symmetry elements it contains, which increases the number of symmetry operations that can be performed on these elements. To work out what point group a molecule belongs to and to determine its properties, the victim molecule is subjected to a given set of symmetry operations about each of the symmetry elements that may be present. You look at the molecule and then decide, by intuition, experience, or trial and error, which symmetry elements are present. The point group is then identified by the set of operations that leaves the molecule indistinguishable from its starting condition. The starting point is thus to know what the symmetry elements and operations are, and then to learn how to determine which ones are present in a molecule. Usually a simple three-dimensional model will help when doing this; it is sometimes difficult to 'see' the symmetries present from a sketch even if it is in perspective. It is a skill that improves with practice, and, as with riding a bicycle, if it is not done for a while, you can be a bit 'wobbly' to begin with.
# 
# ## 6.3 Symmetry operations and Symmetry Elements
# Symmetry operations act _via_ those symmetry elements that the molecule contains, which may be an axis, mirror plane or centre of inversion. The operation moves the molecule in space to a new, perhaps indistinguishable position, but the symmetry elements remain fixed. The words 'operation' and 'element' are often used interchangeably, but technically the operation can only occur about a symmetry element. For example, with a water molecule which has the same symmetry as ClO$_2$, Fig. 11, you will usually see a mirror plane (the element) and the reflection (the operation) at the same time. The operations and elements are linked because certain operations can only act on certain elements. If an operation does not leave the molecule indistinguishable, then it is not present.
# 
# ![Drawing](matrices-fig8.png)
# 
# Figure 8. Indistinguishable squares.
# _____
# 
# The effect of any valid symmetry operation is always to leave the molecule in an _indistinguishable_ state$^*$ not necessarily an _identical_ state. To understand this important distinction further, consider the square shown in Fig. 8, where the label is used only to identify one corner. Rotating the square by $90^\text{o}$ (in 'math speak' operating with a $+\pi/2$ rotation operator) makes the right-hand square indistinguishable from the left-hand one. Only after four similar rotation operations are the two squares *identical*.
# 
# A study of group theory shows that there are only five types of operators that could leave a molecule in an indistinguishable state. However, before operating on a molecule, a principal axis must first be chosen. All the symmetry operations are referenced with respect to the _principal axis_, which is the axis of highest rotational symmetry in the molecule. If two or more axes are the same, then one victim must be chosen. Figure 9 shows some examples. It is normal also to choose the principal axis as the z-axis and then to define x- and y-axes at right angles in the usual way. If the molecule is planar, as is naphthalene, the z-axis is usually chosen to project out of the plane.
# 
# ![Drawing](matrices-fig9.png)
# 
# Figure 9 Principal axes (dashed) and the rotation operators about this symmetry element (axis) that make the molecules indistinguishable.
# ______
# 
# The operators are:
# 
# ### **(i) Identity**
# 
# The identity id labelled $E$. No atoms change position with this operation and all molecules possess the Identity.
# 
# ### **(ii) Rotation**
# Rotation about an axis, labelled $C_n$. This symmetry element is an axis that often coincides with an x-, y-, or z-axis, but might be in any other direction depending upon the molecule. Rotation by $180^\text{o}$ is labelled $C_2$, by $120^\text{o}\; C_3$, etc. The subscript defines how many times the operation that makes the molecule identical has to occur. A molecule may have rotation about more than one axis. The label $C$ is a shorthand for cyclic.
# 
# ### **(iii) Reflection**
# Reflection in a mirror plane$^{**}\; \sigma$ The symmetry element is a plane. There are three types of mirror planes: vertical if the mirror edge runs along the principal axis, horizontal if this axis is at $90^\text{o}$ to the mirror, and dihedral if the mirror divides two axes; see figure 10. There may be more than one of each type of mirror plane in a molecule. For example, water or ClO$_2$ has two vertical planes labelled $\sigma$ and $\sigma'$ or $\sigma_V$ and $\sigma_V'$ (or $\sigma(y,z)$ and $\sigma(x,z)$), see Figure 11. A horizontal mirror plane is labelled $\sigma_h$, a dihedral plane $\sigma_d$. One or more superscript dashes ' are added if more than one of a type of mirror plane is present. In cases where a mirror plane falls on two axes this may alter- natively be labelled $\sigma(x, z)$ etc.
# 
# 
# ### **(iv) Inversion**
# Inversion through a centre $i$. The element is the centre of inversion the operation always changes coordinates from $(x, y, z) \to (-x, -y, -z)$ and _vice versa_. See figure 12. Only one inversion centre is possible.
# 
# ### **(v) Rotation - reflection**
# A combined rotation - reflection operation, $S_n$, also called an _improper rotation_. There may be more than one of these. The axis subscript $n$ is defined as in (ii). The operation is rotation followed by reflection in a plane perpendicular to the rotation, $\sigma_nC_n$. Figure 14 shows an example of an $S_4$ operation ($S$ is from the word *Sphenoidisch*).
# 
# $^*$ An operation that 'takes it into itself' is used in some texts to mean indistinguishable.
# $^{**}$ The Greek letter sigma, $\sigma$ , is used to represent the initial letter of the word _Spiegelung_ or reflection.
# 
# ![Drawing](matrices-fig10.png)
# 
# Figure 10. The three different types of mirror plane.
# ______

# Three examples of rotation operations are shown in figure 9. The ammonia molecule when rotated about the principal axis by $120^\text{o}$ becomes indistinguishable. It is also indistinguishable if rotated by twice this amount, but if rotated three times it is more than indistinguishable; it is identical to the starting state of the molecule. The molecule is also indistinguishable if rotated by $-120^\text{o}$ or $-240^\text{o}$. The label $C_3$ represents one rotation operation as the molecule is moved by $120^\text{o} = 360^\text{o}/3$ of a turn; two rotations are labelled $C_3^2$ and three $C_3 \equiv E$. If you are unclear about this, label the H atoms 1, 2, 3 and draw out the pictures or make a model; it really does help to do this yourself. The water molecule has only to be rotated by $\pm 180^\text{o}$ to become indistinguishable, which is a $C_2$ operation. The fluorinated acetylene can have any angle of rotation to become indistinguishable and this is labelled $C_\infty$.
# 
# Now consider symmetry or mirror planes. There are three types, as shown in figure 10, defined relative to the principal and other axes. The molecule SOCl$_2$, figure 11, has one chlorine atom that is in front of the plane and one that is behind it and the SO atoms are in the plane. This molecule has only one vertical mirror plane and no other symmetry operations are valid, other than the identity. Only the mirror plane makes the molecule indistinguishable. The symmetry operations for chloride dioxide, which has a $C_{2V}$ point group label, as do water, SO$_2$, and pyridine among many others, are also shown in figure 11, except the identity, which is always present but changes nothing. The first step is to define the principal axis and as ClO$_2$ is bent into a V shape, the axis has the direction shown in the figure. (The molecule could be also drawn the other way up.) There are only four types of operations in the $C_{2V}$ point group, which are (i) the identity $E$, (all molecules have this); (ii) rotation by $180^\text{o}$ around the principal axis labelled $C_2$; and (iii) there are two vertical mirror planes $\sigma_V$ and $\sigma_V'$ . The superscript dash is only used only to distinguish one axis from the other. If a set of $x, y, z$-axes are drawn on the molecule with $z$ as the principal axis, then the mirror planes could alternatively be labelled $\sigma(xz)$ and $\sigma(yz)$. There is a possible ambiguity because some authors place the x-axis in the plane of the molecule and some place the y-axis here. You need to check this when looking at different point group tables. The mirror planes are vertical because their edge runs along the principal axis. If the axis passed perpendicularly through the middle of a mirror plane, this would be a horizontal mirror plane, figure 10.
# 
# To summarize: in the $C_{2V}$ point group, the only symmetry operations present are the identity, whose symbol is $E$, and is present whatever the point group, one rotation $C_2$, and two mirror planes, $\sigma_V$ and $\sigma_V'$. A $C_3$ or $C_4$ rotation, which would be rotation by $120$ or $90^\text{o}$  respectively, cannot make the molecule indistinguishable from its starting position and neither can an inversion or any type of improper rotation operation $S_n$, and thus they are not present.
# 
# ![Drawing](matrices-fig11.png)
# 
# Figure 11. Left: The one mirror plane in a $C_s$ point group molecule. Right: Symmetry operations in a $C_{2V}$ molecule. The $C_2$ operation is rotation by $180^\text{o}, \sigma$ and $\sigma '$ operations are reflections in mirror planes as shown. Labels a and b help when performing operations; the atoms are identical.
# ______
# The naphthalene molecule, figure 12, has many more symmetry elements than the centre of inversion. These are indicated in figure 13. The principal axis could be any of the three $C_2$ axes because they are all equal; the coordinates drawn show that the out of plane direction is chosen to be $z$ and so this will be chosen as the principal axis. You could choose another orientation of axes if you wanted to, but however the axes are chosen the molecule still has three $C_2$ axes, three mirror planes running along these axes, a centre of inversion and a mirror plane in the plane of the molecule, labelled $\sigma_h$ because it is perpendicular to the principal axis. The other element is the identity $E$. With this information the point group can be identified. The next section shows how this may be done.
# 
# ![Drawing](matrices-fig12.png)
# 
# Figure 12. Each of these molecules has an inversion centre. This is shown with a red dot and is at the S atom in SF$_6$ so not visible. Every atom can be moved through the inversion centre to an equivalent point on the opposite side of the molecule leaving it indistinguishable. The operation always changes coordinates from $(x, y, z) \to (-x, -y, -z)$ and _vice versa_.
# 
# ![Drawing](matrices-fig13.png)
# 
# Figure 13 Symmetry elements in naphthalene. The principal axis is out of plane along $z$.
# 
# ![Drawing](matrices-fig14.png)
# 
# Figure 14. The $S_4$ rotation-reflection operation applied to the tetrahedral molecule CCl$_4$. The atoms are labelled only to allow the operations to be followed they are otherwise identical.
# ______
# 
# ## 6.4 A strategy to identify molecular Point Groups
# 
# The web site www.molecule-viewer.com has $\approx 500$ examples of molecules in all commonly used point groups. The 3D images are rotatable and planes and axes can be added as hints to test your ability to identify the point group.
# 
# When trying to assign a point group, first see if the molecule is a 'special case', that is tetrahedral, octahedral, icosahedral (football shaped), or is 'cylindrical/linear' such as O$_2$, HCl or FCCF, FCCH, etc. and identify it on this basis alone; see Section 6.5 for examples. You can always check later to see if you have guessed correctly by comparing with the point group (character) table. 
# 
# Next, look for any obvious overall rotational symmetry; for example, benzene clearly has sixfold symmetry and pyridine twofold, and this often indicates the principal axis direction and the symmetry label for the highest rotation operator. If a centre of inversion is present, this severely limits the choices of point groups. Particular axes or mirror planes can now be hunted down. Usually these will be enough to restrict your search to one or two point groups. At this point, you will have a list of some rotations and mirror planes and perhaps an inversion. The next step could be to look at tables of point groups and to see how best to match them with your findings so far. The table you choose may suggest the presence of some feature that you have missed.
# 
# If the highest (principal) axis is twofold symmetric, the point group will be restricted to those with a $2$ in their subscript, $C_{2V}, D_{2h}$, etc. The groups with $C_2, C_3$, etc. labels are single axis groups, meaning that _only one rotation axis_ is present. Molecules with more than one rotation axis are labelled $D$. 
# 
# Three groups have low symmetry and are $C_1$, fully asymmetric, $C_s$, e.g. SOCl$_2$, with only one mirror plane, and $C_i$, which only has  a centre of inversion. At the other end of the scale, the cubic groups, tetrahedral, octahedral, and icosahedral molecules, have very many symmetry elements and are easily identified.
# 
# Better than guessing is to use a systematic way of finding the point group and a 'route map' algorithm is shown in figure 17. The route map follows roughly the same method as just outlined. Sometimes a shortcut can be made by using the point group labels because they contain a shorthand version of the symmetry operations. The $C$ and $D$ groups are very common and the meaning of the labels is shown in figure 16. Assigning point groups is a strange skill; you can become very proficient quite quickly, but lose this skill equally quickly if it is not practiced. However, with a little revision, this soon returns.
# 
# ## Summary
# 
# **(a)**$\quad$ Check for special cases; diatomic, octahedral and tetrahedral molecules.
# 
# **(b)**$\quad$ Look for rotation axes; highest order axis is the principal axis, this gives the first subscript, $n$.
# 
# **(c)**$\quad$ Determine orientation of any $C_2$ axes perpendicular to principal axis. If none is perpendicular
# then letter is $C$ (or $S$) else $D$.
# 
# **(d)**$\quad$ Determine the orientation of any mirror planes relative to principal axis; this gives subscripts
# $V,h,d$.
# 
# **(e)**$\quad$ Identify all remaining symmetry elements/operation and check with point group tables.
#  
# ![Drawing](matrices-fig15.png)
# 
# Figure 15. Navigating the point group character table. The symmetry species $A_1,A_2,E$ are also called Mulliken labels. (Note that in some texts the notation may be different $C_3^+ \to C_3$ and $C_3^-\to C_3^2$). There are three classes of operations in this point group.
# _____
# 
# ![Drawing](matrices-fig16a.png)
# ![Drawing](matrices-fig16b.png)
# 
# Figure 16. notation for $C$ and $D$ point groups
# ______
# 
# ## 6.5 Examples of Point Groups
# 
# Most of the H atoms are not included in many structures. To make the figures clear the scale of the molecule is not the same in each figure. 
# 
# ![Drawing](matrices-pointgroup1.png)
# ![Drawing](matrices-pointgroup2.png)
# ![Drawing](matrices-pointgroup3.png)
# ![Drawing](matrices-pointgroup4.png)
# ![Drawing](matrices-pointgroup5.png)

# ![Drawing](matrices-fig17.png)
# 
# figure 17. 'Road-map' to assign point groups.
# ___________
# 
# ## 6.6 Products of Operators
# 
# To determine what a group multiplication table is, it is necessary to examine the products of two or more symmetry operations. These are then compared with the properties of a mathematical group, and this set of operations may then be associated with a particular group. How this is done is explained in the next few sections.
# 
# Using the symmetry operations shown in figures 11-16 and figure 19, or the matrix representation of the next section, the group multiplication table will be constructed for the $C_{2V}$ point group. The operations are $E, C_2, \sigma_V, \sigma_V'$ and a molecule of this point group is shown in figures 11 and 19. The product table is made by multiplying every operation by every other one, both ways round - for example, $\sigma_V C_2$ and $C_2 \sigma_V$ - and then determining if the product is also one of the operations, which it must be if $\sigma_V$ and $C_2$ both belong to the group. The rules of this 'game' are given in Section 6.7.
# 
# 
# $$\displaystyle \begin{array}{l|lll}
# C_{2V} & E & C_2 & \sigma_V & \sigma_V' \\
# \hline
# E & E & C_2 &\sigma_V & \sigma_V'\\
# C_2 & C_2 &E&\sigma_V'& \sigma_V\\
# \sigma_V &\sigma_V &\sigma_V' & E & C_2\\
# \sigma_V' & \sigma_V' & \sigma_V & C_2 & E\\
# \hline
# \end{array}$$
# 
# Figure 18. Multiplication table for $C_{2V}$. Notice that the matrix of operations is circulant, each row is rotated by one position relative to the one above it the matrix is also diagonal. (The multiplication is $A B$ with $A$ as the side column $ B$ the top row.)
# ____________
# 
# Figure 19 shows one $C_2$ rotation; two of them will make the molecule identical or $C_2C_2 = C_2 = E$ and similarly using figure 11 shows that reflecting in either of the mirror planes twice each in succession, also produces an identical molecule; therefore, $\sigma^2 = E$ and $\sigma '^{2} = E$. Multiplying an operator by itself produces the diagonal terms in figure 18. The identity multiplied by itself is still the identity; similarly, the identity multiplied by any other operator leaves the operator unchanged so this produces the left-hand column and top row. What remains are the other off-diagonal terms, such as $\sigma_VC_2$ and $C_2\sigma_V$ and these are left for you to confirm. The result for this point group is a symmetrical product table meaning, that in the $C_{2V}$ point group all the operators commute with one another, which is called an _Abelian_ group. This is not always true; for example, see the $C_{3V}$ table produced in Q 17. Being able to form a new species by multiplying two others is very useful when determining, say, the effect of a reflection in, for example, $C_3$  or $C_5$  where angles are not $90^\text{o}$ and when one reflection and a rotation is already worked out. 
# 
# The $C_{2V}$ table shows that the operations form a group, because they conform to the rules of a group as described in Section 6.7 and each row in the body of the table is therefore a representation of the group. It is not a very convenient representation however, because these can hardly be distinguished from the operators. Another representation can be imagined where all the entries in the table would be 1. This would follow the rules for forming a group but would be useless, as one operation could not be distinguished from another. The clever part was the development of a representation of each point group, such as $C_{2V}$ or $D_{2h}$, in a meaningful and practically useful way, and to this end, matrices can be used.
# 
# ![Drawing](matrices-fig19.png)
# 
# Fig. 19. Rotation in $C_{2V}$. The positive $x$ direction projects out of the image.
# ___________
# 
# ## 6.7 Pertinent properties of a Mathematical Group
# 
# The word 'group' in the context of molecular point groups has a precise mathematical definition. The group consists of a set of members that are the symmetry operations and follow four rules.
# 
# **(a)**$\quad$ There must be an identity operator that commutes with all others in the group and leaves them unchanged. The identity is always labelled $E$.
# 
# **(b)**$\quad$ The product of two operators $A$ and $B$ is also an operator and member of the group, i.e. $AB$ belongs to the group as does $AA$ and $BB$.
# 
# **(c)**$\quad$ The operators follow the associative product rule $(AB)C=A(BC)$.
# 
# **(d)**$\quad$ Every operator $A$ has an inverse $A^{-1}$ that is also an operator and member of the group. 
# 
# $\qquad$ Any operator $A$ that operates on its inverse produces the identity $AA^{-1} = E$. 
# 
# $\qquad$ Therefore, by this rule, the operator $A^{-1}$ must be a member of the group.
# 
# The members of the $C_{2V}$ group are $E, C_2, \sigma_V, \sigma_V'$. The multiplication table, figure 18, shows that rule (ii) is followed, because each of the entries in the table is a member of the group. In symmetry operations on molecules, it is common for the inverse and the operator to be identical, i.e. an operator can be its own inverse; for example, $C_2C_2 = C_2 = E$ meaning that $C_2^{-1} = C_2$. This shows that rule (iv) is followed.
# 
# ## 6.8 Symmetry Operations as matrices
# Although we can perform symmetry operations in a geometrical sense, as was done to produce the $C_{2V}$ product table, these can be rather awkward to use. It turns out that a symmetry operation can be performed as a matrix multiplication using as a basis either a molecule's atoms, or its orbitals or bonds. The _trace_ of the matrix, the sum of its diagonal terms, will form a _representation of each operation_ and so form a representation of the group. The matrix used for each symmetry operation used must be a unitary matrix, $| \pmb{M} | = 1$, because bond angles and lengths must be unchanged to maintain a molecule's symmetry. As an example, the symmetry properties of chlorine dioxide ClO$_2$ will be examined, using as a basis set the three atoms and with the oxygen atoms labelled $a$ and $b$ for convenience, this _basis_ is ($Cl, O_a, O_b$), figure 19. The oxygen atoms are labelled only to keep track of them; they are otherwise identical. The identity operation is represented by the matrix equation where the vector containing the atoms does not change,
# 
# $$\displaystyle [E]\begin{bmatrix} Cl \\O_a\\ O_b \end{bmatrix}=\begin{bmatrix} Cl \\O_a\\ O_b \end{bmatrix}$$ 
# 
# and, by the rules of matrix multiplication, the matrix$[E]$ must be a $3 \times 3$ matrix that keeps the left- and right-hand column matrices equal. Note that the two column matrices are identical; $E$ is, after all, the identity! Because the identity matrix leaves the column vector unchanged, it must be the unit matrix $\pmb{1}$,
# 
# $$\displaystyle \begin{bmatrix} 1&0&0\\ 0&1&0\\0&0&1 \end{bmatrix} \begin{bmatrix} Cl\\O_a\\O_b \end{bmatrix}=\begin{bmatrix} Cl\\O_a\\O_b \end{bmatrix}$$
# 
# A $C_2$ operation or $180^\text{o}$ rotation exchanges the oxygen atom positions $a$ and $b$ but leaves the Cl atom unchanged;
# 
# $$\displaystyle C_2\; \text{operation causes:}\quad Cl \to Cl,\quad O_a \to O_b, \quad O_b \to O_a $$
# 
# The matrix equation must therefore have the form
# 
# $$\displaystyle [C_2]\begin{bmatrix} Cl \\O_a\\ O_b \end{bmatrix}=\begin{bmatrix} Cl \\O_b\\ O_a \end{bmatrix}$$ 
# 
# To find out what matrix is needed, a little trial and error is required. In the matrix equation, $\displaystyle \begin{bmatrix} 0 & 1 \\ 1 & 0  \end{bmatrix} \begin{bmatrix} A \\ B \end{bmatrix} = \begin{bmatrix} B \\ A \end{bmatrix} $ swaps the positions of $A$ and $B$. Thus we can write the $C_2$ matrix operation as
# 
# $$\displaystyle \begin{bmatrix} 1&0&0\\ 0&0&1\\0&1&0\end{bmatrix}\begin{bmatrix} Cl \\O_a\\ O_b \end{bmatrix}=\begin{bmatrix} Cl \\O_b\\ O_a \end{bmatrix}$$ 
# 
# Now clearly this procedure is a complex business in a molecule even with only three atoms, but, in fact, you effectively do this operation in your head when you look at a picture of the molecule and reflect or rotate it. Suppose that the molecule is again rotated by $180^\text{0}$, then it must return to where it started, i.e. it must be identical. This would mean that the following equation must be true, and, although we have already seen this is the case, it can be proved with the multiplication,
# 
# $$\displaystyle [C_2][C_2]\begin{bmatrix} Cl \\O_a\\ O_b \end{bmatrix}=[C_2]\begin{bmatrix} Cl \\O_a\\ O_a \end{bmatrix}=\begin{bmatrix} Cl \\O_a\\ O_b \end{bmatrix}$$
# 
# Because the $C_2$ matrix swaps the positions of the a and b atoms, this equation will work because the first $C_2$ swaps them and the second swaps them back. Alternatively, this double operation could be written as
# 
# $$\displaystyle [C_2]^2\begin{bmatrix} Cl \\O_a\\ O_b \end{bmatrix}=\begin{bmatrix} Cl \\O_a\\ O_b \end{bmatrix}\equiv [E]\begin{bmatrix} Cl \\O_a\\ O_b \end{bmatrix}$$
# 
# where $C_2^2\equiv [E]$. Notice that in this last calculation, the associative product rule of matrices, and of a group (rule (iii)), was used because $C_2C_2$ was worked out first. To show that this last result is true, work out the direct multiplication,
# 
# $$\displaystyle \begin{bmatrix} 1&0&0\\ 0&0&1\\0&1&0\end{bmatrix}\displaystyle \begin{bmatrix} 1&0&0\\ 0&0&1\\0&1&0\end{bmatrix}=
# \displaystyle \begin{bmatrix}
# 1\times 1+0+0 & 0 & 0 \\0\times 1+0+0 & 0+0+1\times 1 &0\\0\times 1+1\times 0+0 & 0&  1
# \end{bmatrix}=\displaystyle \begin{bmatrix} 1&0&0\\ 0&1&0\\0&0&1\end{bmatrix}$$
# 
# Repeating the calculation for the reflections produces the matrices
# 
# $$\displaystyle [\sigma(x,z)]=\begin{bmatrix} 1&0&0\\ 0&0&1\\0&1&0\end{bmatrix}\displaystyle \quad [\sigma(y,z)]=\begin{bmatrix} 1&0&0\\ 0&1&0\\0&0&1\end{bmatrix}$$
# 
# which are the same as other matrices in this group. If we perform operations on any pair of matrices, say rotation and reflection, a group multiplication table can be built. With this, and by applying methods from group theory, a character table that describes all the symmetry properties of a molecule can be produced. Instead of using the atoms as a basis, the same matrices would be produced by three unit length vectors, one along each bond and one along the principal axis, and each originating at the Cl atom.
# 
# The matrices just calculated represent the operations in the $C_{2V}$ point group with three atoms or unit vectors along the principal axis and bonds and can be collected together as
# 
# $$\displaystyle \qquad\qquad\begin{array}{cccc}
# C_{2V} & E & C_2 & \sigma (x,z) & \sigma '(y,z) \\
# \hline 
# \Gamma_R &
# \begin{bmatrix} 1&0&0\\ 0&1&0\\0&0&1\end{bmatrix}&
# \begin{bmatrix} 1&0&0\\ 0&0&1\\0&1&0\end{bmatrix}&
# \begin{bmatrix} 1&0&0\\ 0&0&1\\0&1&0\end{bmatrix}&
# \begin{bmatrix} 1&0&0\\ 0&1&0\\0&0&1\end{bmatrix}\\
# \end{array} \qquad\qquad\qquad\text{(9.0)}$$
# 
# which is one form of a reducible representation $\Gamma_R$, and is reducible because it is not in its simplest form since some matrices have off-diagonal terms, therefore the characters in the point group cannot be determined directly. The trace of each matrix produces the table;
# 
# $$\displaystyle 
# \qquad \text{atom basis set}\\
# \begin{array}{c|ccc}
# C_{2V} & E & C_2 & \sigma (x,z) & \sigma '(y,z)\\
# \hline 
# \Gamma_R &
# 3 & 1 & 1 & 3\\
# \hline \end{array}$$
# 
# If any other basis set covering the same 'space' were used then the trace of the matrices giving rise to the reducible representation would be the same. If a basis in a different 'space' were used, such as $x, y, z$ unit vectors, then a different reducible representation would be produced, as described in the next section. Two similar spaces could be an atoms' p orbitals in the form $p_x, p_y$, and $p_z$ or as $p_0, p_{-1}$, and $p_{+1}$ where the numbers represent the $m$ quantum numbers. These two forms of orbitals can be transformed into one another; $p_z = p_0; p_x = (p_{+1} + p_{-1})/ 2$, and $p_y = -i(p_{+1} - p_{-1})/2$ hence their 'space' is the same.
# 
# The process of working out the effect of each operator is not complicated, but can prove tedious; however, it is important because if all the $C_{2V}$ operations can be identified with just one molecule of this point group the same rules must apply to _all molecules of the same point group_ no matter how many atoms it has. As these have all been worked out; all we usually need to do is to identify the point group. Figure 20 shows a few molecules belonging to the $C_{2V}$ point group.
# 
# ![Drawing](matrices-fig20a.png)
# ![Drawing](matrices-fig20b.png)
# ![Drawing](matrices-fig20c.png)
# 
# Figure 20. Some molecules belonging to the $C_{2V}$ point group. For clarity some H atoms are ignored. The structures are only approximately to scale relative to one another. Where the $C_2$ axis is shown the orange sphere indicates the 'centre of gravity' of the molecule, i.e. the 'point' in the point group.
# ______

# ## 6.9 Representations based on matrices
# 
# A basis set can consist of unit vectors along the x-, y- and z-axes.  The translational vectors can be imagined along the axes shown in Figure 19 or 21. The matrices we now workout are quite general and do not belong to any particular point group. 
# 
# ### **(i) Reflections**
# 
# A reflection of each $x,y$ or $z$ vector in turn in the $z-y$ plane will leave the $z$ and $y$ vectors unchanged (fig 21), but invert the $x$, therefore the matrix equation is
# 
# $$\displaystyle \sigma(y,z)= \begin{bmatrix} -1&0&0\\ 0&1&0\\0&0&1\end{bmatrix},\qquad \sigma(y,z)\begin{bmatrix} x\\ y\\z\end{bmatrix}=\begin{bmatrix} -x\\ y\\z\end{bmatrix} \qquad\qquad\qquad\text{(9.1)}$$
# 
# where the symbol $\sigma(y,z)$ represents the matrix operator. (Technically / pedantically it would be better to use a notation such as $O(\sigma(y,z))$ for the matrix operator but this is not really necessary as the context should make the usage clear.) A similar calculation for the $x-z$ plane changes only the $y$ coordinate giving
# 
# $$\displaystyle \sigma (x,z)= \begin{bmatrix} 1&0&0\\ 0&-1&0\\0&0&1\end{bmatrix},\qquad \sigma (x,z)\begin{bmatrix} x\\ y\\z\end{bmatrix}=\begin{bmatrix} x\\ -y\\z\end{bmatrix} \qquad\qquad\qquad\text{(9.2)}$$
# 
# and $x-y$ plane
# 
# $$\displaystyle \sigma (x,y)= \begin{bmatrix} 1&0&0\\ 0&1&0\\0&0&-1\end{bmatrix},\qquad  \sigma (x,y)\begin{bmatrix} x\\ y\\z\end{bmatrix}=\begin{bmatrix} x\\ y\\-z\end{bmatrix} \qquad\qquad\qquad\text{(9.3)}$$
# 
# ### **(ii) Rotations**
# 
# A rotation about the principal axis at angle $\theta=2\pi/n$ is labelled $C_2$ when $n=2$ and so forth. The general rotation matrix for an angle $\theta$ is
# 
# $$\displaystyle C_n= \begin{bmatrix} \cos(\theta)&\sin(\theta)&0 \\ -\sin(\theta) & \cos(\theta) & 0 \\ 0&0&1 \end{bmatrix},\qquad C_n \begin{bmatrix} x\\ y\\z\end{bmatrix}=\begin{bmatrix} x\cos(\theta)+y\sin(\theta)\\ -x\sin(\theta)+y\cos(\theta)\\z\end{bmatrix} \qquad\qquad\qquad \text{(9.4)}$$
# 
# Note that some authors may use an alternative matrix, which is the inverse of this, and has the $-\sin()$ and $+\sin()$ swapped. The only difference is whether you consider the axes or molecule to be rotated. 
# 
# ### **(iii) Rotation- inversion (improper rotation)**
# 
# The rotation-inversion matrix $S_n$ can be obtained as the product of a rotation $C_n$ and a reflection $\sigma_{xy}$ in the plane perpendicular to the principle axis. The rotation-inversion matrix is 
#  
# $$\displaystyle S_n= \begin{bmatrix} \cos(\theta)&\sin(\theta)&0 \\ -\sin(\theta) & \cos(\theta) & 0 \\ 0&0&-1 \end{bmatrix},\qquad S_n  \begin{bmatrix}1&0&0\\0&1&0\\0&0&-1 \end{bmatrix}=\begin{bmatrix} \cos(\theta)&\sin(\theta)&0 \\ -\sin(\theta) & \cos(\theta) & 0 \\ 0&0&-1 \end{bmatrix}\qquad\qquad\qquad\text{(9.5)}$$
# 
# ### **(iv) Inversion and Identity**
# 
# The two remaining operations, inversion $i$ and the identity are simple and can be written down directly and are
# 
# $$\displaystyle i= \begin{bmatrix} -1&0&0\\ 0&-1&0\\0&0&-1\end{bmatrix}, \qquad E=\begin{bmatrix} 1&0&0\\ 0&1&0\\0&0&1\end{bmatrix} \qquad\qquad\qquad\text{(9.6)}$$
# 
# Finally, it is often very easy to multiply matrices to find another symmetry operation. This can be done because their product must be a member of the group. A useful application is with the $C_{3V}$ point group where one mirror plane can be found from another by rotation by $120^\text{o}$, i.e. multiply matrices $C_3\cdot\sigma$. 
# 
# ![Drawing](matrices-fig21.png)
# 
# Figure 21. Left. Reflection of the $x$ unit vector in the $y-z$ plane swaps $x$ coordinates only. Right. The effect of a mirror on a  circular vector $R_z$ about the $z$-axis is to reverse the direction of rotation.
# ____________
# 
# ### 6.9.1 Matrices generating the $C_{2V}$ point group
# 
# The $C_{2V}$ point group has operations $E, C_2,\sigma_V\equiv\sigma(x,z),\sigma_V'\equiv\sigma(y,x)$ and the matrices for these operations with a basis set of three orthogonal unit vectors along $x,y,z$ arranged as a table is, 
# 
# $$\displaystyle 
# \begin{array}{ccccc}
# C_{2V} & E & C_2 & \sigma (x,z) & \sigma '(y,z) \\
# \hline 
# &
# \begin{matrix}x\\y\\z\end{matrix} \begin{bmatrix}\bbox[5px,border:1px solid red] 1&0&0\\ 0&\bbox[5px,border:1px solid red]1&0\\0&0&\bbox[5px,border:1px solid red]1\end{bmatrix}&
# \begin{bmatrix} -1&0&0\\ 0&-1&0\\0&0&1\end{bmatrix}&
# \begin{bmatrix} 1&0&0\\ 0&-1&0\\0&0&1\end{bmatrix}&
# \begin{bmatrix} -1&0&0\\ 0&1&0\\0&0&1\end{bmatrix}\\
# \end{array} \qquad\qquad\qquad\text{(9.7)}$$
# 
# A _representation_ of the matrix is used to form the point group, rather than using the whole matrix itself, and this is the trace (sum of diagonals) of the matrix. However, we need to be careful here because the matrices above are diagonal and each consists of a block of $1\times 1$ matrices, and so the trace of each matrix is $\pm 1$. (The block form is outlined only for the first matrix). Thus the point group so far is the list of characters placed in rows, starting with the totally symmetric representation which is the row of $+1$ and in this case is that of the $z$ vector. This is followed by rows for $x$ and $y$,
# 
# $$\displaystyle 
# \begin{array}{c|cccc|cc}
# C_{2V} & E & C_2 & \sigma (x,z) & \sigma '(y,z) \\
# \hline
# \Gamma_1& 1 & 1& 1& 1 &z\\
# \Gamma_2& 1 & -1 & 1 & -1 & x\\
# \Gamma_3& 1 & -1 & -1 & 1 & y\\
# \hline
# \end{array} \qquad\qquad\qquad\text{(9.8)}$$
# 
# However, this is not complete as the matrices are $3\times 3$, giving in this case just three irreducible representations, but there are four classes or types of operation, so something is missing, i.e. the number of irreducible representations is not equal to the number classes of operations, see fig 15.  
# 
# We could use a different basis set, for example, either by using rotational vectors or with a basis set of vectors from the orthogonal set of the angular parts of the five d-atomic orbitals. However, because the point group is a mathematical group we can use the properties of a group to work out the missing row of characters. To do this we need to use a result from the Great Orthogonality Theorem which is that the characters in each row of a point group are orthogonal to one another (Bishop 1993, chapter 7). The equation looks rather intimidating but is very simple to use. Any two rows $a$ and $b$ in the point group, and $a$ and $b$ could be the same row, must satisfy
# 
# $$\displaystyle \sum_i g_i\chi^a_i\chi^{b*}_i=h\delta_{a,b}\qquad\text{Orthogonality of rows}\qquad\qquad\qquad\text{(9.9)}$$
# 
# where $i$ ranges over all the columns of characters which is the number of different types symmetry operations (classes) and $g_i$ is the number in each class, for example in $C_{3V}$, $g_i$ is $2$ for operation $C_3$. $\chi$ is the value of the character itself, which is a complex number in some point groups and $*$ indicates the complex conjugate, which has no effect unless the character is a complex number. If the irreducible representations are different then the Kronecker delta function is $\delta_{a,b} = 0$ if $ a\ne b$ otherwise it is one and finally $h$ is the total number of operations $h=\sum_i g_i$, and for $C_{3V}$ this is $1+2+3=6$. 
# 
# A second formula is usually needed and is that the sum of the squares of the characters in the identity (E) add up to the total number of operations, or
# 
# $$\displaystyle \sum_i \chi_i^2(E) =h\qquad \text{sum of squared identity's characters = sum of operations}$$
# 
# A third formula shows that each pair of columns are orthogonal to one another and the sum square of the characters in any one column is equal to $h/g_i$. The columns are $c_a$ and $c_b$, the characters $\chi$ in the column position $i,j$
# 
# $$\displaystyle \sum_{c_a,c_b} \chi^{c_a}_i\chi^{c_b*}_j=\frac{h}{g_i}\delta_{ij},\qquad\text{Orthogonality of columns}$$
# 
# Returning to the missing row in the point group, this representation is labelled $\Gamma_4$ and calculating the column for $E$ becomes $1^2 +1^2 +1^2 +\chi^2_{\Gamma_{4E}}=4$ making $\chi_{\Gamma_{4E}}=1$.
# 
# Consider symmetry species $\Gamma_2$ and $\Gamma_3$ in the $C_{2V}$ table above then $a = 2,b = 3$, eqn. 9.9 becomes,
# 
# $$\displaystyle \sum_i g_i\chi^2_i\chi^{3*}_i= (1 \times 1 \times 1)+(1 \times -1 \times-1)+(1 \times 1 \times -1)+(1 \times-1 \times 1)=0$$
# 
# If $a=b\equiv\Gamma_2$ i.e. the same row then
# 
# $$\displaystyle \sum_i g_i\chi^a_i\chi^{b*}_i= (1 \times 1 \times 1)+(1 \times -1 \times-1)+(1 \times 1 \times 1)+(1 \times-1 \times -1)=4$$
# 
# To find the missing row we let the table be
# 
# $$\displaystyle 
# \begin{array}{c|cccc|cc}
# C_{2V} & E & C_2 & \sigma (x,z) & \sigma '(y,z) \\
# \hline
# \Gamma_1& 1 & 1& 1& 1 &z\\
# \Gamma_2& 1 & -1 & 1 & -1 & x\\
# \Gamma_3& 1 & -1 & -1 & 1 & y\\
# \Gamma_4& 1 & u & v & w & ?\\
# \hline
# \end{array} $$
# 
# and we need to find $u,v,w$. Three equations are needed, thus using $\Gamma_1\Gamma_4$ etc.,
# 
# $$\displaystyle \begin{align}\sum_i g_i\chi^1_i\chi^{4*}_i&= (1 \times 1 \times 1)+(1 \times 1 \times u)+(1 \times 1 \times v)+(1 \times 1 \times w)=1+u+v+w=0\\
# \sum_i g_i\chi^2_i\chi^{4*}_i&= (1 \times 1 \times 1)+(1 \times -1 \times u)+(1 \times 1 \times v)+(1 \times-1 \times w)=1-u+v-w=0\\
# \sum_i g_i\chi^3_i\chi^{4*}_i&= (1 \times 1 \times 1)+(1 \times -1 \times u)+(1 \times -1 \times v)+(1 \times 1 \times w)=1-u-v+w=0 \end{align} $$
# 
# and from these equations $v=w=-1$ and $u=1$. 
# 
# As a check $\Gamma_4\Gamma_4$ can be found which is 
# 
# $$\displaystyle \sum_i g_4\chi^4_i\chi^{4*}_i= (1 \times 1 \times 1)+(1 \times u \times u)+(1 \times v \times v)+(1 \times w \times w)=1+u^2+v^2+w^2=4$$
# 
# The full $C_{2V}$ character table is made by rearranging the rows placing the most symmetric at the top of the table and decreasing down the rows.  Mulliken notation symmetry species are added to the far left and functions to which these correspond to the right most columns. The missing row we have just evaluated is identified as the $A_2$ symmetry species.
# 
# $$\displaystyle \begin{array}{l|rrrr|l|ll}
# C_{2V} & E & C_2 & \sigma(xz) & \sigma'(yz) &  & & \\
# \hline
# A_1 & 1&1&1&1   & z  &x^2,y^2,z^2\\
# A_2 & 1&1&-1&-1 & R_z & xy\\
# B_1 & 1&-1&1&-1 & x, R_y & xz\\
# B_2 & 1&-1&-1&1 & y, R_x & yz\\
# \hline \end{array}\qquad\qquad\qquad\text{(9.10)}$$
# 

# The functions on the right of the table describe the symmetry species for rotations $R$ and for some other functions which will be described later on. Figure 21 (right) shows reflection of a rotational vector $R_z\to -R_z$. Rotation by $C_2,\;180^\text{o}$ leaves the $R_z$ vector unchanged, but reflection in the $z-y$ plane reverses it as does reflection in the $z-x$ plane. By the same reasoning the rotation about $x,y$ and $z$ can be found for each symmetry operation. The effects of a symmetry operation on a rotational vector are
# 
# $$\displaystyle C_2 \qquad \begin{bmatrix} -1&0&0\\ 0&-1&0\\0&0&1\end{bmatrix}\begin{bmatrix} R_x\\ R_y\\R_z\end{bmatrix}=\begin{bmatrix} -R_x\\ -R_y\\R_z\end{bmatrix} $$
# 
# $$\displaystyle \sigma(x,z)\qquad \begin{bmatrix} -1&0&0\\ 0&1&0\\0&0&-1\end{bmatrix}\begin{bmatrix} R_x\\ R_y\\R_z\end{bmatrix}=\begin{bmatrix} -R_x\\ R_y\\-R_z\end{bmatrix} $$
# 
# $$\displaystyle \sigma(y,z)\qquad \begin{bmatrix} 1&0&0\\ 0&-1&0\\0&0&-1\end{bmatrix}\begin{bmatrix} R_x\\ R_y\\R_z\end{bmatrix}=\begin{bmatrix} R_x\\ -R_y\\-R_z\end{bmatrix} $$
# 
# $$\displaystyle 
# \begin{array}{cccc}
# C_{2V} & E & C_2 & \sigma (x,z) & \sigma '(y,z) \\
# \hline 
# &
# \begin{matrix}R_x\\R_y\\R_z\end{matrix}\begin{bmatrix} \bbox[5px,border:1px solid red]1&0&0\\ 0&\bbox[5px,border:1px solid red]1&0\\0&0&\bbox[5px,border:1px solid red]1\end{bmatrix}&
# \begin{bmatrix} -1&0&0\\ 0&-1&0\\0&0&1\end{bmatrix}&
# \begin{bmatrix} -1&0&0\\ 0&1&0\\0&0&-1\end{bmatrix}&
# \begin{bmatrix} 1&0&0\\ 0&-1&0\\0&0&-1\end{bmatrix}\\
# \end{array} \qquad\qquad\qquad\text{(9.11)}$$
# 
# As the matrices are diagonal the irreps are found by looking at each row in the matrix across the symmetry operations, i.e. we consider each matrix as a block diagonal of three $1\times 1$ matrices. This is the same situation described above for $x,y,z$ vectors. The trace of the matrix is used as a representation but, of course, this is just the value of a $1\times 1$ matrix. The block diagonal is shown by the red squares in the first matrix only. 
# 
# $$\displaystyle \begin{array}{l|rrrr}
# C_{2V} & E & C_2 & \sigma(xz) & \sigma'(yz) \\
# \hline
#   & 1 & -1& -1 &1 &R_x\\   & 1 &-1 &1 &-1&R_y \\  & 1& 1 &-1 &-1 &R_z\\
# \end{array}$$
# 
# which is similar to obtained from the $xyz$ basis set. We could find the complete point group by the method based on orthogonality just as above, eqn. 9.9.
# 
# In this particular point group each of the translational vectors $x,y,z$ or rotational ones $R_{x,y,z}$ transforms into itself, e.g. $x\to x$ or are reversed in direction e.g. $x\to -x$. This is not always true as is shown shortly by the $C_{3V}$ point group where some symmetry operations mix vectors $x\to f(x,y)$, for example, where $f$ is a function of $x$ and $y$ and similarly for $y$ and $z$ vectors.
# 
# ## **Summary**
# 
# A *symmetry species* means that in a given point group the *characters* represent the effects the *symmetry operations* have on the molecule. The way linear, $x, y, z$ and product e.g. $xz, z^2\cdots$ and rotational operators $R_{x,y,z}$ transform are shown in the right-hand most columns. The entries in the body of the table are called the *characters*, and hence the name *character table*. The symbols (Mulliken labels,$A_1,E_{2g}$ etc.) in the left-hand column are the labels of the *irreducible representations* but usually these are called the symmetry species. The ordering is such that the totally symmetric representation is always the top row and lower symmetry below this. (There are rules for determining the order but these need never bother us). The *classes* are the columns in the table, often there are two operations with identical characters and these are preceded by a number, e.g. $2C_3$ in the $C_{3V}$ point group.
# 
# In all point groups, the top row has one of the symbols $A, A', A_1, A_{1g}'$ or $A_{1g}$ depending on the point group but $\Sigma_g$ for $C_{\infty V}$  and $\Sigma_g^+$ for $D_{\infty h}$ and is  always the totally symmetric representation; the lowest row is the 'least symmetric'. The symmetry species $B_2$ in the $C_{2V}$ point group has the properties that it is unchanged by the identity $E$; is changed by $180^\text{o}$ rotation about the $C_2$ axis and by reflection in the mirror plane $\sigma$, but unchanged by reflection in mirror plane $\sigma'$.
# 
# The diagram, figure 15, shows the various properties contained within the point group table, in this case for $C_{3V}$. The Mulliken label '$E$' in the bottom left-hand column in the table means that this irreducible representation is doubly degenerate. This should not be confused with $E$ the identity operation.
# 

# ### 6.9.2 Generating a Point Group. Matrices generating the $C_{3V}$ point group
# 
# Molecules of the $C_{3V}$ point group allow a more general description of how the characters in the table are obtained. Molecules such as $\mathrm{NH_3, NF_3,SOCl_3,CHCl_3, CClH_3 }$ belong to this point group. When a molecule has a three-fold or higher rotation axis this introduces some degeneracy, for example in CHCl$_3$ molecules, rotation about the x-axis or about the y-axis (see fig 21a) will lead to identical energy levels because the moment of inertial about these two axis is the same. Figure 21a shows the effect of rotation and reflection on a molecule belonging to the $C_{3V}$ point group. $x_1,y_1$ represent a unit vector situated on the C atom and along the $x,y$ axes and $x_2,y_2$ the transformed vector after the operations shown.
# 
# ![Drawing](matrices-fig21ad.png)
# 
# Figure 21a Shows how a vector ending at point $x_1,y_1$ moves with a $C_3$ rotation (left) and reflection in the $zx$ plane, $\sigma_V$, (right). The images show the geometry looking down the z-axis, i.e. in the $x-y$ plane of a molecule belonging to  the $C_{3V}$ point group such as CHCl$_3$. The arrows ending at  $x_1,y_1$ represent unit vectors (length $r=1$) and $x_2,y_2$ the end point of the transformed vector after $C_3$ rotation. In rotation the transformed vector $x_2,y_2$ is a mixture of the original unit vectors. The C atom is situated where the axis cross. 
# _________
# 
# ### **(i) Rotations**
# 
# The coordinates of $x_2,y_2$ in fig 21a after rotation are 
# 
# $$\displaystyle x_2=r\cos(\theta-\alpha),\qquad y_2=-r\sin(\theta-\alpha)$$
# 
# The trig identities 
# 
# $$\displaystyle \cos(\theta-\alpha)=\cos(\theta)\cos(\alpha)+\sin(\theta)\sin(\alpha)\\
# \sin(\theta-\alpha)=\sin(\theta)\cos(\alpha)-\cos(\theta)\sin(\alpha)$$
# 
# are now used and letting $r=1$, since we use unit vectors, produces
# 
# $$\displaystyle x_2=x_1\cos(\theta)+y_1\sin(\theta)\\y_2=-x_1\sin(\theta)+y_2\cos(\theta)$$
# 
# A $C_3$ ( or $C_3^+$) rotation of the $x_1y_1$ vector ( $\theta=2\pi/3=120^\text{o}$)  produces a new vector $x_2,y_2$,
# 
# $$\displaystyle C_3\qquad x_2= -\frac{x_1}{2}+\sqrt{3}\frac{y_1}{2},\qquad  y_2= -\sqrt{3}\frac{x_1}{2}-\frac{y_1}{2}$$
# 
# which are conveniently made into matrix form as
# 
# $$\displaystyle C_3\qquad\qquad \begin{bmatrix}x_2\\y_2\end{bmatrix}=\begin{bmatrix} -\dfrac{1}{2}&+\dfrac{\sqrt{3}}{2}\\ \dfrac{-\sqrt{3}}{2} &-\dfrac{1}{2}\end{bmatrix}\begin{bmatrix}x_1\\y_1\end{bmatrix}$$
# 
# The $C_3^2$ operation which may also be considered as $C_3^-$ i.e. left rotation by $120^\text{o}$ rather than clockwise by $240^\text{o}$, has $\theta=4\pi/3$ therefore
# 
# $$\displaystyle C_3^2\qquad\qquad \begin{bmatrix}x_2\\y_2\end{bmatrix}=\begin{bmatrix} -\dfrac{1}{2}& -\dfrac{\sqrt{3}}{2}\\ \dfrac{\sqrt{3}}{2} &-\dfrac{1}{2}\end{bmatrix}\begin{bmatrix}x_1\\y_1\end{bmatrix}$$
# 
# In both these cases the 2D part of the rotation matrix, eqn. 9d, could be used, for example with $\theta=2\pi/3$
# 
# $$\displaystyle C_3= \begin{bmatrix} \cos(\theta)&\sin(\theta) \\ -\sin(\theta) & \cos(\theta)  \end{bmatrix}= \begin{bmatrix} -1/2  & \sqrt{3}/2 \\ -\sqrt{3}/2  &  -1/2 \end{bmatrix}$$
# 
# ### **(ii) Reflections**
# 
# Reflection in the $xz$ plane just reverses a $y$ vector but leaves $x$ unchanged. The effect of reflection in  other mirror planes is found by multiplying the $\sigma_V$ matrix by a rotation matrix. This is a valid operation because both the rotation and reflection matrices belong to the same $C_{3V}$ point group.
# 
# $$\displaystyle \begin{align} \sigma_V(xz)\qquad\qquad \begin{bmatrix}x_2\\y_2\end{bmatrix}&=\begin{bmatrix} 1 &  0\\ 0 &-1\end{bmatrix} \begin{bmatrix}x_1\\y_1\end{bmatrix}\\ \sigma_V'\equiv C_3\sigma_V\qquad\qquad \begin{bmatrix}x_2\\y_2\end{bmatrix}&=\begin{bmatrix} -\dfrac{1}{2}&\dfrac{\sqrt{3}}{2}\\ -\dfrac{\sqrt{3}}{2} &-\dfrac{1}{2}\end{bmatrix}\begin{bmatrix} 1 & 0\\ 0 & -1 \end{bmatrix}\begin{bmatrix}x_1\\y_1\end{bmatrix}=\begin{bmatrix} -\dfrac{1}{2}& -\dfrac{\sqrt{3}}{2}\\ -\dfrac{\sqrt{3}}{2} &\dfrac{1}{2}\end{bmatrix}\begin{bmatrix}x_1\\y_1\end{bmatrix}\\ \sigma_V''\equiv C_3^2\sigma_V\qquad\qquad \begin{bmatrix}x_2\\y_2\end{bmatrix}&=\begin{bmatrix} -\dfrac{1}{2}& -\dfrac{\sqrt{3}}{2}\\ \dfrac{\sqrt{3}}{2} & -\dfrac{1}{2}\end{bmatrix}\begin{bmatrix} 1 & 0\\ 0 & -1 \end{bmatrix}\begin{bmatrix}x_1\\y_1\end{bmatrix}=\begin{bmatrix} -\dfrac{1}{2}& \dfrac{\sqrt{3}}{2}\\ \dfrac{\sqrt{3}}{2} &\dfrac{1}{2}\end{bmatrix}\begin{bmatrix}x_1\\y_1\end{bmatrix}\end{align}$$
# 
# The transformation matrix for translations ($x,y,z$) and rotations ($R_{x,y,z}$) for the $C_{3V}$ point group is
# 
# $$\displaystyle \scriptsize \begin{array}{c|c|c|c|c|c|c|c|}
# \hline
# C_{3V} &E &C_3&C_3^2 &\sigma_v&\sigma_V'&\sigma_v''& \text{rot/trans}\\
# \hline
# &1&1&1&1&1&1&z\\
# &1&1&1&-1&-1&-1&R_z\\
# &\begin{bmatrix} 1&0\\ 0 &1\end{bmatrix}&\begin{bmatrix} -\dfrac{1}{2}&\dfrac{\sqrt{3}}{2}\\ -\dfrac{\sqrt{3}}{2} &-\dfrac{1}{2}\end{bmatrix} &\begin{bmatrix} -\dfrac{1}{2}& -\dfrac{\sqrt{3}}{2}\\ \dfrac{\sqrt{3}}{2} &-\dfrac{1}{2}\end{bmatrix} & \begin{bmatrix} 1&0\\ 0 &-1\end{bmatrix}&\begin{bmatrix} -\dfrac{1}{2}& -\dfrac{\sqrt{3}}{2}\\ -\dfrac{\sqrt{3}}{2} &\dfrac{1}{2}\end{bmatrix} &\begin{bmatrix} -\dfrac{1}{2}&\dfrac{\sqrt{3}}{2}\\ \dfrac{\sqrt{3}}{2} &\dfrac{1}{2}\end{bmatrix} & \begin{array}{ll} (x,y)\\(R_x,R_y)\end{array}\\
# \hline
# \end{array}\qquad\qquad\text{(9.12)}$$
# 
# The characters for the second row have been added without showing the calculation. These can be obtained as described for $C_{2V}$ but with angle $2\pi/3$ etc. where necessary. This transformation matrix is very cumbersome to use, but it turns out that it is sufficient for most purposes to tabulate just the sum of the diagonals, the trace of the matrices. Notice that the first two rows have a single value as a matrix, i.e. a 1d matrix but that the last row has 2d matrices thus a trace is the sum of these diagonals. The rudimentary group table so formed is,
# 
# $$\displaystyle \begin{array}{c|c|c|c|c|c|c|c|}
# \hline
# C_{3V} &E &C_3&C_3^2 &\sigma_v&\sigma_V'&\sigma_v''& \text{rot/trans}\\
# \hline
# A_1&1&1&1&1&1&1&z\\
# A_2&1&1&1&-1&-1&-1&R_z\\
# E&2&-1&-1&0&0&0& (x,y),(R_x,R_y)\\
# \hline
# \end{array}\qquad\qquad\qquad \text{(9.13)}$$
# 
# where the doubly degenerate symmetry species is labelled $E$. As each type of reflection has the same behaviour as the others these can be combined. $C_{3V}$ and $C_{3V}^2$ can similarly be combined. The functions $x,y,z$ are needed when using symmetry to determine if dipole transitions are allowed or not, the squared terms $x^2+y^2,z^2$ etc are needed for Raman transitions, since Raman depends on change in polarisability which in projection is a squared function, (area) and the other functions are used in bonding as they show the behaviour of d- and f-orbitals. How there functions are assigned to a symmetry species is described next. The resulting table is 
# 
# $$\displaystyle \begin{array}{c|c|c|c|c|c|c|c|}
# \hline
# C_{3V} &E &2C_3 &3\sigma_v& \text{Rot/Trans}& \text{d orbitals} & \text{f orbitals}\\
# \hline
# A_1&1&1&1&z&x^2+y^2,z^2&z^3,x(x^2-3y^2)\\
# A_2&1&1&-1&R_z&&y(3x^2-y^2\\
# E&2&-1&0& (x,y),(R_x,R_y)&(x^2-y^2,xy)(xz,yz)&(xz^2,yz^2)(xyz,z(x^2-y^2))\\
# \hline
# \end{array}\qquad\qquad\qquad\text{(9.14)}$$

# ## 6.9.3 Atomic wavefunctions
# 
# Atomic wavefunctions are particularly important to chemists as they define where electrons are to be placed and so their symmetry is important to bonding theory. To find out which symmetry species in each point group these orbitals correspond to, the same general method is followed as explained above. Only the orbital's shape, the angular part of the wavefunction is needed because the radial parts are spherically symmetrical. We shall examine how the set of d-orbitals changes after different types of symmetry operation. Figure 21b shows the effect symmetry operations have on a $d_{xy}$ orbital in the $C_{4V}$ point group. Why the characters are $\pm 1$ as opposed to $0, \pm 1/2$ or other values is worked out. 
# 
# ![Drawing](matrices-fig21ac.png)
# 
# Figure 21b. Showing the effect symmetry operations have on a $d_{xy}$ orbital in the $C_{4V}$ point group. The characters are also listed.
# ____________
# The angular part of d-orbitals are usually described in spherical polar coordinates $r,\theta,\phi$ as the spherical harmonics $Y_\ell^m(\theta,\phi)$ with angular momentum quantum number $\ell = 2$ for d-orbitals and the 'magnetic' or azimuthal quantum number $m=0,\pm 1,\pm 2\pm\ell$ which gives five degenerate orbitals $m=-2-1,0,1,2$ when $\ell=2$. The radial coordinate can be taken to be constant because the radial part of the wavefunction is spherically symmetrical so is unchanged by any symmetry operation.  
# 
# Orbitals are degenerate solutions of a differential equation, the Schroedinger eqn., and therefore a linear combination is also a solution meaning that when an orbital is moved to a new position a linear combination of spherical harmonics is produced. The equations produced are quite general and apply to any point group. The starting equation to produce a new set of coordinates by operating on an orbital is 
# 
# $$\displaystyle f'= O^Sf_i\qquad\quad \tag{9.15}$$
# 
# where $f_i$ is a spherical harmonic $ Y_\ell^i(\theta,\phi) $ and $f'$ one or more spherical harmonics (a linear combination) in transformed coordinates and $O^S$ is an operator for any symmetry operation, $C_3, \sigma_v, i$ etc., on orbital $i$ (see Bishop 1993, chapter 5). The spherical harmonics will be identified by the subscripts in the orbitals as
# 
# $$\displaystyle f_1=d_{x^2-y^2},\quad f_2=d_{xy},\quad f_3=d_{xz},\quad f_4=d_{yx},\quad f_5=d_{z^2}$$
# 
# and replacing the $r,\theta,\phi$ coordinates in $f$'s by equivalent $x,y,z$ which are the ones we use, and $f'$ coordinates by $x'y'z'$, ignoring constants and the radial parts, the functions become
# 
# $$\displaystyle f_1=(x^2-y^2)/2,\quad f_2=xy,\quad f_3=xz,\quad f_4=yz,\quad f_5=z^2$$
# _________
# ### **(i) Converting the orbitals to $x,y,z$**
# 
# The spherical polar coordinates are shown in chapter 5 fig 17. As an example, let $\sigma=Zr/a_0$ and $a_0$ be the Bohr radius and $Z$ the atomic number then
# 
# $$\displaystyle \begin{align}\psi(3d_{xy})&=(81\sqrt{2\pi})^{-1}\left(\frac{Z}{a_0}\right)^{3/2}\sigma^2e^{-\sigma/3}\sin^2(\theta)\sin(2\phi)\\
# &\sim r^2 \sin^2(\theta)\sin(2\phi)\\ &\sim r^2  \sin^2(\theta)2\sin(\phi)\cos(\phi)=2\underbrace{r\sin(\theta)\cos(\phi)}_{ x}\cdot \underbrace{r\sin(\theta)\sin(\phi)}_{ y}\sim xy\end{align}$$
# ___________
# 
# 
# ### 6.9.4 Working out how rotation and reflection affects an orbital
# 
# ### **(i) Rotations**
# 
# Returning to eqn. 9.15, we focus on the $f_1$ orbital and start by rotating it according to a $C_n$ operation where $\theta=2\pi/n$.  The rotation matrix of eqn. 9.4 is
# 
# $$\displaystyle \begin{bmatrix}x'\\y'\\z' \end{bmatrix}=\begin{bmatrix} \cos(\theta)&\sin(\theta)&0 \\ -\sin(\theta) & \cos(\theta) & 0 \\ 0&0&1 \end{bmatrix} \begin{bmatrix}x\\y\\z \end{bmatrix}\qquad\qquad\qquad\text{(9.16)}$$
# 
# which transforms $x,y,z$ vectors to a new orientation at $x',y',z'$ giving 
# 
# $$\displaystyle x' = x\cos(\theta)+y\sin(\theta),\quad y' = -x\sin(\theta)+y\cos(\theta),\quad z'=z$$
#  
# The inverse equations will shortly be needed and are
# 
# $$\displaystyle \begin{bmatrix}x\\y\\z \end{bmatrix}=\begin{bmatrix} \cos(\theta) & -\sin(\theta)& 0 \\ \sin(\theta) & \cos(\theta) & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix}x'\\y'\\z' \end{bmatrix}\qquad\qquad\qquad\text{(9.17)}$$
# 
# In eqn. 9.15 the coordinates on the rhs. are in $x,y,z$ as the original orbital $f_1$ is in $x,y,z$, but $x',y',z'$ after transforming  but we want both side to have the same set of coordinates. Transforming using matrix eqn. 9.17 does this giving,
# 
# $$\displaystyle x = x'\cos(\theta)-y'\sin(\theta),\quad y = x'\sin(\theta)+y'\cos(\theta),\quad z=z'$$
# 
# therefore operating on $f_1$  by rotation about the principle axis by $\theta$, which is $O^\theta$, gives
# 
# $$\displaystyle \begin{align} f'&=O^\theta(x^2-y^2)/2 \\&=\big(x'\cos(\theta)-y'\sin(\theta)\big)^2/2-\big(x'\sin(\theta)+y'\cos(\theta)\big)^2/2\\&= \cos(2\theta)f_1'-\sin(2\theta) f_2'\end{align} \qquad\qquad \qquad \text{(9.18)}$$
# 
# As the coordinates are now the same on both sides we can make them both $x,y,z$ (instead of $x'y'z'$) and following the same procedure for the other orbitals gives,
# 
# $$\displaystyle \begin{align}O^\theta d_{xy}&= \sin(2\theta)d_{x^2-y^2}+\cos(2\theta)d_{xy}\\
#  O^\theta d_{xz}&= +\cos(\theta)d_{xz}-\sin(\theta)d_{yz}\\
#  O^\theta d_{yz}&= \sin(\theta)d_{xz}+\cos(\theta)d_{yz}\\
#  O^\theta d_{x^2}&= d_{z^2}\end{align}$$
# 
# which can be put into matrix form and in order $f_1\to f_5$ as *columns*
# 
# $$\displaystyle \quad\qquad\begin{matrix} x^2-y^2\quad\;& xy\quad\;& xz\quad\;&yz \quad\;& z^2\end{matrix}\tag{9.19}$$
# $$\displaystyle  O^\theta(f) = \begin{bmatrix}\cos(2\theta) & \sin(2\theta) & 0 & 0 & 0 \\ -\sin(2\theta) & \cos(2\theta) & 0 & 0 & 0\\0 & 0 & \cos(\theta) & \sin(\theta) & 0\\0 & 0 & -\sin(\theta) & \cos(\theta) & 0\\0 & 0 & 0 & 0 & 1\\ \end{bmatrix}$$
# 
# and in this form it is clear that there are two $2\times 2$ and one $1\times 1$ block matrices. When needed, the rotation reflection operation $S_n, i$ and $E$ can be written down directly, see eqns. 9.5, 9.6.  
# 
# Some code to work out the rotation matrices is shown next.

# In[2]:


# Rotation matrices for d-orbitals using Sympy

x,y,z,theta,n = symbols('x,y,z,theta,n')

def rotated(theta):
    xd =  x*cos(theta) - y*sin(theta)
    yd =  x*sin(theta) + y*cos(theta)
    zd = z
    f01=[(xd**2 - yd**2)/2, xd*yd, xd*zd, yd*zd, zd*zd]
    f02=[]
    for i in range(5):
        f02.append(simplify(expand(f01[i] ) )  )
    return f02

rotated(theta)


# In[3]:


# Example C2 rotation 
n = 2
theta = 2*pi/n    
theta*180/pi,rotated(theta)


# ### **(ii) Reflection**
# 
# The reflection matrices can be generated starting with say $\sigma_{xz}$ (see fig 21a) and using the rotation matrix to generate others, for example if the mirror plane is at angle $\theta$ to the $xz$ plane 
# 
# $$\displaystyle \sigma(\theta) =\begin{bmatrix}\quad \cos(\theta)&\sin(\theta)&0 \\ -\sin(\theta) & \cos(\theta)&0\\0&0&1  \end{bmatrix}\begin{bmatrix} 1 &0&0 \\ 0 & -1&0\\0&0&1  \end{bmatrix}=\begin{bmatrix} \cos(\theta)&-\sin(\theta)&0 \\ -\sin(\theta) & -\cos(\theta)&0\\0&0&1  \end{bmatrix}$$
# 
#  The inverse matrix, which it turns out is the same matrix, (as a check $ \sigma(\theta)\sigma( \theta)^{-1}=\pmb 1$) is 
#  
# $$\displaystyle \begin{bmatrix}x\\y\\z\end{bmatrix}=\begin{bmatrix} \cos(\theta)&-\sin(\theta)&0 \\ -\sin(\theta) & -\cos(\theta)&0\\0&0&1  \end{bmatrix}\begin{bmatrix}x'\\y'\\z'\end{bmatrix}$$
#  
# 
# Starting with $d_{x^2-y^2}$ for any mirror plane obtained from the $yz$ plane by rotation by $\theta$ and using the same method as for rotations produces,
# 
# $$\displaystyle \begin{align}O^\sigma d_{x^2-y^2}&=((\cos(\theta)x'-\sin(\theta)y')^2-(\sin(\theta)x'+\cos(\theta)y')^2)/2\\&= \cos(2\theta)d_{x^2-y^2}-\sin(2\theta)d_{xy}\\
# O^\sigma d_{xy}&= -\sin(2\theta)d_{x^2-y^2}-\cos(2\theta)d_{xy}\\
# O^\sigma d_{ xz}&=\cos(\theta)d_{xz}-\sin(\theta)d_{yz}\\
# O^\sigma d_{yz}&=-\sin(\theta)d_{xz}-\cos(\theta)d_{yz}\\
# O^\sigma d_{z^2}&=d_{x^2}\\
# \end{align}$$
#  
# These equations can be put into a matrix as columns
# 
# $$\displaystyle \begin{matrix}\qquad\qquad x^2-y^2\qquad\;& xy\quad \quad\;& xz\quad\;&yz \quad\;& z^2\end{matrix}\tag{9.20}$$
# $$\displaystyle \quad O^\sigma(f) = \begin{bmatrix}\cos(2\theta) & -\sin(2\theta) & 0 & 0 & 0 \\ -\sin(2\theta) & -\cos(2\theta) & 0 & 0 & 0\\0 & 0 & \cos(\theta) & -\sin(\theta) & 0\\0 & 0 & -\sin(\theta) & -\cos(\theta) & 0\\0 & 0 & 0 & 0 & 1\\ \end{bmatrix}$$
# 
# The $C_{4V}$ point group has operations $E, 2C_4, C_2, 2\sigma_v,2\sigma_d$  and we can work out the rotation and reflection matrices for each operation as shown below. Each matrix is block diagonal, two blocks of $2\times 2$ and one of $1\times 1$ which is the $d_z^2$ orbital. Only the $d_{xz}$ and $d_{yz}$ orbitals are mixed by the $C_4$ and $\sigma_d$ operations so these are added as matrices in the table below, the other are added as diagonal matrices in this row but in the rest of the table consists of single characters as the total matrix in this case becomes one of two blocks of $1\times 1$ one of $2\times 2$ and one of $1\times 1$. 
# 
# $$\displaystyle \begin{array}{cccccccc}
# C_{4V}& E & C_4 & C_4^3 & C_2 & \sigma_V & \sigma_V' & \sigma_d & \sigma_d'\\
# \hline
# d_{z^2}  & 1 & 1 & 1 & 1 & 1 & 1 & 1& 1\\
# d_{x^2-y^2} & 1 & -1 & -1 &  1 &  1  &  1 & -1 & -1\\
# d_{xy}      & 1 & -1 & -1 &  1 & -1  & -1 &  1 &  1\\
# d_{xz}d_{yz}&\begin{bmatrix}1,0\\0,1\end{bmatrix} &\begin{bmatrix}0,1\\-1,0\end{bmatrix}&\begin{bmatrix}0,-1\\1,0\end{bmatrix}&\begin{bmatrix}-1,0\\0,-1\end{bmatrix}&\begin{bmatrix}-1,0\\0,1\end{bmatrix}&\begin{bmatrix}-1,0\\0,1\end{bmatrix}&\begin{bmatrix}0,1\\1,0\end{bmatrix}&\begin{bmatrix}0,1\\1,0\end{bmatrix}\\        
# \hline
# \end{array}$$
# 
# To complete the table so far the sum of the diagonals (trace) of the $2\times 2$ matrices are added together. The $C_4$ and reflections can be grouped where the characters are the same, which lead to the table
# 
# $$\displaystyle \begin{array}{rrrrr}
# & E & 2C_4 & C_2 & 2\sigma_V & 2\sigma_d \\
# \hline
# d_{z^2}  & 1 & 1 & 1 & 1 & 1 \\
# d_{x^2-y^2} & 1 & -1 &  1 &  1  & -1 \\
# d_{xy}      & 1 & -1 &  1 & -1  &  1 \\
# d_{xz}d_{yz}& 2& 0 & -2 & 0 & 0\\        
# \hline
# \end{array}$$
# 
# The number of symmetry species must be equal to the number of classes (columns) in the table. This means that although the d-orbital set forms complete basis set one symmetry species is missing from the table. We can use eqn 9.9, and the orthogonality of rows to find the other row of characters. The value under the identity is found first using the fact that the sum squared of the characters equals the number of operations. This is $1$ and is added to the table. 
# 
# $$\displaystyle \begin{array}{rrrrr}
# & E & 2C_4 & C_2 & 2\sigma_V & 2\sigma_d \\
# \hline
# d_{z^2}     & 1 &  1 &  1 &  1  & 1 \\
# d_{x^2-y^2} & 1 & -1 &  1 &  1  & -1 \\
# d_{xy}      & 1 & -1 &  1 & -1  &  1 \\
# d_{xz}d_{yz}& 2 &  0 & -2 &  0  & 0\\
# \Gamma & 1& a & b & c & d\\
# \hline
# \end{array}$$
# 
# for example row  1 and row $\Gamma$ give $1+2a+b+2c+2d=0$, etc which results in $a=b=1,c=d=-1$. Re-ordering the rows and adding the Mulliken labels the complete table is 
# 
# $$\displaystyle \begin{array}{c|rrrrr}
# C_{4V}& E & 2C_4 & C_2 & 2\sigma_V & 2\sigma_d \\
# \hline
# A_1 & 1 &  1 &  1 &  1  & 1 & \\
# A_2 & 1 &  1 &  1 & -1  &-1 & \\
# B_1 & 1 & -1 &  1 &  1  & -1 & \\
# B_2 & 1 & -1 &  1 & -1  &  1 & \\
# E   & 2 &  0 & -2 &  0  & 0& \\
# \hline
# \end{array}$$
# 
# How different functions transform in this point group is not yet added but we do know what these are for d-orbitals by the way this calculation has been done. The next section describes how the symmetry species for a given function can be calculated. 

# ### 6.9.5 Working out an orbital's symmetry species
# 
# To workout what symmetry species each orbital corresponds to, or in general any basis set function,  we have to operate on each orbital in turn and then use a projection operator. (See 6.16 for another example). First a table is made by describing the action on each orbital. This produces $\pm$ itself or $\pm$ another orbital function or a combination thereof. 
# 
# The $C_{4V}$ point group has operations $E, 2C_4, C_2, 2\sigma_v,2\sigma_d$. Using the equations derived above the effect a $C_4$ rotation on each d-orbital can be listed. Similarly for reflections, but as the class is $2$ for reflections and $C_4$ these must be split into two, e.g. $C_4$ and $C_4^3$ for example. A $5 \times 8$ table is then formed in which each orbital is subjected to each symmetry operation In each operation the orbital is changed into $\pm$ into itself or $\pm$ one of the other orbitals because angles are multiples of $90^\text{o}$. The orbitals $f_1\cdots f_5$ correspond to $d_{x^2-y^2},d_{xy},\cdots$ as shown in the first two columns.
# 
# $$\displaystyle \begin{array}{c|ccccc}
#  &E & C_4 & C_4^3 & C_2 & \sigma_v & \sigma_V' & \sigma_d & \sigma_V'\\
# \hline
# d_{x^2-y^2}&f_1 & -f_1 & -f_1 &  f_1 &  f_1 &  f_1 & -f_1 & -f_1\\
# d_{xy}     &f_2 & -f_2 & -f_2 &  f_2 & -f_2 & -f_2 &  f_2 &  f_2  \\
# d_{xz}     &f_3 & -f_4 &  f_4 & -f_3 &  f_3 & -f_3 & -f_4 &  f_4 \\
# d_{yz}     &f_4 &  f_3 & -f_3 & -f_4 & -f_4 &  f_4 & -f_3 &  f_3 \\
# d_{z^2}    &f_5 &  f_5 &  f_5 &  f_5 &  f_5 &  f_5 &  f_5 &  f_5  \\
# \hline
# \end{array}\qquad\qquad\qquad \text{(9.20.1)}$$
# 
# The next step is to use the _projection operator_ to workout which symmetry species each orbital belongs to and this is done using the characters of the point group. This is quite simple but care is needed. We must multiply each element of the first row in table above with each character in the first row of the character table and sum the values. Next, the second row of the character table is multiplied and summed in the same fashion and so on until all symmetry species ($A_1,\cdots E$) have been operated on by the first row in the table 9.20.1. This is then repeated for each orbital, i.e. each row in the $C_{4V}$ table.
# 
# The equation for the projection operator is 
# 
# $$\displaystyle L_M=\frac{d}{h}\sum_{j=1\cdots h}\chi_j^MO^S_j \tag{9.21}$$
# 
# where $M$ is the Mulliken symmetry species label, $A_g, B_{3g}$, and so forth, $h$ is the order of the group, $d$ the dimension of the irreducible representation, and the sum is over all the classes. Since we only want to know which orbital belongs to which symmetry species $d/h$ can be ignored. The function $L_M$ is the list of orbital names ($d_{xy}$ etc.) which belong to symmetry species $M$. The simplest way to perform the calculation, and avoid arithmetic slips, is to use some code and use Sympy for symbolic calculation. The matrix F below contains the orbital changes for each operation in $C_{4V}$ and PG is the character table. Matrices AB and Dorb are just for labelling the results.
# 
# The calculation shows that the orbitals belong to symmetry species as follows
# 
# $$\displaystyle d_z^2 \to A_1,\quad d_{x^2-y^2}\to B_1,\quad d_{xy}\to B_2,\quad (d_{xz},d_{yz})\to E$$
# 
# The full table is shown below together with the the symmetry species for p-orbitals and dipoles which transform as translations in $x,y$ and $z$.
# 
# $$\displaystyle \begin{array}{c|rrrrr|c|c}
# C_{4V}& E & 2C_4 & C_2 & 2\sigma_V & 2\sigma_d \\
# \hline
# A_1 & 1 &  1 &  1 &  1  & 1 & z &z^2\\
# A_2 & 1 &  1 &  1 & -1  &-1 & \\
# B_1 & 1 & -1 &  1 &  1  & -1 & & x^2-y^2\\
# B_2 & 1 & -1 &  1 & -1  &  1 & & xy\\
# E   & 2 &  0 & -2 &  0  & 0&(x,y) & (xy,yz)\\
# \hline
# \end{array}$$

# In[4]:


# projection operator method for d-orbitals in C4V point group

f1,f2,f3,f4,f5,row,col, F,PG = symbols('f1,f2,f3,f4,f5,row,col,F,PG')

# matrix of d-orbital changes f1,..f5 as in text
F  = Matrix([[f1,-f1,-f1, f1,f1, f1,-f1,-f1],[f2,-f2,-f2, f2,-f2,-f2, f2,f2]\
           , [f3,-f4, f4,-f3,f3,-f3,-f4, f4],[f4, f3,-f3,-f4,-f4, f4,-f3,f3],[f5,f5,f5,f5,f5,f5,f5,f5]])

PG = Matrix([[1, 1, 1,1,1,1, 1, 1],[1, 1, 1,1,-1,-1,-1,-1],\
             [1,-1,-1,1,1,1,-1,-1],[1,-1,-1,1,-1,-1, 1, 1],[2,0,0,-2,0,0,0,0]])  # point group

AB   = Matrix(['A1','A2','B1','B2','E'])              # Mulliken labels
Dorb = Matrix(['d(x2-y2)','dxy','dxz','dyz','dz2'])   # symmetry operations.
F,PG


# In[5]:


#print(shape(F),shape(PG))
Fcol = 8  # F columns
Frow = 5  # F rows
arow = 5  # PG rows
for i in range(arow):               #  rows of orbitals
    for k in range(Frow):           #  rows of characters
        s = 0
        for j in range(Fcol):
            s = s + F[k,j]*PG[i,j]  # make product then sum values 
        if s != 0:
            print('{:4s} {:4s}'.format( str(AB[i]), str(Dorb[k]) ) )    # print( 'sum',s) 


# ## 6.10 Similarity and Classes
# 
# If symmetry elements in a group $C_3^+, C_3^-, \sigma$, etc. are equivalent they satisfy the _similarity transformation_. For example, if $A, B$, and $C$ are elements of a group then $A$ and $B$ are equivalent, and are said to be *conjugate*, only if they satisfy the similarity transform
# 
# $$\displaystyle \pmb{A}=\pmb{C}^{-1}\pmb{BC}\qquad\qquad \tag{9.22}$$
# 
# Equivalent members of a group form a *class*, a class being a column of characters in the point group table. The number before the symmetry operation is the number of operations in the class, see figure 15. In $C_{2V}$ there is only one member of each class, in $C_{3V}$ there are two classes of $C_3$ operations and three of $\sigma_V$.
# 
# Looking at the multiplication table for $C_{2V}$, figure 18, the product $C_2\sigma_VC_2 = C_2\sigma'_V = \sigma_V$, and as $C_2$ is its own inverse or $C_2 = C_2^{-1}$, this equation has the form of a similarity transform, $\sigma_V = C_2^{-1}\sigma_VC_2$. As each class is one dimensional in this case, the result of the similarity transformation of $\sigma_V$ has to be $\sigma_V$. 
# 
# ![Drawing](matrices-fig21ae.png)
# 
# Fig21c. $C_{3V}$ rotations. The $C_3^-$ operation moves the vector along the $N-H_1$ bond so that $H_1 \to H_3$ etc. and $C_3^+$ moves $H_1 \to H_2$ etc. The first row of the $C_3^-$ matrix is therefore $[0\;0\;1]$ and the first row of the $C_3^+$ matrix $[0\;1\;0]$
# _________________
# 
# In $C_{3V}$ we have not yet worked out the direct product table and matrices, this is done in questions 17-19, but suppose that we want to see if the rotations $C_3^+$ and $C_3^-$ (figure 21c) are related by a similarity transform involving a mirror plane and if they are whether they belong to the same class. Using the H atoms In NH$_3$ as a basis the matrices are
# 
# $$\displaystyle C_3^-\equiv\begin{bmatrix} 0&0&1\\ 1&0&0\\0&1&0\end{bmatrix},\qquad \sigma_V\equiv\begin{bmatrix} 0&1&0\\ 1&0&0\\0&0&1\end{bmatrix} $$
# 
# where the $\sigma_V$ mirror plane is along the bond you choose to be N-H3.
# 
# and the transform is $\pmb{A} = \sigma_V^{-1}C_3^-\sigma_V$. By direct calculation, we find that $\sigma_V^{-1} = \sigma_V$, i.e. $\sigma_V$ is its own inverse, making the matrix product,
# 
# $$\displaystyle \pmb{A} = \begin{bmatrix} 0&1&0\\ 1&0&0\\0&0&1\end{bmatrix}\begin{bmatrix} 0&0&1\\ 1&0&0\\0&1&0\end{bmatrix}\begin{bmatrix} 0&1&0\\ 1&0&0\\0&0&1\end{bmatrix}= \begin{bmatrix} 0&1&0\\ 0&0&1\\1&0&0\end{bmatrix}=C_3^+$$
# 
# The matrix multiplication can be checked using Sympy.

# In[6]:


A, C3, SV = symbols('A, C3, SV')

SV = Matrix([[0,1,0],[1,0,0],[0,0,1]]  )  # sigma V matrix
C3 = Matrix([[0,0,1],[1,0,0],[0,1,0]]  )  # C3 minus matrix
A  = SV**(-1)*C3*SV
A


# The calculation indicates that $C_3^+$ and $C_3^-$ belong to the same class for all point groups that contain a $\sigma_V$ mirror plane. In $C_{3V}$ there is one column for $C_3$ operations and has two members in its class, $C_3^+$ and $C_3^-$. These are not usually expressed individually in the point group but instead $2C_3$ is used a column heading because the characters are the same for both operators.
# 
# ##  6.11 Direct Products
# 
# Using characters, it is simple to form a direct product of two or more symmetry species. If operators are $\pmb{A}$ and $\pmb{B}$, the direct product is written as $\pmb{A} \otimes \pmb{B}$. The symbol $\otimes $ means calculate the direct product by multiplying pairs of characters together column-wise. One of the other symmetry species of the point group must be produced. If either A or B is the totally symmetric representation, the top line in the table, the result is always the other symmetry species B or A respectively.
# 
# In the $C_{2V}$ table (see Section 6.9), the product $\pmb{B}_1 \otimes \pmb{B}_2 = \pmb{A}_2$ as may be seen by multiplying the two elements of symmetry species $B_1$ and $B_2$ column by column and identifying the pattern of characters produced. If two species that are doubly or triply degenerate ($E$ or $T$ Mulliken labels) form a direct product, this has to be reduced in the normal way to a sum of irreducible representations. For example, in $C_{3V}$, figure 15, the direct product $\pmb{E}\otimes \pmb{E}$ produces the result$\pmb{E}\otimes \pmb{E} = 4\pmb{E} \oplus \pmb{C}_3 \oplus 0\times\pmb\sigma_V$. The symbol $\oplus$ means the symmetry species are added or, more properly, that $4E,\, C_3$ but no $\sigma_V$ are included in $\pmb{E}\otimes \pmb{E}$. It is common in many texts just to use + instead of $\oplus$. Reducing direct products is explained in Section 6.13, but first one important use of them is illustrated.
# 
# ## 6.12 Allowed and Forbidden transitions. Vanishing Integrals
# 
# In the spectroscopy of molecules, their symmetry species together with the point group can be used to determine whether an electronic, vibrational, or rotational transition is going to appear in the spectrum. This can be reversed and the presence or absence of lines in a spectrum can sometimes be used to decide geometry. First, the method using the symmetry species is illustrated and then justified by examining the symmetry of the vibrational wavefunctions and normal modes in a molecule. You can simply follow the method illustrated in this first part without having to understand why it works. Finally a connection is made between the transition moment and the symmetry species, i.e. why is it that we can use symmetry to evaluate the integrals.
# 
# The transition moment $M$ is proportional to the expectation value of the operator for that type of transition.  This can be written for states $a$ and $b$ with wavefunctions $\psi_a,\psi_b$ and a transition operator $\vec{\mu}$ such as for a vibrational or electronic transition as 
# 
# $$\displaystyle M = \displaystyle \int \psi^{a*}\,\vec{\mu}\,\psi^b d\tau\qquad\qquad \tag{9.23}$$
# 
# where $\tau$ covers all the nuclear/electonic coordinates the wavefunctions may have. The _intensity_ of a transition is $M^2$. The transition operator could be a dipole for absorption/emission or polarisability for Raman transitions. The superscript * indicated a complex conjugate should be taken if the wavefunction is complex but has no effect otherwise. We shall assume that the wavefunctions are real and drop the * from equations.
# 
# Symmetry can be used to determine if the transition moment integral is finite or not, i.e. whether the integral vanishes or not. Only a knowledge of the 'shape' of the wavefunctions and operator are required and no integration is involved and is a sophisticated extension of the odd - even rules to determine if an integral is exactly zero or not. It should be remembered, however, that even if the transition is allowed its intensity may be very small. What the symmetry calculation does is to tell us whether the transition probability is expected to be *exactly zero* or not, and if not it does not tell us anything about what its value will be.
# 
# The absorption or emission of radiation involves an electric-dipole operator. This will transform as a linear vector in the $x-, y-$, or $z-$direction because it depends linearly on the change in charge distribution that occurs with the transition. The dipole moment is a vector $\vec{\mu} = q\cdot \vec{r}$ where $q$ is the charge distribution and $\vec{r}$ the displacement vector during a transition. Raman scattering, however,  depends on having a change in the _polarizability_ of the molecule $\vec\alpha$. This is a measure of how easily the electron 'cloud' forming the molecular orbitals changes shape in the presence of the electric field of the radiation. This change is proportional to operators in two dimensions $xy, x^2-y^2$, etc., and these are shown in the last column of the point group table. 
# 
# If a transition is allowed between two states $S_1, S_2$ with symmetry species $\displaystyle\Gamma_{S_1},\ \Gamma_{S_2}$ respectively, then the direct product 
# 
# $$\displaystyle \Gamma_{S_1} \otimes \Gamma_\mu \otimes \Gamma_{S_2} \to \text{must include totally symmetric symmetry species if transition allowed} $$
# 
# where $\Gamma \mu$ is the symmetry species of the operator for absorption/emission or that for Raman transitions. This direct product has to include the totally symmetric representation of the molecule's point group, which is always the top row of the character table. 
# 
# This equation can be rewritten in an equivalent form as 
# 
# $$\displaystyle \Gamma_{S_1} \otimes \Gamma_{S_2}= \Gamma_\mu  \to \text{ if the transition is allowed} $$
# 
# and the product can be made in any order. A check then has to be made in the point group to identify what species the product belong to, but usually this is done by consulting direct product tables. If one of the states involved is the ground state (say $\Gamma_{S_1}$), this always belongs to the totally symmetric representation (see below under 'polyatomic molecules') and as these characters are all $1$ multiplying by this leaves any other symmetry species unchanged and in this special case 
# 
# $$\displaystyle  \Gamma_{S_2}= \Gamma_\mu  \to \text{for ground state only, if the transition is allowed} $$
# 
# If the transition is of the electric dipole type then the operator's symmetry species, ($\Gamma_\mu$) must transform as $x, y$ or $z$ as shown in the third major column of the point group. For Raman transitions, the operator transforms as products $xy, yz$, etc. For example, in a molecule with $C_{2V}$ symmetry species such as SO$_2$, a Raman transition from a state with symmetry species $A_2$ to that with species $A_1$ will be allowed because an operator $\Gamma_\mu$ transforming as $xy$ belongs to the $A_2$ symmetry species, making
# 
# $$\displaystyle \Gamma_{S_1} \otimes \Gamma_{\mu(xy)} \otimes \Gamma_{S_2}= A_1 \otimes A_2 \otimes A_2 =A_1$$
# 
# where $A_1$ is the totally symmetric species. An electric dipole transition would not be allowed because no $x, y$ or $z$ operator belongs to the $A_2$ symmetry species necessary to make the product $A_1$, see the $C_{2V}$ character table above. A transition between the states with $B_2$ and $B_1$ symmetry is not allowed with an electric dipole operator even though both $x$ and $y$ operators have these symmetries. The reason is that the direct product is not totally symmetric. For the y-direction transition, 
# 
# $$\displaystyle  B_2  \otimes B_2  \ne B_1 $$
# 
# similarly the x-direction operator produces the direct product 
# 
# $$\displaystyle B_1  \otimes B_1 \ne B_2$$
# 
# The z-direction operator is also no good, producing an $A_2$ direct product. There are, however, allowed dipole transitions for example from a state with symmetry species $\Gamma_{S_1}=\Gamma_{S_2} =A_1$, with an $z$ direction dipole and also when $\Gamma_{S_1} =A_1,\, \Gamma_{S_1}=B_1$ with a $x$ direction dipole and $\Gamma_{S_1} =A_1,\, \Gamma_{S_1}=B_2$ with a $y$ direction dipole.

# ### **(i) Diatomic molecules**
# 
# The direct calculation of eqn. 9.23 has to give the same answers as using symmetry species and this is now illustrated, first with diatomic and then with polyatomic molecules.  Equation 9.23 is evaluated with harmonic oscillator wavefunctions. These have the form
# 
# $$\displaystyle \begin{array}{ll}\psi_0 = N_0e^{-\alpha q^2/2}& \psi_1=\sqrt{2\alpha}\, q\,\psi_0 \\ \psi_2 = 2^{-1/2} (2\alpha q^2-1)\,\psi_0 & \psi_3 = \sqrt{\alpha/3} ( 2\alpha^{3/2} q^3 - 3\sqrt{\alpha} \, q)\,\psi_0 \end{array}$$
# 
# where $q$ is the displacement from the equilibrium position,  $N_0=(\alpha/\pi)^{1/4}$ is the normalisation constant and the subscript to $\psi$ indicates the vibrational quantum number, $v=0,1,\cdots$. The constant $\displaystyle \alpha=\sqrt{\mu k}/2\hbar$ where $\mu$ is the reduced mass and $k$ the force constant; $\alpha$ has dimensions of 1/length$^2$. The wavefunctions are orthonormal, i.e. normalised and orthogonal to one another, this means that $\displaystyle \int \psi_n\psi_m dq=\delta_{n,m}$. 
# 
# In a vibrational transition an oscillating dipole is needed to couple the radiation to the molecule.  The dipole on a heteronuclear diatomic molecule naturally changes as the bond vibrates. As the extension and contraction of a  bond is small compared to the bond length ($\le 10$ %) the dipole similarly changes only slightly and can be calculated by expanding it as a Taylor series about $q=0$, the average internuclear position as, 
# 
# $$\displaystyle \mu=\mu_0 +q\left(\frac{d\mu}{dq}\right)_{q=0}+\cdots$$
# 
# thus we assume that the transition dipole is a linear function of the atom's displacement. The transition moment integral, eqn. 9.23 between energy levels with quantum numbers $v=0 \to v=1$ then becomes
# 
# $$\displaystyle M = \int_{-\infty}^\infty \psi_0\,\mu\,\psi_1=c\mu_0\int_{-\infty}^\infty \psi_0\psi_1dq+c\left(\frac{d\mu}{dq}\right)_{q=0}\int_{-\infty}^\infty \psi_0\,q\,\psi_1dq$$
# 
# where the constant $c=\sqrt{2\alpha}N_0^2$. The first term is zero as the wavefunctions are orthogonal and this can be confirmed by direct integration as $\psi_1$ is an odd function. The second integral has a finite value as it is an even function in $q$, ignoring the constants for clarity,
# 
# $$\displaystyle \int_{-\infty}^\infty (\psi_0\,q)\,\psi_1dq=\int_{-\infty}^\infty \psi_1\,\psi_1dq\ne 0$$
# 
# The integral evaluates to $\displaystyle \sqrt{\frac{\pi}{\alpha}}N_0^2$ but this is unimportant as the value of the derivative ($d\mu/dq$) is generally not known, but is $\sim 10$ Debye/nm. What is important, however, is that this integral does not vanish and so the transition $v=0\to 1$ is allowed, but what we don't know is exactly how intense it will be. 
# 
# In the harmonic oscillator the symmetry of the wavefunctions ensures that only the $v=0\to 1$ transition can occur, but in reality the potential is anharmonic and transitions to other levels do occur, they are weak and called *overtones*, $v=0 \to v\ne 1$, as a ball-park number weak means $\lt 0.1$ of allowed transition . Weak transitions can also occur when more than one upper vibrational level is excited, these are called *combination bands*. 
# 
# ### **(ii) Polyatomic molecules**
# 
# The vibrations of the polyatomic vibrations are not those of individual pairs of atoms moving randomly with respect to others, but a collective in-phase motion of all atoms. The way these move is governed by the molecules symmetry and are called *normal modes* of which there are $3N-6$ for $N$ atoms and $3N-5$ if the molecule is linear. These normal modes are described by the complicated motion of the displacement of each atom, the *normal coordinates* and 'modes' and 'coordinates' are sometimes used interchangeably.
# 
# The energy of a molecule is the sum of kinetic and potential energy terms of the $3N-6$ modes, all of which contain squared terms
# 
# $$\displaystyle E=\frac{1}{2}\dot q^2_1 + \frac{1}{2}\lambda_1q^2_1+\frac{1}{2}\dot q^2_2 + \frac{1}{2}\lambda_2q^2_2+\cdots + \frac{1}{2}\lambda_{3N-6}q^2_{3N-6}$$
# 
# where $\dot q^2$ is the time derivative, i.e. the velocity. (The mass is absorbed into $q$). After a symmetry operation the energy must be unchanged and replacing the displacements $q$ by vectors means that because of the squared terms the normal coordinate is either unchanged or only changes sign, i.e. 
# 
# $$\displaystyle \vec q\overset{ sym\;op}\longrightarrow \pm \vec q$$
# 
# This result means that any normal coordinate is either symmetric or antisymmetric with respect to the symmetry operation.  In CO$_2$ for example, there is only a symmetric stretch, an antisymmetric stretch and symmetric bends and only arrows need be drawn to show these motions while keeping the centre of gravity unchanged.
# 
# If there is a degeneracy then the energy is
# 
# $$\displaystyle \begin{align}E&=\frac{1}{2}\dot q^2_1 + \frac{1}{2}\lambda_1q^2_1+\frac{1}{2}\dot q^2_2 + \frac{1}{2}\lambda_1q^2_2+\cdots + \\&= \frac{1}{2}(\dot q^2_1+\dot q^2_2)+\frac{\lambda_1}{2}(q^2_1+q^2_2)\end{align}$$
# 
# so the symmetry operation should not change $q^2_1+q_2^2$ which means that for degenerate vibrations the normal coordinate becomes a combination of $q_1$ and $q_2$. 
# 
# The wavefunction for a polyatomic molecule is the product of those for the normal modes,
# 
# $$ \displaystyle \Psi=c\psi(q_1)\psi(q_2)\cdots \psi(q_{3N-6})$$
# 
# which for $v=0$ and harmonic oscillators  becomes
# 
# $$\displaystyle \Psi_{v=0}=const\cdot e^{-\alpha q_1^2/2}e^{-\alpha q_2^2/2}\cdots$$
# 
# In this $v=0$  wavefunction each exponential $e^{-\alpha q_i^2/2}$ is symmetric about $q=0$ and therefore so is the whole wavefunction. In symmetry terms this means that
# 
# >the ground state wavefunction belongs to the totally symmetric representation
# 
# which is the top row in any point group. A symmetry operation only changes $q\to \pm q$ and as only $q$ squared occurs in the wavefunction a non-degenerate vibration is unchanged by any symmetry operation. A degenerate pair of vibrations, for example, contributes terms such as $\displaystyle e^{-\alpha(q_1^2+q_2^2)}$ and a symmetry operation leaves the value of $q_1^2+q_2^2$ unchanged thus the ground state, $v = 0$ wavefunction is totally symmetric to any symmetry operation. 
# 
# When $v = 1$ is excited the wavefunction has the form $\Psi_{v=1}=q\,\Psi_{v=0}$ which means that it transforms as the normal coordinate $q$ which transforms in the same way as the symmetry species of one of translations $x,y$ or $z$ in the point group table.

# 
# ### **(iii) Connecting the transition moment integral to symmetry species**
# 
# The transition moment integral has been evaluated using the odd/even nature of wavefunctions and, without explanation, as the product of their symmetry species. In polyatomic molecules a molecular orbital or a normal mode has a complicated mathematical description and it is much easier to use their symmetry properties to decide if the transition is allowed or not. The down-side of this is that the magnitude of the transition is not known, but in practice this is not so important, the presence or otherwise of a transition is usually enough to determine the information we seek, i.e. what a molecule's point group is as this gives clues about its structure.
# 
# An absorption spectrum is a observable quantity and its transitions must have the same energy for all indistinguishable orientations of the molecule. It follows therefore that *any transition moment integral must have the same value for all symmetry operations on the molecule*. For example rotating by $180^\text{o}$ so that the molecule is indistinguishable from that before the rotation cannot change the molecules energy levels and so cannot change its spectrum. The same is true for all operations.
# 
# The  remaining task is to connect the integral eqn. 9.23 with the direct product of the symmetry species of each term in the integral. This removes the task of integrating so that the transition can be determined to be either allowed or forbidden. We therefore want to show that the direct product of the three functions in the integral contains the totally symmetric representation $\Gamma^1$. 
# 
# To find if the totally symmetric representation occurs in the product $\Gamma^a\otimes \Gamma^\mu\otimes \Gamma^b$ of the irreducible representations $\Gamma^a,\Gamma^b$ and $\Gamma^\mu$ we use the point group and multiply character by character for each symmetry species $\Gamma$ and sum over all the classes. This is very similar to the method given below (section 6.13) to reduce a representation. The sum is
# 
# $$\displaystyle  S = h^{-1}\sum_i g_i\chi^a_i\chi^\mu_i\chi^b_i\qquad\qquad \tag{9.24}$$
# 
# where $i$ sums over all operations, i.e. all classes. The $g_i$ is the number in each class in the point group. If this summation is *not zero* it means that the totally symmetric representation is present just once in the product and therefore the integral will not be zero. We can write the integration as
# 
# $$\displaystyle \int \psi^a\,\vec{\mu}\,\psi^b d\tau\overset{all\,sym\; ops}\longrightarrow S\int \psi^a\,\vec{\mu}\,\psi^b d\tau$$
# 
# where the transition is allowed when $S \ne 0$. The integral itself is unchanged in any of the symmetry operations as shown above. 
# 
# The same conclusion can be arrived at by calculating the direct product using only two of the three terms, such as $\Gamma^a\otimes\Gamma^B$ and determining if this product contains the third, e.g. $\Gamma^\mu$. This means not summing but simply identifying the pattern of characters. However, if a doubly or triply degenerate representation is present a reducible representation can be produced and the tabular method (section 6.13) will in addition be needed to find the symmetry species.
# 
# As an illustration we use the $D_4$ point group with eqn. 9.24. Suppose that the states $a$ and $b$ belong to the symmetry species $B_1$ and $B_2$ respectively and we want to know if any transition with a dipole in any of $x,y,z$ directions is possible. By 'state' is meant, for example, two different vibrational levels or two electronic states such as the ground state and an excited state.
# 
# $$\displaystyle \begin{array}{c|rrrrr|c|c}
# D_{4}& E & 2C_4 & C_2 & 2C' & 2C'' \\
# \hline
# A_1 & 1 &  1 &  1 &  1  & 1 &   &z^2\\
# A_2 & 1 &  1 &  1 & -1  & -1 & z &\\
# B_1 & 1 & -1 &  1 &  1  & -1 & & x^2-y^2\\
# B_2 & 1 & -1 &  1 & -1  &  1 & & xy\\
# E   & 2 &  0 & -2 &  0  & 0&(x,y) & (xy,yz)\\
# \hline
# \end{array}$$
# 
# The direct calculation for $z\equiv A_2$ transition has the species $A_2, B_1, B_2$, note that the multiplication order does not matter and that $A_2$ represents the z-direction dipole. The summation eqn 9.24 has symmetry operations $E + 2C_4 + \cdots$ and the sum is shown in that order. The order of the group is $h=1+2+1+2+2$ and the number of times the totally symmetric representation is present is 
# 
# $$\displaystyle S = \frac{1}{8}(1 + 2\cdot 1\cdot(-1)\cdot(-1) + 1 + 2 + 2)= 1$$
# 
# so that this transition is allowed. We can also see this by multiplying together the characters of $B_1$ and $B_2$ which gives $1, 1, 1, -1, -1$ which is $A_2$. 
# 
# If the transition were in the $x$ or $y$ direction, which are equivalent in this point group, the species would be $E$ for the dipole and $B_1,B_2$ for the states. The calculation now is
# 
# $$\displaystyle S = \frac{1}{8}(2 + 0 -2 + 0 + 0)= 0$$
# 
# so an $x$ or $y$ transition is not allowed between $B_1\to B_2$ or in fact between any $A$ or $B$ symmetry states. If the states were both $E$ symmetry species then a $z$ dipole ($A_2$) produces
# 
# $$\displaystyle S = \frac{1}{8}(4 + 0+ 4 + 0 + 0)= 1$$
# 
# and the transition would be allowed. If there were $x$ or $y$ dipoles and two states of $E$ symmetry the calculation is equivalent to $E\otimes E\otimes E$ 
# 
# $$\displaystyle S = \frac{1}{8}(8 + 0 -8 + 0 + 0)= 0$$ 
# 
# so this transition would not be allowed.
# 
# Having learned how to find from a fundamental perspective whether or not a transition is allowed or not, we tend not to use the method of eqn. 9.24, basic though it is, but instead use direct product tables and look up the products of the symmetry species. Disappointing but practical.
# 
# ### **(iv) El-Sayed Rules in spectroscopy**
# 
# The intersystem crossing transition between a singlet excited state to a triplet state is formally forbidden but may occur because of spin-orbit coupling allowing a change in angular momentum to occur. Transitions are  enhanced by heavy atoms, often called the 'heavy-atom effect', however, paramagnetic species, such as O$_2$ also enhance spin-orbit coupling. The spin-orbit operator has the form $H_{SO}\sim \pmb\ell\cdot \pmb s$ where $\pmb\ell$ and $ \pmb s$ are the orbital and spin angular momentum vectors respectively. In terms of symmetry the spin-orbit operator transforms as $R_k,k={x,y,z}$ and appears in column 3 in the point group table. 
# 
# The 'product of symmetry species' approach can be used in this case and gives rise to the  El-Sayed  rules i.e.
# 
# >'intersystem crossing is faster when there is a change of symmetry'. 
# 
# For example, experiment shows that the rate constant of intersystem crossing 
# 
# $$\displaystyle ^1\pi\pi^* \to \,^3n\pi^*\gg\,^1\pi\pi^* \to\, ^3\pi\pi^*,\qquad\qquad ^1n\pi^* \to \,^3\pi\pi^* \gg\, ^1n\pi^* \to\, ^3n\pi^*$$
# 
# The comparison is best observed in N-heterocyclics such as pyrazine and quinoline and carbonyl compounds such as benzaldehyde, which have both $n\pi$ and $\pi\pi$ excited states. 
# 
# The spin orbit operator belongs to the same irreducible representation as $R_k,k={x,y,z}$ which means for an allowed transition the product $\Gamma_S\otimes R_k\otimes\Gamma_T$ must contain the totally symmetric representation, $A_1,A_g$ etc. for any $x,y,z$.  When singlet and triplet have the same configuration then $\Gamma_S=\Gamma_T$ and thus for a transition $S\to T $ to be allowed the spin-orbit operator $R_k$ must belong to the totally symmetric representation. Examining the point group tables shows that molecules in $C_{nV}, D_{nh} D_{nd}$ and a few other point groups do not have a $R_k$ operator that is totally symmetric so $S\to T$ transitions cannot occur by spin-orbit coupling in molecule of these point groups, i.e. are formally forbidden and hence occur very slowly. It is, of course, possible that $^1\pi\pi^*$ and $^3\pi\pi^*$ belong to different symmetry species and then the $S\to T$ transition may be allowed if there is a suitable symmetry species for $R_k$, but in practice the rate constant is still small compared to $\pi\pi^* \rightleftharpoons n\pi^*$ transitions. The original papers are by M.A. El-Sayed J. Chem. Phys. 36, p 573 - 74, 1962 and J. Chem. Phys. 38, p 2834 - 38,1963.
# 
# ### **(v) Vibronic transitions and Herzberg - Teller coupling**
# When a vibration is involved in an electronic transition the transition is called *vibronic*. The transition moment changes a little to reflect this and becomes
# 
# $$\displaystyle R_{ev} =  \int \psi^a_e\psi_v^a\,\vec{\mu}\,\psi^b_e\psi_v^b d\tau  \qquad\qquad\tag{9.25}$$
# 
#  Normally the vibrational part is separated out as by the Born Oppenheimer principle the electrons move far more rapidly than do the nuclei.
# 
# $$\displaystyle R_{ev} =  \int \psi^a_e\vec{\mu}\psi^b_ed\tau\int \psi_v^a\psi_v^b dq=R_e\int \psi_v^a\psi_v^b dq\qquad\qquad\tag{9.26}$$
# 
# where $R_e$ is the electronic term. The square of the absolute value of the vibrational integral is called the _Franck-Condon_ factor, i.e. $\displaystyle \big |\int \psi_v^a\psi_v^b dq\,\big |^2$.
# 
# Molecular vibrations can distort the molecule making forbidden transitions slightly allowed and the intensity for the transition will be borrowed from a nearby allowed transition if such exists. This *intensity stealing* or *Herzberg - Teller coupling* gives rise to the vibrational features in the absorption spectrum of benzene in particular (Steinfeld 1981; Atkins & Friedmann 1997) but is general. In benzene the transition from the ground state to the first excited state $^1S_0(A_{1g})\to\, ^1S_1(B_{2u})$ is forbidden but this transition becomes weakly allowed because a vibration of $e_g$ symmetry species (in the $D_{6h}$ point group) can change the ground or excited state geometry. This forms the new symmetry of the ground state and the method outlined above is used to determine if the transition is allowed. The intensity of the transition is borrowed, or 'stolen' from an allowed transition nearby in energy and with the same symmetry as the vibrationally modified state. If this electronic state is present the symmetry may be suitable but the transition intensity can still be minute if in the energy gap to the allowed transition is large. 
# 
# Figure 21d (left) shows the low resolution spectrum of benzene vapour and the forbidden transition $v_{00}$ is indicated where it would appear.  The lines labelled 'A' in the spectrum are transitions to the lowest singlet state with $B_{2u}$ symmetry plus $1,2\cdots$ quanta of the $v_1=a_{1g}$ plus one $v_6=e_{2g}$ vibration, i.e. $nv_1+v_6$. The line labelled $B_0^0$ at $808\,\mathrm{cm^{-1}}$ to lower energy than the origin starts from $v=1$ in the ground state to $v=0$ in the lowest ($B_{2u}$) singlet so is a hot band and its intensity is sensitive to temperature. All these lines appear because of the change in geometry, i.e. the intensity stealing. On the right of the figure are shown the ground and excited states energies (not to scale and without vibrations) along with their symmetry labels.
# 
# ![Drawing](matrices-fig21a.png)
# 
# Figure 21d. The vapour phase absorption spectrum where the missing $0-0$ transition is labelled $v_{00}$ and is at $38086.1\,\mathrm{cm^{-1}}$.  The transitions 'A' are to different vibrational levels in the excited state with increasing quanta $n=0,1,2\cdots$ of $nv_1$ plus an $e_{2g}$ vibration. (Spectrum redrawn from J. Callomon, T. Dunn, I. Mills, Phil. Trans. Roy. Soc. (Lond) v259, p499, 1966). The relative energy of the ground and excited states is sketched on the right and is not to scale and without vibrations.
# _________
# Benzene belongs to the $D_{6h}$ point group, and looking at this (see www.molecule-viewer.com  for tables) the translations needed for dipole transitions are  $z\to A_{2u}, (x,y)\to E_{1u}$ and $x,y$ are in the plane of the molecule. The triple products, ignoring any vibrations for the moment are, 
# 
# $$\displaystyle \begin{align}\Gamma_{S_0}  \otimes \Gamma_{\mu_z}  \otimes \Gamma_{S_1} &= A_{1g}  \otimes A_{2u}  \otimes B_{2u} = B_{2g}\qquad \mu_z \\ \Gamma_{S_0}  \otimes \Gamma_{\mu_{x,y}}  \otimes \Gamma_{S_1} &= A_{1g}  \otimes E_{1u}  \otimes B_{2u} = E_{2g}\qquad \mu_x, \mu_y \end{align}$$
# 
# and the transition is therefore forbidden for all transition dipole directions to the first excited state of $B_{2u}$ symmetry species. To work out the product either multiply the characters in the symmetry species, make a list and find the symmetry species that this corresponds to by inspection of the point group table, or look it up in a product table. If we suppose now that a vibration is used to change the symmetry of the state, either ground or excited we need the symmetry product to be totally symmetric, ($A_{1g}$), thus 
# 
# $$\displaystyle \Gamma_{S_0}  \otimes \Gamma_{vib}\otimes \Gamma_{\mu_{x,y}} \otimes \Gamma_{S_1} \equiv \Gamma_{vib}  \otimes E_{1u} \otimes B_{2u}=A_{1g}$$
# 
# and if we choose $\Gamma_{vib}=E_{2g}$ this becomes the case because as we have just seen $E_{1u}  \otimes B_{2u} = E_{2g}$ and $E_{2g} \otimes E_{2g}$ contains $A_{1g}$ as any symmetry species multiplied by itself always contains the totally symmetric representation.  
# 
# To gain some insight into why the vibration induces a transition we can expand the electronic part of the transition dipole $R_{ev}$ eqn. 9.26, as a Taylor series in a vibrational coordinate, say, $q$ and about the equilibrium position (superscript 0) to give
# 
# $$\displaystyle R_e= R_{e}^0 + \sum_i\left( \frac{\partial R_e^0}{\partial q_i}\right)_0 q_i\qquad\qquad\tag{9.27}$$
# 
# Substituting back into eqn 9.26, since now we cannot assume that the Born-Oppenheimer condition applies, gives, after rearranging,
# 
# $$\displaystyle R_{ev}=R_e^0\int \psi^a_v\psi^b_v d\tau+\sum_i\left( \frac{\partial R_e^0}{\partial q_i}\right)_0\int \psi_v^a q_i\psi_v^bdQ_i\qquad\qquad\tag{9.28}$$
# 
# The first term is the same as eqn. 9.26 when Born-Oppenheimer applies and in benzene is zero by symmetry, but the second term is not and therefore some intensity is observed. Exactly how much depends on the slope of the electronic transition dipole with each vibrational mode, i.e. how much the molecular orbital's symmetry is changed by a vibration and therefore a totally symmetric vibration cannot help break the symmetry. Note how similar this equation is in general form to that for a diatomic molecule.  See J. M. Hollas, 'Modern Spectroscopy' 4th ed. publ. Wiley 2008 for a fuller description of intensity stealing.

# ## 6.13 Reducible Representations
# 
# When, for instance, the orbitals in a molecule are operated on with rotations or reflections, a reducible representation $\Gamma_R$ is produced. This can be decomposed into some irreducible representations, which are the rows that appear in the character table. The effect that symmetry operations have, were calculated in Section 6.8 by setting up a matrix for each atom/orbital and working out the effect of each operation. A matrix has the following general form with unit Cartesian vectors, $x_1, y_1, z_2$ and so forth on each atom.
# 
# $$\displaystyle C_2\begin{bmatrix} x_1\\ y_1\\z_1\\x_2\\y_2\\ \vdots \end{bmatrix}=\begin{bmatrix} 0&0&-1&-\cdots\\ 0&1\\0& & \ddots\\\vdots\\1\\ \vdots \end{bmatrix}\begin{bmatrix} x_1\\ y_1\\z_1\\x_2\\y_2\\ \vdots \end{bmatrix}$$
# 
# The dimensions of the matrix depend on the number of basis vectors used to describe how the orbitals or vibrations change. The trace of each matrix is then calculated for each class of symmetry operation and then collected together to form the reducible representation. However, such an elaborate approach is not necessary, and the trace can be found more easily by following a small set of rules and so obtaining the reduced representation directly.
# 
# First, the atom positions or orbitals are drawn and labelled, then the molecule is subject to each of the symmetry operations of the point group in turn. A table is started and the first row is the list of symmetry operations. The second row is filled in according to four rules for each symmetry operation operating on each base vector.
# 
# **(a)**$\qquad$  If unchanged, a value of $1$ is entered in the table.
# 
# **(b)**$\qquad$ If changed in sign, $-1$ is entered.
# 
# **(c)**$\qquad$If moved in position, $0$ is entered.
# 
# **(d)**$\qquad$ Add up all the numbers and enter the result under the symmetry operations column.
# 
# Consider now the p orbitals on SO$_2$, as shown in figure 22, we will use these as a basis for the calculation. They could also be envisaged as unit vectors pointing up from the atoms. The figure tries to show a perspective view; the principal axis, $z$, is equally placed between atoms 1 and 3 and through atom 2. The molecule has $C_{2V}$ symmetry and the point group, which is shown above, has four operations $E, C_2$ and two reflections $\sigma (xz)$ and $\sigma'(yz)$.
# 
# Operating on the molecule according to the rules (i) to (iii) produces the reduced representation $\Gamma_R$.
# 
# $$\displaystyle 
# \begin{array}{c|ccc}
# C_{2V} & E & C_2 & \sigma (x,z) & \sigma '(y,z)\\
# \hline 
# \Gamma_R &
# 3 & -1 & -3 & 1\\
# \hline \end{array}$$
# 
# The $3$ in the identity operator column is produced because each orbital is unchanged with this operator. Rotation about the $C_2$ or z-axis moves vectors on orbitals $1$ and $3$, so they count zero, and inverts orbital $2$; therefore $-1$ results. The reflections are calculated similarly.
# 
# ![Drawing](matrices-fig22.png)
# 
# Figure 22. p-orbitals in $C_{2V}$ symmetry. The shading indicated $\pm$ phases and are used to identify changes under the symmetry operations. Instead of orbitals vectors on each atom pointing in the $y$ direction could be used.
# _____
# 
# ### **(i) The Tabular Method**
# 
# To reduce the representation, a tabular method described by Carter (1997), is very convenient and simple to use although the following formula can alternatively be used. 
# 
# $$\displaystyle N_M=\frac{1}{h}\sum_ig_i\chi^{red}(c_i)\chi^*(c_i)\qquad\qquad\tag{9.29}$$
# 
# where $N_M$ is the number of times the irreducible representation for symmetry species with Mulliken label $M$ is present in the reducible representation, $h$ is the total number of operations, $g_i$ the number of operations in class $i$, $\chi^{red}(c_i)$ the $i^{th}$ character in the row of characters in the *reducible* representation with character $c_i$ and $\chi^*(c_i)$ that for symmetry species, row  $M$, in the  point group. The superscript $*$ represents the complex conjugate of character $c_i$ and is used should $c_i$ be a complex number.
# 
# The tabular method starts with the reducible representation table just produced, and multiplies each term it contains by the corresponding element in the character table for each of the symmetry species, and then by the number of operations in each class. One product is shown in detail in the table. The number of operations in a class is shown in the heading of each column in the point group; see figure 15. In the $C_{3V}$ point group this is two for the $C_3$ operation. The sum of each row is made and divided by the total number of symmetry operations $h$.
# 
# The $C_{2V}$ point group only has one operation in each class, and the number of symmetry operations $h = 4$. The table produced for the reduced representation of the $C_{2V}$ molecule is,
# 
# $$\displaystyle \begin{array}{l|rrcr|l}
# C_{2V} & E & C_2 & \bbox[5px,border:2px solid red] 1\sigma(xy) & 1\sigma'(yz) & \displaystyle \text{sum}/h \\
# \hline
# \Gamma_R & 3 & -1 & \bbox[5px,border:2px solid blue]{-3} &1 & \\
# \hline
# A_1 & 3 & -1 &-3 &1 &0/4=0\\
# A_2 & 3 & -1 &(-1)\times \bbox[5px,border:2px solid blue]{-3}\times\bbox[5px,border:2px solid red] 1=3 & -1 &4/4=1\\
# B_1 & 3 & 1 &-3 &-1 &0/4=0\\
# B_1 & 3 & 1 &3 &1 &8/4=2\\
# \hline
# \end{array} $$
# 
# The $-1$ in the centre under $\sigma(xy)$ is the character from the $C_{2V}$ point group and in the top row $1$ is unusually placed in front of the $\sigma$ to show where the value comes from in the product. The ratio of the sum and $h$ must always be an integer. If not, a mistake has been made. This reduced representation of the p orbitals in SO$_2$ is therefore composed of an $A_1$ and two $B_2$ irreducible representations; $\Gamma_R \equiv A_1 + 2B_2$. We therefore expect molecular orbitals based on p orbitals to have these symmetries.
# 
# In the $C_{3V}$ character table, figure 15, and relevant parts shown below, the electric dipole transition from $A_1$ to $E$ states is potentially allowed under $x-$ or $y-$ polarized transitions, because the $x$ and $y$ operators belong to the $E$ symmetry species. However the product $A_1 \otimes E \otimes E = E \otimes E$ is not an irreducible representation because the direct product is $E = 4,\, C_3 = 1, \,\sigma_V = 0$. 
# 
# $$\displaystyle \begin{array}{l|rrr|l}
#  C_{3V} & E & 2C_3 & 3\sigma_V  &  \\
# \hline
# A_1 & 1 & 1  &1 &z \\
# A_2 & 1 & 1  &-1 & \\
# E   & 2 & -1 &0 &(x,y) \\
# \hline
# \end{array} $$
# 
# Reducing the $E \otimes E$ direct product is done as follows using the tabular method. Each entry in the table is the number from the reduced representation multiplied by the character and then multiplied by the number in each class. The total number of classes $h = 6$ and the $E \otimes E$ table is
# 
# $$\displaystyle \begin{array}{l|rrr|l}
#   & E & 2C_3 & 3\sigma_V  & \displaystyle \text{sum}/h \\
# \hline
# \Gamma_R & 4 & 1 & 0 & \\
# \hline
# A_1 & 4 & 2  &0  & 6/6=1\\
# A_2 & 4 & 2  &0  & 6/6=1\\
# E   & 8 & -2 &0  & 6/6=1\\
# \hline
# \end{array} $$
# 
# Therefore the product $A_1 \otimes E \otimes E \equiv A_1 \otimes(A_1+A_2+E)=A_1+A_2+E$. The product $E\otimes E$ therefore contains one of each of the other species in the point group so that the transition would be allowed because $A_1$ is present. If the states were both of $E$ symmetry species and the transition dipole also $E$ the product would be $E\otimes E\otimes E$ and then the result of reducing the reducible representation would be $3E+A_1+A_2$ and the transition allowed. In this particular case we could simply have used the previous result and made $E\otimes(E+A_1+A_2)$ and using the characters in the point group found $E\otimes A_1=E$ etc, and so $E+A_1+A_2+E+E$.

# ## 6.14 Normal Mode vibrations
# 
# A normal mode is one of a set of the elementary vibrations of any vibrating object and is fundamental to understanding vibrations of any kind, whether in the engine or suspension of a car, in a washing machine, in a guitar string or in a molecule. If you were able to see a molecule, other than a diatomic, and watch the motion of the atoms, they would appear to be moving rather chaotically. However, to a good approximation it is always possible to assume that the complicated vibrations of any body can be broken down into a set of normal modes and then each of the individual stretching and bending normal motions of a molecule would be apparent. Rather than concentrating on the motion of one atom at a time, the motion of several together is considered and if we get this right, the motion will be that of one of the normal modes. This would, in effect, be a transformation from viewing the molecule in lab $xyz$ coordinates, to viewing internal coordinates based on the motions of these several atoms and there is, fortunately, a systematic way of finding them. In any molecule, all the atoms are coupled through their bonds and in a normal mode, all the atoms move with the same frequency, have fixed amplitude ratios and fixed phase relationships between them. Unless the normal mode is, by symmetry, degenerate with another, each normal mode has a unique frequency.
# 
# The symmetry of a normal mode is described by one of the irreducible representations in the molecule's point group. The fact that a symmetry label can be attached to a normal mode encapsulates the idea of collective synchronized motion; some examples are given below for a planar molecule such as HBF$_2$. Notice that it is possible to have different normal mode displacements, and consequently vibrational frequencies, but with the same
# overall symmetry.
# 
# The first normal mode shown in figure 23 belongs to symmetry species $A_1$ and each atom is moved in such a way as to stretch each bond in phase with the other. If the molecule were HBF$_2$, the centre of mass is below the atom labelled $x$ on a line from $z \to x$. In the bottom right of the figure, the $B_1$ mode has atoms moving together, but the central atom moves into the plane of the figure while the other three atoms move out, and vice versa.
# 
# No external force can act on a molecule due solely to its vibrational motion; therefore, no extra displacement or rotation of the molecule can occur during normal mode vibrations and the centre of mass must remain in the same place. If there are N masses (atoms) connected by forces (chemical bonds), then there are $3N$ modes in a three-dimensional object. However, because we want vibrational normal modes, we have to remove three modes due to translation and three for rotation (two if the molecule is linear) as we are not interested in the whole body's rotation or translation. This leaves $3N - 6$ vibrational normal modes for polyatomic molecules and $3N - 5$ for linear ones.
# 
# ![Drawing](matrices-fig23.png)
# 
# Figure 23 Normal mode displacements for a planar molecule ZXY$_2$ (based on a figure in Appendix C, Carter 1997). The arrows indicate the direction the atoms move in, but not relatively how far they move; $\pm$ show motion out of the plane of the paper. Typical bond displacements at room temperature are 1% of the bond length. The symmetry labels are also shown and each mode has its own unique frequency $\nu_1,\, \nu_2$ etc.
# _____
# 
# ## 6.15 Application to molecular vibrations
# 
# Group theory plays an important part in working out what spectroscopic transitions can occur in a molecule, and this was outlined in Section 7.6.12. In addition, it can be used to work out what the normal mode symmetry species are going to be. Recall that a normal mode is the collective motion of the atoms in a molecule such that a constant phase relationship is maintained between them; see Figs 27, 78, and 79. Often these are characterized as symmetric stretch, asymmetric stretch, and bending vibrations, which is fine in a small molecule, such as SO$_2$ or water. However, in larger molecule the vibrational motion becomes more complex and cannot be so simply described, but is instead characterized as a symmetry species of the molecule. The normal mode vibrations are given by the Mulliken labels listed in the left-hand column of the point group but usually with a lower case letter, e.g. $b_{2g}$ rather than $B_{2g}$.
# 
# The method outlined for working out orbital symmetry species can be adapted for use here. Again, a table is made up with a top row as the symmetry operations and entries filled in according to rules very similar for those for orbitals.
# 
# **(a)** Choose, either a set of three orthogonal vectors on each atom, or a single vector along each bond to be the basis set for the calculation. Which basis you choose will depend on the problem. The components of these vectors are added up to form the characters needed. This is be done with a set of rules.
# 
# **(b)** Move the molecule according to the symmetry operations in the point group.
# 
# **(c)** For a mirror plane and inversion, add 1 to the character under the symmetry operation for each vector unchanged and $-1$ for each vector inverted and zero for each atom moved. Do the same for rotations if by $90^\text{o}$ or $180^\text{o}$, i.e. for $C_4$ and $C^2$ axes.
# 
# **(d)** For other rotations $C_3$ for example,(or rotation-inversion) a rotation matrix has to be used to calculate the components of the character because the operation may mix the $x, y$ and $z$ components. The trace of the (rotation) matrix gives the character needed.
# 
# **(e)** Reduce the representation produced. If the three orthogonal vectors on each atom were used as the basis set, remove the symmetry species due to three translations, i.e. movement in space, and three rotations from the final list as these are implied by the orthogonal vectors. (The three rotations correspond to whole body rotation and not any internal molecular motion.)
# 
# ![Drawing](matrices-fig24.png)
# 
# Figure 24 Unit vectors with which to work out vibrational symmetry species. The axes are rotated slightly wrt. the molecule, which is in the $x-z$ plane so that the $y$ direction can be seen. 
# _____
# 
# The case of a $C_{2V}$ molecule such as water is easy to follow; the operators are $E, C_2$ and $\sigma V(x,z)$ and $\sigma'V(y,z)$. There are nine vectors, so the identity has a character $E = 9$. The effect of the $180^\text{o}$ rotation is shown in Figure 24. The vectors add up to zero for the H atoms as they move, $+1$ for the $z_1$ vector, and $-1$ for both the $x_1$ and $y_1$ producing $-1$ overall, which is the character for the reducible representation for the $C_2$ operator. The two reflections are worked out similarly to rotation. Reflection in the $x-z$ plane leaves $x$ and $z$ unchanged making six in total but each $y$ is inverted therefore the character for the $\sigma(x,z)$ is $3$. The character for $\sigma'(y,z)$ is one because the H atoms move under this operation so count zero and only $x_1$ becomes $-1$ and $z_1$ and $y_1$ are unchanged.
# 
# The reducible representation is therefore
# 
# $$\displaystyle \begin{array}{l|ccc}
# C_{2V} & E &C_2 & \sigma(xz) & \sigma'(yz) \\
# \hline
# \Gamma_R & 9 & -1 & 3 & 1\\
# \hline
# \end{array}$$
# 
# This table reduces to
# 
# $$\displaystyle \begin{array}{l|rrrr|r}
# C_{2V} & E &C_2 & \sigma(xz) & \sigma'(yz) & \text{sum}/h \\
# \hline
# \Gamma_R & 9 & -1 & 3 & 1 &\\
# \hline
# A_1 & 9 & -1 & 3 & 1 & 12/4=3\\
# A_2 & 9 & -1 & -3 & -1 & 4/4=1\\
# B_1 & 9 & 1 & 3 & -1 & 12/4=3\\
# B_2 & 9 & 1 &- 3 & 1 & 8/4=2\\
# \hline
# \end{array}$$
# 
# The result is $\Gamma_R =3A_1 +A_2 +3B_1 +2B_2$. There are $3N=9$ modes in total,$3N-6$, thre eof which are vibrations, the others displacements and rotations. These transform as $x, y$ and $z$ and $R_{x,y,z}$ and amount to $A_1 + A_2 + 2B_1 + 2B_2$ leaving $2A_1 + B_1$ as the vibrations. These species correspond to one totally symmetrical stretch and bend, and an asymmetrical stretch, see Figure 26.
# 
# In CHCl$_3$ if we place vectors on each atom this makes a $15$-dimensional basis set and the $C_3$ axis is then going to mix the $x$ and $y$ components if $z$ is along the principal axis, figure 25. Chlorine atom 1 is on the y-axis and the carbon is at the origin. The symmetry operations in $C_{3V}$ are $E, 2C_3$, and $3\sigma_V$. The reducible representation is again found by applying the operators in turn to each vector. The identity $E$ scores $15$, because each vector is unchanged. The mirror planes run along the principal axis and between each pair of Cl atoms. The mirror plane between Cl atoms 2 and 3 also runs along the H-C-Cl bonds containing Cl atom 1. The Cl-2 and Cl-3 atoms are moved by reflection so count zero for all vectors. The $y$ and $z$ coordinates on the C, Cl(1) and C atoms are unchanged and count one each, making six in total. The $x$-direction vectors on these atoms are inverted and count $-1$ each. Do this for one mirror plane only the effect of the number of classes is accounted for when the representation is reduced. The reflections thus count $6 - 3 = 3$.
# 
# The rotations are a little more complicated, but it is clear that the Cl atoms are each shifted by rotation and their vectors count zero. The H and C atoms each have their $x-$ and $y-$directions mixed. The matrices to rotate a molecule are explained in Section 7, and we use this result. The new positions are $x_\theta$ etc. after rotation by an angle $\theta$.
# 
# $$\displaystyle \begin{bmatrix} x_\theta\\ y_\theta\\z_\theta \end{bmatrix}=\begin{bmatrix} \cos(\theta) & \sin(\theta) & 0\\ -\sin(\theta) & \cos(\theta) & 0\\0&0&1 \end{bmatrix}\begin{bmatrix} x\\ y\\z \end{bmatrix} $$
# 
# If the angle is $90^\text{o} \equiv \pi/2$ radians then rotation moves $y$ into $x$ and $x \to -y$ and $z$ is unchanged. The matrix is 
# 
# $$\displaystyle \begin{bmatrix} 0 & 1 & 0\\ -1 & 0 & 0\\0&0&1 \end{bmatrix}$$
# 
# Adding up the components in $x, y$ and $z$ produces $+1$ and this is what is counted. This is the origin of the rule for rotations by right angles. However, it is entirely equivalent and simpler to calculate the trace of the matrix, which is one. The trace is invariant of the basis set, if these cover the same function space, so this can always be used.
# 
# In our molecule the rotation angle $\theta$ is $120^\text{o} \equiv 2\pi/3$ and the matrix is 
# 
# $$\displaystyle \begin{bmatrix} -1/2 & \sqrt{3}/2 & 0\\ -\sqrt{3}/2 & -1/2 & 0\\0&0&1 \end{bmatrix}$$
# 
# which has a trace of zero. Thus the $C_3$ rotation contributes zero to the characters. If you multiply the matrix with the $x, y, z$ vectors, the sum of the all the product vectors' components is also zero. The sum is $-1/2 + \sqrt{3}/2 + 0 - \sqrt{3}/2 - 1/2 + 0 + 0 + 0 + 1 = 0$. The matrix calculation need only be done for one rotation in each class, i.e. we do not need to do both $C_3^+$ and $C_3^-$ as this is taken into account in reducing the representation. The result of the calculation is
# 
# $$\displaystyle \begin{array}{l|ccc}
# C_{3V} & E & 2C_3 & 3\sigma_V \\
# \hline
# \Gamma_R & 15 & 0 & 3 \\
# \hline
# \end{array}$$
# 
# This is reduced using the group table figure 15 to give
# 
# $$\displaystyle \begin{array}{l|ccc|cc}
# C_{3V} & E & 2C_3 & 3\sigma_V & \text{sum} & \text{sum}/6\\
# \hline
# \Gamma_R & 15 & 0 & 3 \\
# \hline
# A_1 & 15 & 0 & 9 & 24 & 4\\
# A_2 & 15 & 0 & -9 & 6 & 1\\
# E   & 30 & 0 & 0 & 30 & 5\\
# \hline
# \end{array}$$
# 
# The reduced representation of CHCl$_3$ vibrations in the Cartesian basis set is therefore composed of $\Gamma_R = 4A_1 + A_2 + 5E$ making $15$ species in total. There are six degrees of freedom, three for translations and three for rotations of the molecule to be subtracted. These add to $A_1 +A_2+4E$ (see $x,y,z$ and $R_x,R_y,R_z$  in the point group table), which leaves $3A_1+3E$ as the vibrational normal mode symmetry species. The number of normal modes remaining is $3N - 6 = 9$, which is $3A_1$ singly and three doubly degenerate $E$ vibrations. The shapes of the normal modes for CHCl$_3$ are illustrated in Herzberg (1964, vol II ), and in figure 25a.
# 
# 
# Rotation axes $C_n$ are numbered as fractions of $360^\text{o}$ or $2\pi/n$ making angle in the rotation matrix $\theta = 2\pi/n$ and the trace of the rotation matrix for rotations with different $n$ is
# $T_{C_n}= 2\cos(2\pi/n) + 1$ . The trace for the first few $C_n$ is
# 
# $$\displaystyle \begin{array}{ccccc} 
# C_2 & C_3 & C_4 & C_5 & C_6 \\
# \hline
# -1 & 0 & 1 &\displaystyle \frac{1+\sqrt{5}}{2} & 2 \\
# \hline
# \end{array} $$
# 
# A similar calculation for rotation-reflection ($S_n$) produces $T_{S_n}=-1+2\cos(2\pi/n)$, see question 26.
# 
# ![Drawing](matrices-fig25.png)
# 
# Figure 25 CHCl$_3$ with orthogonal Cartesian sets of unit vectors. The middle image shows the view along the $z$ axis.
# ______
# 
# ![Drawing](matrices-fig24a.png)
# 
# Figure 25a. Normal modes in CHCl$_3$. The arrows give the direction of simultaneous, in phase, motion of the atoms. Based on figure 91,  Herzberg (1964, vol II,  'Infrared and Raman Spectra' publ van Nostrand 1964.)
# ______

# ## 6.16 Determining the shapes of Normal Mode vibrations and molecular orbitals using Projection Operators
# 
# Having obtained the symmetry species of vibrations or molecular orbitals it is natural to want to see what these look like and to do this projection operators are used. What these operators do is to extract the symmetry adapted functions from the basis functions used to form them and so produce a linear combination L of the basis functions. Many basis functions can be chosen, but the easiest to use are usually vectors pointing along bonds or those representing a $p\pi$ orbital. To form a linear combination, a 'victim' vector is chosen and operated on with each symmetry operation in the point group, $C_2, \sigma(xz)$, etc. to find what vector this turns into. This vector's name (the basis function) is then multiplied with the character from the point group corresponding to that symmetry operation and symmetry species. Finally, the names are added together by moving from one symmetry operation to the next along the top row of the point group. We expect the result to be the sum or difference of the displacement vectors for a molecular vibration, such as $2v_1 + v_2 - v_3 \cdots$ and so forth. The same formula or algorithm is used if the combinations of atomic orbitals that make up molecular orbitals are sought.
# 
# Equation 9.21 in section 9.6.5 shows how to generate the projection operator. In the $C_{3V}$ point group, for example, there is a heading $2C_3$ as there are two members in this class, and this must be split in the summation into $C_3^+$ and $C_3^-$. This is because moving a vector $120^ \text{o}$ to the right, say, will turn it into a different vector than turning $120^\text{o}$ to the left, similarly, any mirror planes must be separated out. The character for symmetry species $M$ is $c_j$ and $O^S_j(v)$ is the effect that operator $O^S$ in column $j$ of the point group has on our victim vector $v$. 
# 
# The result of this operation is to produce a vector whose _name_ is recorded. For example, if $O^S$ is the identity then the result is $E_1(v) = v$, other operations may leave $v$ unchanged or change it into another vector. Finally, note that if the reduced representation of the vibrations or molecular orbitals contains two or more symmetry species of the same type, e.g. $2A_1$, then two or more different victim vectors will have to be chosen to obtain all the linear combinations.
# 
# In H$_2$O, SO$_2$, and other triatomics with $C_{2V}$ symmetry, it is clear that two of the vibrational modes stretch the bonds and one changes the bond angle. To work out the normal mode vectors, some representative vectors are placed along the bonds as in figure 26. You can choose where to put these but physically realistic choices will usually make the calculation simpler.
# 
# ![Drawing](matrices-fig26.png)
# 
# Figure 26 Left: Vectors which are the basis functions used to determine normal modes. Middle:  The asymmetric normal mode $b_1$ as $v_1 - v_2$. Right, the two symmetric $a_1$ modes.
# ______
# 
# In $C_{2V}$, the normal mode vibrations have been found to comprise the symmetry species $2A_1$ and $B_1$. The $L_M$ formula is shown as a table with $v_1$ as the victim vector. The table also shows the characters for each symmetry species.
# 
# $$\displaystyle \begin{array}{l|rrrr|c}
# C_{2V} & E &C_2 & \sigma(xz) & \sigma'(yz) & L_m (d=4,h=4) \\
# \hline
# A_1 & 1 & 1 & 1 & 1 &  \\
# L_{A_1} & v_1 & v_2 & v_1 & v_2  &2(v_1+v_2)\\
# \hline
# A_2 & 1 & 1 & -1 & -1 & \\
# L_{A_2} & v_1 & v_2 & -v_1 & -v_2 & 0\\
# \hline
# B_1 & 1 & -1 & 1 & -1 & \\
# L_{B_1} & v_1 & -v_2 & v_1 & -v_2 & 2(v_1-v_2)\\
# \hline
# B_2 & 1 & -1 & -1 & 1 & \\
# L_{B_2} & v_1 & -v_2 & -v_1 & v_2 & 0\\
# \hline
# \end{array}$$
# 
# Notice that what the vector changes into is (obviously) the same for each symmetry species as this is determined by the symmetry operations. Only the value of the character changes in front of each term The final $L_M$ is not multiplied by $d/h$ because the resulting vectors are instead normalized giving, $L_{A_1} = (v_1 + v_2)/\sqrt{2}$ and $L_{B_1} = (v_1 - v_2)/\sqrt{2}$. Note also that $v_3$ and $v_4$ do not enter into this table, they are at $90^\text{o}$ to $v_1$ and $v_2$ and no operation in this point group can inter-convert them.
# 
# There are two $A_1$ species produced from the reducible representation but only one has been found so far. This is because the vectors $v_1$ and $v_2$ cannot produce a bend vibration, which is the other $A_1$ normal mode. Using vector $v_3$ (figure 26) as the victim this species appears. As this mode has $A_1$ symmetry all the characters $c_j$ in the $L_M$ formula are one, making the calculation easy and the combination is shown below.
# 
# $$\displaystyle \begin{array}{l|rrrr|c}
# C_{2V} & E &C_2 & \sigma(xz) & \sigma'(yz) & L_m (d=4,h=4) \\
# \hline
# A_1 & 1 & 1 & 1 & 1 &  \\
# L_{A_1} & v_3 & v_4 & v_3 & v_4  &2(v_3+v_4)\\
# \hline
# \end{array}$$
# 
# which describes the bending mode (top right in figure 26) in terms of its unit vectors. When normalized this mode is $(v_3 + v_4)/\sqrt{2}$.
# 
# As the centre of mass cannot move during a vibration, there being no force to make it do so, therefore the O atom has to move a small distance away from the H atoms. Vectors could be added onto the O atom to show this and $L_M$ terms re-calculated. The atoms' displacements can also be found when the equations of motion are solved as shown in Section 14 and in figure 25a.
# 
# In the case of degenerate symmetry species as occurs in the $C_{3V}$ and most other point groups, the result of the summation $L_M$ is to produce two or more vector sums that are not orthogonal. Either these have to be made orthogonal by choosing another victim for the degenerate term and then using the Gram - Schmidt method to make the two vectors orthogonal, or, using the method outlined by Carter (1997) and by Vincent (2001), which involves producing a second vector at $90^\text{o}$ to the victim one and repeating the calculation. (The Gram - Schmidt method is described by Arkfen 1970 and many other textbooks).
# 
# The calculation to produce molecular orbitals is essentially the same as for vibrations if the basis vectors are made 'parallel' to the atomic orbitals. For example, for aromatic or other conjugated molecules, a vector pointing in the direction of each $\pi$ atomic orbital is used as a basis. 
# 
# Once the symmetry of the molecular orbitals have been found, a secular equation can be set up to calculate their energies. The matrix elements (or expectation value) will have the form $\langle \psi_1|H|\psi_2\rangle$ between two molecular orbitals labelled 1 and 2. If more than one orbital is present with a particular symmetry, the orbitals will mix and a secular equation is set up to find their energies, provided that the orbitals are made orthogonal first. This can be done with a Gram - Schmidt method and the results normalized. The molecular orbitals each have the form $\psi = a_1\varphi_1 + a_2\varphi_2 + b_3\varphi_3 + \cdots + b_n\varphi_n$ where $\varphi$ are the atomic orbitals and a are the coefficients found using the projection operator method just described. We can replace the orbital by the base vectors t and still work out the coefficients. If for example the Huckel Hamiltonian is used, this operates only on adjacent orbitals and is not zero only for terms with $k = j$ and $j \pm 1$ when it produces a constant value. However, whatever the Hamiltonian, an expectation value is calculated with the rather complicated looking formula
# 
# $$\displaystyle \langle \psi_1|H|\psi_2\rangle =\sum_{k=1}^n\sum_{j=1}^n a_{1k}a_{2k}\langle v_k|H|v_j\rangle$$
# 
# however, many of the terms may be zero, either because of the type of Hamiltonian, e.g. Huckel, or because one of the coefficients $a_{1k}$ or $a_{2j}$ is zero. The expectation values are placed in a secular determinant,
# 
# $$\displaystyle \begin{bmatrix}
# \langle 1|H|1\rangle &\langle 1|H|2\rangle &\cdots \\
# \langle 2|H|1\rangle &\langle 2|H|2\rangle &  \\
# \vdots & \ddots &\\ 
# & & \langle n|H|n\rangle \end{bmatrix}$$
# 
# which is solved to find the $n$ eigenvalues, which are the energies, and the eigenvectors, see section 12.5.

# ## 6.17 Using symmetry to simplify a Huckel MO Calculation
# 
# The Huckel method of calculating energy levels when there are delocalised $p_\pi$ orbitals has been described in section 3 of this chapter where the secular equation was solved directly by finding the roots of a polynomial using symbolic algebra and additionally for cyclic molecules, such as benzene, using circulant matrices. 
# 
# One practical difficulty in these calculations that the secular equations are often very hard to solve, finding the roots of a $6^{th}$ order polynomial is not trivial and so computer algebra is used. In this section using symmetry to reduce the complexity of the calculation is described. There are several steps involved in doing this, but each is mathematically simple. The final secular equation will also be simple to solve.  
# 
# The secular determinant is 
# 
# $$\displaystyle |H_{ij}-ES_{ij}|=0 \qquad \tag{9.30}$$
# 
# and evaluating this produces a polynomial whose roots are the energy levels $E$. At matrix position $ij$ the matrix element $H_{ij}$ is the integral 
# 
# $$\displaystyle H_{ij}=\int f_i^*Hf_j d\tau\qquad\qquad\tag{9.31}$$
# 
# where $H$ is the Hamiltonian, $f$ an atomic $p_\pi$ orbital. The overlap integral is 
# 
# $$\displaystyle S_{ij}=\int f_i^*f_j d\tau\qquad\qquad\tag{9.32}$$
# 
# The molecular orbitals $\phi_i$ are defined as a linear combination of the molecules $f\equiv p_{\pi}$ orbitals, 
# 
# $$\displaystyle \phi_i'=\sum_{j=1}^n c_{ji}f_j\qquad\tag{9.33}$$ 
# 
# This is the general method, and as no information about the symmetry of the molecule is incorporated  in eqn 9.33 this is a 'brute force' approach. If we use information from the molecule's symmetry we can simplify the secular equation, even to make it diagonal, as we shall see with benzene. To do this we make all allowable combinations of orbitals (i.e. LCAO's) as determined by symmetry and use these as the basis set instead of just the list of individual atomic orbitals. In this way a symmetry adapted combination is formed, i.e.
# 
# $$\displaystyle \phi_i=\sum_{j=1}^n c_{ji}\varphi_j\qquad\tag{9.34}$$
# 
# where $c_{ji}$ is the amount of an orbital contributes to the total and each $\varphi_i$ is a combination of $f\equiv p_\pi$ orbitals.  The coefficients $c$ are found by solving the equations
# 
# $$\displaystyle \sum_j(H_{ij}-E_kS_{ij})c_{jk}=0, \quad i=1,2\cdots n,\quad k=1,2\cdots n\qquad\tag{9.35}$$
# 
# with each energy $E_k$. The orbitals must finally be normalised (so that $\sum_i c_{ij}^2=1$) and made orthogonal. The plan is therefore to find the LCAO's, eqn (9.34) by using symmetry to find a new set of orbitals $\varphi$ which are combinations of the $p_{\pi}$ atomic ones $f$ that are consistent with the symmetry of benzene. The Huckel Hamiltonian and overlap will be used to form the secular equation eqn. (9.30).
# 
# ### **(i) Huckel approximation**
# 
# In the Huckel approximation 
# 
# $$\displaystyle H_{ii} = \alpha,\quad \text{and}\quad  H_{ij}=\beta\quad \text{when $i$ and $j$ are nearest neighbours else } H_{ij}=0  \qquad\tag{9.34}$$
# 
# The carbon atom's (Coulomb) self energy is $\alpha$, the resonance energy $\beta$ is a negative number.  The overlap integral is 
# 
# $$\displaystyle S_{ij}=\delta_{ij}\qquad\tag{9.35}$$
# 
# The constants $\alpha$ and $\beta$ are empirical so only the relative ordering of energy levels is obtained.
# 
# ![Drawing](matrices-fig26a.png)
# 
# Figure 26a. Some of the symmetry elements of the $D_6$ and  $D_{6h}$ point group. Th $C_2'$ axes are shown as solid lines and the $C_2''$ as dashed lines, both in the plane of the molecule.
# ___________
# 
# Benzene belongs to the $D_{6h}$ point group but to simplify the calculation, the smaller point group $D_6$ can be used as benzene belongs to this group also; $D_{6h}=D_6\otimes C_i$. All that must be remembered is that once the symmetry species are found they must be changed from those of $D_6$ to $D_{6h}$ by looking at the sign of the $\sigma_h$ character.
# 
# The steps to follow are to generate a reducible representation and then to find out what irreducible representation (irreps) this comprises, this will produce the types of all the different symmetry species the $p_{\pi}$ orbitals can form. The projection operator technique is used next to find the shapes of the 'symmetry orbitals' corresponding to each of the symmetry species present. A new secular determinant will be produced from which the energies are found in terms of $\alpha$ and $\beta$ and finally the MO coefficients $c$.
# 
# The $C_6$ point group is
# 
# $$\displaystyle \begin{array}{c|ccccc}
# C_6& E & 2C_6& 2C_3 & C_2 & 3C_2' & 3C_2''\\
# \hline
# A_1& 1 & 1 & 1 & 1 & 1 & 1\\
# A_2& 1 & 1 & 1 & 1 & -1 & -1\\
# B_1& 1 & -1 & 1 & -1 & 1 & -1\\
# B_2& 1 & -1 & 1 & -1 & -1 & 1\\
# E_1& 2 & 1 & -1 & -2 & 0 & 0\\
# E_2& 2 & -1 & -1 & 2 & 0 & 0\\
# \hline
# \end{array}$$

# ### **(ii) Generating the reducible representation and then the irreps**
# 
# The first step in making the symmetry adapted molecular orbitals is to operate on a victim $p_{\pi}$ atomic  orbital, which we label $f_1$, by each of the symmetry operations in the point group. These atomic orbitals are labelled $f_1\cdots f_6$ 
# 
# Operating with the identity produces a diagonal matrix as the identity changes nothing. The trace of this matrix is $6$ so this is the total of the characters for $E$ in the reducible representation. The $C_6$ rotation about the principle axis moves $f_1$ to position $2$, and $f_2$ moves to position $3$ and so on. Thus the matrix begins as 
# 
# $$\displaystyle C_6\qquad\begin{bmatrix}f_1'\\f_2'\\f_3'\\\vdots\end{bmatrix} = \left[\begin{array}{cccccc} 0 &1 &0 &0 &0 &0\\0& 0 & 1 & 0 & 0 & 0\\   &   &\vdots &   &  \\ &  &  \vdots & &\end{array}\right] \begin{bmatrix}f_1\\f_2\\f_3\\\vdots\end{bmatrix}$$
# 
# and the trace is zero so that the character for this operation is zero. The second $C_6$ operation is $C_6^-$, a left rotation. The $C_3$ rotation produces
# 
# $$\displaystyle C_3\qquad\begin{bmatrix}f_1'\\f_2'\\f_3'\\\vdots\end{bmatrix} = \left[\begin{array}{cccccc} 0 & 0 &1 &0 &0 &0 \\0& 0& 0 & 1 & 0 & 0 \\   &   &\vdots &   &  \\ &  &  \vdots & &\end{array}\right]  \begin{bmatrix}f_1\\f_2\\f_3\\\vdots\end{bmatrix}$$
# 
# and again the character is zero. One $C_2'$ operation is rotation about the $f_1 -f_4$ axis which inverts the orbitals (see figure 26a) and and the matrix is,
# 
# $$\displaystyle C_2'\qquad\begin{bmatrix}f_1'\\f_2'\\f_3'\\f_4'\\f_5'\\f_6'\end{bmatrix} =\left[ \begin{array}{cccccc}-1&0&0&0&0&0\\0&0&0&0&0&0\\0&0&0&0&0&0\\0&0&0&-1&0&0\\0&0&0&0&0&0\\0&0&0&0&0&0\end{array}\right] \begin{bmatrix}f_1\\f_2\\f_3\\f_4\\f_5\\f_6\end{bmatrix}$$
# 
# making the character $-2$. There are two other rotations of this type, see fig 26a. A $C_2''$ operation is a rotation about the line between opposite pairs of atoms and as the atoms are moved the diagonal in the matrix is zero. The reduced representation is 
# 
# $$\displaystyle \begin{array}{c|ccccc}
# & E & 2C_6& 2C_3 & C_2 & 3C_2' & 3C_2''\\
# \hline
# \chi_R & 6 & 0 & 0 & 0 & -2 & 0 \\
# \hline
# \end{array}$$

# Following the method in 6.13 this reducible representation consists of four irreps which in $C_6$ are, $A_2,B_2,E_1$ and $E_2$ which means that in the Huckel approximation there are four distinct energy levels of which two pairs are degenerate.  However, benzene belongs to the $D_{6h}$ point group so we must attend to the horizontal reflection in the plane of the molecule ($\sigma_h$) which has character $-1$ or $-2$, which by looking at the characters in this class shows that the symmetry species become $A_{2u}, B_{2g}, E_{1g}$ and $E_{2u}$.
# 
# ### **(iii) Projection Operator and Normalising**
# 
# The next step is to use the projection operator. To do this each symmetry operation in a class is treated separately, i.e. $2C_6$ in the point group is split into $C_6^+$ and $C_6^-$ and similarly for the other classes. A victim orbital is chosen and the operation performed on this. The result is then multiplied by the character in the point group for each symmetry species in turn, $A_2, B_2$ etc. as in eqn. 9.21. 
# 
# The result for $A_2$ symmetry species, choosing the orbital labelled as $f_1$ as the victim $p_{\pi}$ atomic orbital, is
# 
# $$\displaystyle \begin{align}(A_2)\quad\varphi_1 &= f_1+(f_2+f_6)+(f_3+f_5)+f_4 -(-f_1-f_3-f_5)-(-f_2-f_4-f_6)\\
# &= 2(f_1+f_2+f_3+f_4+f_5+f_6)\end{align}$$
# 
# where $f_1$ is found to give $\pm$ the label it moves into. For example, moving $f_1$ with a $C_2''$ moves it to $-f_6$ as shown in fig 26a. Rotating by the other $C_2''$ axes produces $-f_4$ and $-f_6$. As the Huckel method assumes normalised orbitals, for example overlap integral eqn. (9.32/9.35) is $S_{i,j}=\int f_i^*f_j d\tau=\delta_{ij}$ these new orbitals must be normalised, as for example, the $A_2$ species
# 
# $$\displaystyle N^2\int \varphi_1^*\varphi_1d\tau = 4N^2 \sum_i\sum_j\delta_{ij}=1$$
# 
# which is $4N^2(f_1+f_2+f_3+f_4+f_5+f_6)(f_1+f_2+f_3+f_4+f_5+f_6)=1$ but recalling that the orbitals represented by $f_i$ are orthogonal, normalised atomic orbitals, and as $\int f_if_jd\tau=\delta_{ij}$ this produces $\displaystyle 24N^2=1$ and $N=\sqrt{24}$ which makes the normalised orbital
# 
# $$\displaystyle (A_2)\quad\varphi_1= (f_1+f_2+f_3+f_4+f_5+f_6)/\sqrt{6}$$
# 
# The result for the normalised $B_2$ symmetry species is
# 
# $$\displaystyle (B_2)\quad \varphi_2=(f_1-f_2+f_3-f_4+f_5-f_6)/\sqrt{6}$$
# 
# The doubly degenerate representations need two victim orbitals to produce linearly independent basis functions. We can use $f_1$ and $f_2$ giving
# 
# $$\displaystyle \begin{align} 
# (E_1,f_1)\quad\varphi_3&=2f_1+f_2-f_3-2f_4-f_5+f_6\\
# (E_1,f_2)\quad\varphi_4&=f_1+2f_2+f_3-f_4-2f_5-f_6\\
# (E_2,f_1)\quad\varphi_5&=2f_1-f_2-f_3+2f_4-f_5+f_6\\
# (E_2,f_2)\quad\varphi_6&=-f_1+2f_2-f_3-f_4+2f_5-f_6\\
# \end{align}$$

# The doubly degenerate wavefunctions are made orthogonal$^\dagger$  by adding and then subtracting each normalised pair, for example the $E_1$ pair produces
# 
# $$\displaystyle(E_1,+)\quad \varphi_3=(f_1+f_3-f_4-f_5)/2,\qquad (E_1,-)\quad \varphi_4=(f_1-f_2-2f_3-f_4+f_5+2f_6)/\sqrt{12}$$
# 
# and for $E_2$,
# 
# $$\displaystyle (E_2,+)\quad \varphi_5=(f_1-f_2+f_4-f_5)/2,\qquad (E_2,-)\quad\varphi_6=(f_1+f_2-2f_3+f_4+f_5-2f_6)/\sqrt{12}$$
# 
# with these new wavefunction, which are equivalent to the original atomic ones ($p_\pi$) as they describe the same function space, eqn. 9.30 can be evaluated. The matrix element at position $i,j$ is the integral $H_{i,j}$ which, for $i=j=1$, is
# 
# $$\displaystyle \begin{align}H_{1,1}& =\int\varphi_1^*H\varphi_1d\tau \\&=
# \frac{1}{6}(f_1+f_2+f_3+f_4+f_5+f_6)H(f_1+f_2+f_3+f_4+f_5+f_6)\\&=
# \alpha+2\beta \end{align}$$
# 
# where the definitions in eqns. 9.34 and 9.35 are used. The $(1,1)$ entry in the secular determinant is $H_{1,1}-E_1S_{1,1}$ which is $\alpha+2\beta-E$. Dividing this by $\beta$ and letting $x=(\alpha-E)/\beta$ makes this term $x+2$. 
# 
# The other diagonal values are
# 
# $$\displaystyle H_{2,2}=\alpha-2\beta, \quad H_{3,3}=H_{4,4}=\alpha+\beta, \quad H_{5,5}=H_{6,6}=\alpha-\beta, $$  
# 
# and we find that all $H_{ij} = 0$ if $i\ne j$. The secular determinant (eqn. 9.30) becomes
# 
# $$\displaystyle \begin{vmatrix}x+2&0&0&0&0&0\\0&x-2&0&0&0&0 \\ 0&0&x+1&0&0&0 \\ 0&0&0&x+1&0&0 \\0&0&0&0&x-1&0 \\0&0&0&0&0&x-1 \end{vmatrix}=0 $$
# 
# whose solution is direct as the determinant is diagonal and produces $x=-2,x=2,x=-1$ (twice) and $x=1$ (twice) making the energies in order of increasing energy (as $\beta$ is negative), is 
# 
# $$\displaystyle  \alpha+2\beta,\quad (\alpha+\beta,\quad\alpha+\beta),\quad(\alpha-\beta,\quad\alpha-\beta),\quad\alpha-2\beta$$
# 
# and their symmetry species are $A_{2u},(E_{1g},E_{1g}),(E_{2u},E_{2u}),B_{2g}$.
# ___________
# 
# $^\dagger$ If $A$ and $B$ are two normalised functions add and subtract them to make them orthonormal $\int(A+B)(A-B)d\tau=\int A^2 - \int B^2 - \int AB + \int BA =0$ as each integral is unity.
# 

# In[ ]:




