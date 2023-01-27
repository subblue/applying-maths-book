#!/usr/bin/env python
# coding: utf-8

# # 1 SI Units, Unit conversions, Scientific Constants

# ##  1.1 SI units and prefixes
# 
# The International System of Units (SI) is nowadays used in all textbooks and much of the scientific literature. See Mills et al. (1993) or Cohen et al. (2007) for a full description of all units. The SI system is based on a set of defined units so that a quantity either has one of these units, or is derived from them. There is also a set of named prefixes for numerical values and these are shown below.
# 
# ## 1.2 SI Defined units
# 
# The defined units are the metre (m), the kilogram (kg), the second (s), the kelvin, the unit of thermodynamics temperature (K), the ampere (A), the mole (mol), and the candela, which is luminous intensity (cd).
# 
# ## 1.3 SI Derived units
# 
# All other units we use, such as the joule, are derived from these base units. The joule (J) measures energy and $1 \,J = 1 \mathrm{\,kg\, m^2 \,s^{-2} }$. The SI unit for force, the newton (N) is derived from SI base units using the relationship:
# 
# $$\displaystyle \text{ Force = mass } \times\text{ acceleration  N = kg }\times \mathrm{\,m\, s^{-2}}$$
# 
# One newton is defined as the force required to give a mass of 1 kilogram an acceleration of 1 metre per second per second.
# 
# $$\displaystyle\begin{array}{lll}
# \hline\\
# \text{Quantity} & \text{Symbol} & \text{SI base units} \\
# \hline\\
# \text{Force (newton) } & \text{N} & \mathrm{kg \,m \,s^{-2}} \\
# \text{Energy (force }\times\text{ distance), work, heat, (joule) } & \text{J} & \mathrm{kg \,m^2 \,s^{-2}} \\
# \text{Pressure (force/area)(pascal) } & \text{Pa} & \mathrm{N \,/m^2} \\
# \text{Electrical charge (coulomb } & \text{C} & \mathrm{A\,s} \\
# \text{Electrical potential (volt) } & \text{V} & \mathrm{J/C} \\
# \text{Electrical resistance (ohm) } & \mathrm{\Omega} & \mathrm{V/A} \\
# \text{Electrical capacitance (farad) } & \text{F} & \mathrm{C/V} \\
# \text{Inductance (henry) } & \text{H} & \mathrm{ V\,A^{-1}\,s\equiv m\,kg\,\,s^{-2}\,A^{-2}} \\
# \text{Angular velocity } & \mathrm{\omega} & \mathrm{rad \,s^{-1}}(\text{ often just } \mathrm{s^{-1}} \text{ is reported}) \\
# \text{Heat capacity, molar heat capacity} & C_p, C_V & \mathrm{ J\,K^{-1},\; J\,K^{-1} mol^{-1} }  \\
# \text{Entropy, molar entropy} & \text{S} & \mathrm{ J\,K^{-1},\; J\,K^{-1} mol^{-1} }  \\
# \text{concentration} & \text{[c]} & \mathrm{ mol\, m^{-3},}\text{ conventionally }\mathrm{mol\, dm^{-3} }  \\
# \text{Viscosity$^*$} & \mathrm{\eta} & \mathrm{ Pa\,s \equiv kg\,m^{-1}\,s^{-1} }  \\
# \text{Frequency (hertz)} & \mathrm{Hz} & \mathrm{ s^{-1} }  \\
# \text{wavenumber} & \mathrm{cm^{-1}} & \mathrm{ m^{-1} } \text{ but conventionally }\mathrm{cm^{-1}} \\
# \hline\\
# \end{array} $$
# 
# $^*$ Viscosity is still commonly quoted in centipoise (cP) where $1 \mathrm{\,cP} = 10^{-3}$ Pa s and this unit is used because water has a viscosity of $\approx 1$ cP at room temperature, and ethylene glycol about $18$ cP.
# 
# Powers of 10 can be subsumed into the unit by use of prefixes
# 
# $$\displaystyle \begin{array}{ccc}\\
# 10^{-1} & 10^{-2}& 10^{-3} &10^{-6}& 10^{-9}& 10^{-12} &10^{-15}& 10^{-18} &10^{-21} &10^{-24}\\
# \text{deci} & \text{centi}&\text{milli} &\text{micro} &\text{nano} &\text{pico}& \text{femto}& \text{atto} &\text{zepto} &\text{yocto}\\
# d& c& m&\mu& n &p &f &a &z &y\\
# 10& 10^2 &10^3&10^6 &10^9& 10^{12} &10^{15} &10^{18} &10^{21}& 10^{24} \\
# \text{deca} &\text{hecto}& \text{kilo}& \text{mega}& \text{giga}& \text{tera} &\text{peta }&\text{exa} &\text{zeta} &\text{yotta}\\
# D&H& K& M &G& T&P &E &Z &Y\\
# \end{array}$$
# 
# The bond length of HCl is $0.00000000012745$ m; possible choices for recording this are
# 
# $$\displaystyle  r=1.2745\times10^{-10} \mathrm{\,m} = 0.12745 \mathrm{\,nm} = 127.45 \mathrm{\,pm}$$
# 
# of which the final form is becoming common. The angstrom $\overset{\lower 2pt\text{o}}{\mathrm{A}}$ is $10^{-10}$ m and although not an SI unit, is still very frequently used. This bond length is $1.2745 =\overset{\lower 2pt\text{o}}{\mathrm{A}}$.
# 
# 
# ## 1.4 Atomic units
# 
# When working at quantum problems, it is sometimes easier to use atomic units. In these units the electron charge $e$ is taken to be 1 unit of charge and its mass $m_e$ also 1 unit of mass. The energies are always electrostatic and hence proportional to $e^2/(4\pi\epsilon_0)$ where $\epsilon_0$ is the permittivity of free space and has units $\mathrm{F \,m^{-1} = C^2\,J^{-1}\, m^{-1}}$. This quantity in SI units is $\mathrm{C^2/(C^2\,J^{-1}\, m^{-1}) = J\,m}$ and therefore has dimensions mass $\times$ length$^3 \times$ time$^{-2}$. Planck's constant squared has units of $\mathrm{(J s)^2 = mass^2 \times length^2 \times time^{-1}}$ and using these quantities can now define a length as
# 
# $$\displaystyle a_0=\frac{\hbar^2}{m_e(e^2/4\pi\epsilon_0)}= 0.529177\cdot 10^{-10} \mathrm{m}$$
# 
# This is the Bohr radius and the unit of length in atomic units. It corresponds to the radius of the 1s orbit of a H atom. The unit of energy is the hartree, which is
# 
# $$\displaystyle E_h=\frac{e^2}{4\pi\epsilon_0}\frac{1}{a_0} = 4.35974\cdot 10^{-18} \mathrm{J} = 27.2144 \mathrm{eV}$$
# 
# and is twice the absolute value of the energy of the 1s electron in hydrogen or twice the ionization energy. The unit of time in atomic units is $\hbar/E_h = 2.41888 \cdot 10^{-17}$ s.
# 
# ## 1.5 Converting a number to different units
# 
# It is often necessary to convert between different sets of units. All non-SI units have a definition in terms of the corresponding SI units. An example conversion table for pressure is
# 
# $$\displaystyle \begin{array}{lll}
# \mathrm{ 1\;Pa} & =\mathrm{ 1\,Nm^{-2} = 1\,kg\,m^{-1}s^{-2} }\\
# \mathrm{1\; bar}& = \mathrm{100\, kPa}\\
# \mathrm{1\; atm}& = \mathrm{101\; 325\, Pa}\\
# \mathrm{1\; atm}&= \mathrm{760\; Torr}\\
# \mathrm{1\; psi^*}& = \mathrm{6894.76\,Pa }\end{array}$$
# 
# *psi is pounds per square inch.
# 
# A good source of conversions is the CRC Handbook of Chemistry and Physics (Weast).
# 
# The search engine Google allows one to ask questions such as 'calculate the speed of light in furlongs per fortnight'. The answer, by the way, is $1.8 \times 10^{12}$ and a furlong is $220$ yards. However, although these programs will do this for you it is clearly better to have an idea of what to do yourself.
# 
# To convert pressure data from the units of Torr to the SI unit of $\mathrm{N\,m^{-2}}$ (or Pa), one of two methods can be used
# 
# ### **Method 1; Direct Substitution**
# 
# Substitute the value of $1$ torr for its equivalent in $\mathrm{N\,m^{-2}}$ as if 'torr' were a variable in the equation:
# 
# $$\displaystyle p = 760\, \mathrm{torr} = 760 \times(133.322\, \mathrm{N\, m^{-2}}) = 101325\, \mathrm{N\,m^{-2}}$$ 
# 
# ### **Method 2; Multiply by 1**
# 
# This is the most reliable method to use. The equation is multiplied by $1$ by using a unit conversion so that the unit to remove is on the denominator. For example from the definition of a torr,
# 
# $$\displaystyle  1 = 133.322 \,\mathrm{N m^{-2}/\mathrm{torr} }$$
# 
# and substituting makes the equation
# 
# $$\displaystyle p = 760 \,\mathrm{torr} = 760\,\mathrm{ torr}\times \frac{133.322\,\mathrm{ N\, m^{-2} }}{\mathrm{torr} } = 101325\,\mathrm{N\, m^{-2}}$$
# 
# and the units cancel giving the result in the new units. To express a pressure of $62$ psi in
# Pascal, using the conversion table above and the 'multiply by 1' method, gives
# 
# $$\displaystyle 62\, \mathrm{psi} = 62\, \mathrm{psi}\left( \frac{6894.7\,\mathrm{Pa}}{\mathrm{psi}}\right) = 4.27\cdot 10^5\, \mathrm{Pa}$$
# 
# Planck's constant is $h = 6.626 \cdot 10^{-34}$ J s, but in some cases it is easier to use this in units of eV ps where $1 \mathrm{ps} = 10^{-12}$ s, and one electron volt, $1 \mathrm{eV} = 1.602 \cdot 10^{-19}$ J, giving
# 
# $$\displaystyle h=6.626\cdot 10^{-34} \mathrm{J\,s}\left( \frac{\mathrm{eV}}{1.602\cdot 10^{-19}\,\mathrm{J}}  \right) \left( \frac{\mathrm{ps}}{10^{-12}\, \mathrm{s}} \right) = 4.136\cdot 10^{-3} \mathrm{eV\,ps}$$
# 
# This is even more useful in cm$^{-1}$ ps, which is calculated as
# 
# $$\displaystyle h=6.626\cdot 10^{-34} \mathrm{J\,s}\left( \frac{\mathrm{cm^{-1}}}{1.9862\cdot 10^{-23}\,\mathrm{J}}  \right) \left( \frac{\mathrm{ps}}{10^{-12}\, \mathrm{s}} \right) = 33.36\cdot 10^{-3} \mathrm{cm^{-1}\,ps}$$
# 
# making $\hbar = 5.3 \mathrm{\,cm}^{-1}$ fs.
# 
# ## 1.6 Conversion table: Energy units and related quantities
# 
# $$\displaystyle \begin{array}{c|ccccc1 }
# \hline\\
#  & \mathrm{J} & \mathrm{kJ\, mol^{-1}} &\mathrm{eV} &\mathrm{Hz} &\mathrm{cm^{-1}}  \\[0pt]
#  \hline\\
# \mathrm{J} & 1&6.022\times10^{20}  & 6.241\times10^{18}  &1.509\times10^{33}  &5.034\times10^{22}\\[5pt]
# \mathrm{kJ\,mol^{-1}} & 1.661\times10^{-21} &1  & 1.036\times10^{-2}  &2.506\times10^{12}  &83.59\\[5pt]
# \mathrm{eV}&1.602\times10^{-19} & 96.48 &1  &2.418\times10^{14} &8.065\times10^3\\[5pt]
# \mathrm{Hz}&6.626\times10^{-34} & 3.990\times10^{-13} & 4.136\times10^{-15}  &1 &3.336\times10^{-11}\\[5pt]
# \mathrm{cm^{-1}}&1.986\times10^{-23} & 1.196\times10^{-2} &1.240\times10^{-4}  &2.998\times10^{10}  & 1\\
# \hline\end{array}$$
# 
#   
#  To convert $6$ eV into $\mathrm{cm^{-1}}$  read along from eV in the left column and multiply by the number under 
# $\mathrm{cm^{-1}}$ in the top row, e.g.
# 
# $$\displaystyle 6 \mathrm{eV}\equiv 6 \times \frac{8.065 \cdot10^3 \mathrm{cm^{-1}}}{ 1\,\mathrm{eV}} = 4.839\cdot10^4 \mathrm{cm^{-1}} $$
# 
# To convert $k_BT$ into $\mathrm{cm^{-1}}$ at $T = 300$ K
# 
# $$\displaystyle  1.381\cdot10^{-23} (\mathrm{J/K}) \times300 \,\mathrm{K} \equiv (1.381\cdot10^{-23} \times 300) \,\mathrm{J} \times \frac{5.034\cdot 10^{22} \mathrm{cm^{-1}}}{1\,\mathrm{J}} = 208.5 \,\mathrm{cm^{-1}}$$ 
# 
# ## 2 Table of Scientific Constants
# 
# Values mainly from CODATA 2018, and NIST SP961 2019.
# 
# $$\displaystyle \begin{array}{lll}\\
# \hline
# \text{ Quantity} & \text{Symbol, equation} & \text{value}\\
# \hline
# \text{Speed of light in vacuum}       &c & 2.99792458\cdot 10^{8} \;\mathrm{m\,s^{-1}}\text{ (exact)} \\ 
# \text{Planck constant}  &h & 6.62607015\cdot10^{-34}\;\mathrm{ J\,s} \text{ (exact)}\\
# \text{Planck constant, reduced } & \hbar=h/2\pi & 1.054571817\cdot10^{-34}\;\mathrm{ J\,s} \text{ (exact)}\\
# \text{Electron charge magnitude } & e& 1.602176634\cdot 10^{-19}\;\mathrm{ C} \text{ (exact)}\\
# \text{Electron mass } & m_e & 9.109 383 7015(28)\cdot 10^{-31}\;\mathrm{ kg}\\
# \text{Proton mass } & m_p &1.672 621 923 69(51)\cdot 10^{-27}\;\mathrm{ kg}\\
# & & 1.007 276 466 621(53) u=\\&& \quad 1836.152 673 43(11) m_e\\ 
# \text{Neutron mass} &m_n & 1.008 664 915 95(49) u  \\
# \text{Unified atomic mass unit}^{**} &u=\mathrm{(mass\; ^{12}C\;atom)}/12 & 1.66053906660(50)\cdot 10^{-27}\;\mathrm{ kg}\\
# \text{Permittivity of free space } &\epsilon_0=1/\mu_0c^2 & 8.8541878128(13)\cdot 10^{-12}\;\mathrm{ F\,m^{-1}}\\
# \text{Permeability of free space } &\mu_0/(4\pi\times 10^{-7}) & 1.0 \;\mathrm{N\,A^{-2}}\\
# \text{Bohr radius, } (m_\text{nucleus} = \infty)   & \alpha_0=4\pi\epsilon_0\hbar^2/m_ee^2 &  0.529177210903(80)\times 10^{-10}\;\mathrm{ m}\\
# \text{Rydberg energy } & hcR_\infty=m_e e^4/2(4\pi\epsilon_0)^2\hbar^2& 13.605693122994(26)\;\mathrm{eV}\\
# \text{Bohr magneton } &\mu_B=e\hbar/2m_e & 5.7883818060(17)\cdot 10^{-11}\;\mathrm{ MeV\,T^{-1}}\\
# \text{Nuclear magneton } &\mu_N=e\hbar/2m_p & 3.15245125844(96)\cdot10^{-14} \;\mathrm{ MeV T^{-1}}\\
# \text{Gravitational constant  } &G_N & 6.67430(15)\cdot 10^{-11}\mathrm{ m^3\,kg^{-1}\,s^{-2}}\\
# \text{Standard gravitational accel. } &g_N & 9.80665\;\mathrm{m\,s^{-2}}\\
# \text{Avogadro constant  } &N_A & 6.02214076\cdot 10^{23}\;\mathrm{ mol^{-1}} \text{ (exact)}\\
# \text{Boltzmann constant } &k_B & 1.380649\cdot 10^{-23}\;\mathrm{ J\,K^{-1}} \text{ (exact)}\\
# \text{Molar volume, ideal gas at STP  } &N_Ak_B(273.15\;\mathrm{K})/(101325\;\mathrm{Pa}) &22.41396954\;\mathrm{ dm^3 mol^{-1}}\\
# \text{Faraday constant} & F & 96485.4\;\mathrm{C\,mol^{-1}\;(A\,s\,mol^{-1})}\\
# \text{Molar gas constant} & R & 8.31447\;\mathrm{J\,mol^{-1}\,K^{-1}}\\
# \text{Hartree energy} & E_h& 4.35974\cdot 10^{-18}\;\mathrm{J}\\
# \end{array}$$
# 
#  ** The molar mass of $^{12}$C is $11.9999999958(36)$ g.
# 
# ## 2.1 Some common and non SI unit conversions.
# 
# $$\displaystyle \begin{array}{lll}
# \hline
# 1\text{ inch}& \equiv &0.0254 \text{ m}\\   
# 1\text{ G (gauss)} &\equiv &10^{-4}\text{ T (Tesla)}\\
# 1\text{ eV} &=&1.602176634\cdot 10^{-19} \text{J (exact)} & \\
# k_BT  \text{at }  300 \text{ K} &= & 1/38.681740   \text{ eV}\\
# 1 \text{ Debye}& \equiv& 3.33\cdot 10^{-30} \text{ C m}\\
# 1\; \overset{\text{o}}A & \equiv & 0.1 \text{ nm}\\
# 1 \text{ C} &\equiv& 2.99792458\cdot 10^9 \text{ esu}\\
# 1 \text { atmosphere} &\equiv& 760 \text{ torr} \equiv 101325\text{ Pa}\\
# 1\; \ell\text{ atm} & = &101.3 \text{ J}\\
# 0 ^\text{o}\text{ C} &\equiv& 273.15  \text{ K}\\
# 1\text{ cP} & \equiv& 10^{-3}\text{ Pa s}\\
# 1 \text{ mm Hg}& \equiv& 1 \text{ torr}\\
# 1\text{ calorie} & \equiv & 4.184\text{ J}\\
# 1\text{ erg} & \equiv &10^{-7}\text{ J}\\
# 1 \text{ dyne (dyn)} & \equiv& 10^{-4}\text{ N}\\
# 1 \text{ enzyme unit U} & = & 1\;\mathrm{\mu mol/min}\\
# \hline \end{array}$$
#   

# ## 3.0 Significant figures and rounding numbers
# 
# A measurement always has two parts: a numerical value and its associated units. It is essential to report numbers to the appropriate number of decimal places. This is done either according to what is possible from the experimental conditions, or from the precision of numbers used in a calculation; and therefore some adjustment, called rounding, of the number is necessary.
# 
# There is no single or perfect way of representing a number and so there is room for some personal preference, but scientific notation offers the least ambiguity. To represent a number in this notation, write it with one leading digit followed by the decimal point and then more digits followed by a power of $10$; for instance,
# 
# $$\displaystyle 97453.1 \to 9.74531 \cdot 10^4,\qquad 0.0245 \to 2.45 \cdot 10^{-2}$$
# 
# and the leading zeros in the second number are seen not to be significant. If a series of numbers are to be compared, then it is best to use the same power of $10$ for each; for example, the following three numbers shows how the steady increase in value can easily be recognized. It is even clearer when these numbers are rounded up, say to the nearest 1000, and this gives the values shown on the right.
# 
# $$\displaystyle\begin{align}
# 984.7 &\to 0.9847 \cdot 10^3 \to 1\cdot 10^3\\
# 60357 &\to 60.357\cdot10^3 \to 60\cdot 10^3\\
# 124560 &\to 124.560 \cdot 10^3 \to 125 \cdot 10^3\end{align}$$
# 
# By the nature of any experiment, a measurement is only known to a certain number of significant digits. These are independent of the size of the number. For instance, $123.4$ and $0.00001234$, both contain four significant figures and these numbers can be written as $1.234 \cdot 10^2$ and $1.234 \cdot 10^{-5}$ respectively. The number $123.40$ contains five significant figures, because writing the last zero implies that this digit is known. The limited number of significant figures occurs because of many unknown variations in the way a quantity is measured. Imprecision may occur due to a sloppy experimental technique, but supposing that this is not the case, 'noise' may be added to a measurement from any number of sources. By 'noise' is meant the random variability that is seen in any measurement that you would prefer is not there and which often masks the true signal. Perhaps this noise is due to interference from other instruments in the laboratory, from mains voltage fluctuations, or from temperature variations. Noise could be present because measurements were at the limit of your instrument's capability. Often data is subsequently analysed to extract the information it contains; the slope of a line, for example, and then the number of significant figures reported as the slope and its uncertainty must be carefully considered. A calculator or computer will happily produce ten or more digits in an answer; almost invariably far more than is realistic. First, to decide how many significant figures there really are, the original data must be looked at before reaching a conclusion. The result can then be rounded to the correct number of significant figures at the very end of any calculation. Both the result and its associated error will need to be rounded, and this is done with a set of rules.
# 
# In rounding numbers we examine the last digit to be retained and
# 
# **(1)**$\quad$  Retain no more digits beyond the first uncertain one.
# 
# $\quad$**(i)** Increase this digit by $1$ if the residue is greater than $5$.
# 
# $\quad$**(ii)** Leave this digit unchanged if the residue is less than $5$.
# 
# $\quad$**(ii)** When the residue is exactly $5$, leave this digit unchanged if previous one is even, or increase it by $1$ if it is odd. This makes rounding unbiased.
# 
# Using these rules produces the following values when rounded to $4$ significant figures or three decimal places.
# 
# $\quad$**(i)** $\quad 1.02055 \to 1.021$;  rule (i). 
# 
# $\quad$**(ii)** $\quad 1.02345 \to 1.023$;  rule (ii). 
# 
# $\quad$**(iii)**  $\quad 1.02350 \to 1.024$;  rule (iii) increase by $1$, as the last retained digit is $3$.
# 
# $\quad$**(iv)**  $\quad 1.02450 \to 1.024$;  rule (iii) unchanged as significant digit is $4$ and is even.
# 
# **(2)**$\quad$  In addition or subtraction, do not retain any more digits in the answer than the number with the smallest number of digits,
# 
# $\qquad 21.1 + 2.035 + 6.12 = 29.255 \to 29.3$.
# 
# **(3)**$\quad$  In multiplication or division, the result should have no more digits than the least precise number, which has the smallest number of significant digits: 
# 
# $\quad$**(i)**  $\quad 21.1 \times 0.029 \times 83.2 = 50.91008 \to 51$, because $0.029$ has $2$ significant digits. This result would be better written as $51.0$ because $51$ implies that the number is an integer which is known exactly. 
# 
# $\quad$**(ii)** $\quad 291 \times 272/0.086 = 920 372.093 \to 9.2 \cdot 10^6$ because $0.086$ has $2$ digits.
# 
# **(4)**$\quad$ The log of a number should have as many digits to the right of the decimal point
# (the mantissa) as there are significant digits in the number.
# 
# **(5)**$\quad$  The mean of a number has as many significant figures as the observations upon which it is made; only such a number of significant digits should be retained so that the uncertainty in the mean corresponds approximately to its standard deviation.
# 
# Examples of these rules are
# 
# $$\displaystyle\begin{array}121.1 + 2.035 + 6.12 &= 129.255 &\to 129.3\\
# 291 \times 272/0.086 &= 920 372.093 &\to 9.2 \cdot 10^6\\
# \log(4.000 \times 10^{-5}) &= -4.3979 &\\
# 10^{12.5} &= 3.16 \cdot 10^{12} &
# \end{array}$$

# ## 3.1 Experimental results with experimental uncertainties or errors
# 
# There are only a few instances is chemistry where a value you are trying to measure or calculate is an integer, for example a quantum number or the number of atoms in a unit cell of a crystal but in most cases the result will be a real number. The number of significant figures quoted indicates how precise you consider the number to be. As an example a mass of $2.3457$ g measured on an analytical balance must be assumed to be somewhere in the range $2.34565 \to 2.34575$, i.e an _absolute error bound_ of $0.00005$ g which is half the last significant figure. 
# 
# The _relative error bound_ is the absolute bound divided by the value itself, $0.00005/2.3457 = 2.1315\cdot 10^{-5}$ or $0.0000213$ but the error bound is $0.00005$ so this answer is only significant as far as the $5^{\mathrm{th}}$ decimal place or $0.00002$ so is reported as $2\cdot 10^{-5}$.  The _percentage error_ is $100$ times the relative bound.
# 
# The error quoted with a measured mean value represents the chance that the data will fall between the error values quoted, usually this is approximately $68$%, meaning that by random chance $32$% of the times a measurement is made, a value outside the range will be observed. In some cases a $95$% limit may be used; this will depend on how the standard deviation or the standard error is defined and this is explained in more detail in Chapter 13. We shall suppose that this has been decided upon, so that just the numbers are examined. To report a result as $7.56 \pm 0.03456$ kJ/mole would be wrong. It is reasonable to assume that the result is the mean value of several measurements because an error is given. The mean is quoted to only three significant figures, one part in a thousand, or two decimal places, and as the error is usually calculated from the data, it cannot be known to more figures than this. The result should be reported as $7.56 \pm 0.03$ kJ/mole as the $3$ in the quoted error falls in the same decimal place as the $6$ in the number. If you want to be cautious, then $7.56 \pm 0.04$ would be acceptable as would $7.56 \pm 0.03_4$ to indicate the figure that could be rounded.
# 
# As a rule of thumb, experimental uncertainties (errors) should be rounded to one significant figure, unless the measurement is very precise, then two figures may be used. The error would normally always be rounded up. Once the uncertainty in the measurement is determined, the significant figures in the measured value may need to be revised. This would be the case if the error were determined by some means other than from analysing the data. Reporting  $\Delta G = -6051.78 \pm 30$ J/mole is just not reasonable as the error is so large; the result should be $\Delta G = -6050\pm 30 $ J/mole, as an uncertainty of $30$ means that the result could be as small as $6020$ or as large as $6080$, so that the trailing digits, $1, 7, 8$ do not matter; see figure 26.
# 
# A second rule of thumb is that the last significant figure in any stated answer should be of the same order of magnitude, i.e. in the same decimal position, as the uncertainty. For example,
# 
# $$\displaystyle 92.81 \pm 0.3 \text{ should be reported as } 92.8\pm 0.3\\ 
# 92.81\pm 3 \to 93\pm 3\\
# 92.81\pm 30 \to 90 \pm 30$$
# 
# In the last case, the rounded result is a little smaller than the result but the error is so large that this is of no consequence. In other words, the $92.81$ is only one of many results that could have been obtained had the experiment been repeated many more times and values from at least $60 \to 120$ are to be expected. In cases such as $92.5\pm 0.35$, you might not want to round up either to $93$ or down to $92$. In this case the rounded number could be reported as $(0.92_5 \pm 0.04) \times 10^2$.
# 
# ![Drawing](chapter1-fig26.png)
# 
# Figure 26. Illustrating the error of $\pm 30$ on the number $-6051.78$. The red dot is the original number, the blue dot the rounded one, and the red line the error. 
# ________
# 
# ## 4 Glossary of Selected Mathematical Symbols
# 
# $$\begin{array}{lll}
# \hline
# \large\text{Symbols} & \large\text{meaning}\\
# a=b & \text{Equality, with numbers }\pi = 3.14159 \cdots \text{ dots are added as necessary.}\\
# a\ne b & a \text{ is not equal to }b\\
# a\equiv b & \text{Identity; } a \text{ is identical to }b, (a + b)^2 \equiv a^2 + 2ab + b^2. \text{ Rarely used.}\\
# a\lt b & a \text{ is less than }b\\
# a\gt b & a \text{ is greater than }b\\
# \le & \text{less than or equal to}\\
# \ge & \text{greater than or equal to}\\
# \ll & \text{much less than}\\
# \gg &\text{much greater than}\\
# a \propto b & a \text{ is proportional to } b\\
# a\approx b & a\text{ is approximately equal to }b\\
# a\sim b & a\text{ is of the order of }b, \text{ or a changes at the same rate as }b\\
# 373 \overset{\text{^}}= 100\,^\text{o} & \text{Indicates change of units to equivalent value}\\
# \pm a & \text{Values are plus and minus }a\\
# \mp a & \text{Values are minus and plus }a\\
# a \text{^} b & a\text{ raised to power }b\\
# \displaystyle \frac{a}{b}, \; a/b,\; a\div  b & a\text{ divided by }b\\
# \angle & \text{angle}\\
# a\perp b,\; a \,||\, b & a\text{ perpendicular and parallel to }b\\
# \infty & \text{infinity}\\
# \to &\text{tends to or approaches, used as  }a\to \infty,\;a\to 0\\
# \sum & \text{summation, }\displaystyle  \sum_{i=0}^n x_i=x_0+x_1+x_2 + \cdots+x_n \\
# \prod & \text{product, }\displaystyle  \prod_{i=0}^n x_i=x_0x_1x_2 \cdots x_n \\
# ! & \text{Factorial } n! = 1.2.3.4.5\cdots n\\
# \delta(x) & \text{Kronecker Delta }\delta (x) = 1\text{ if }x = 0 \text{ else is zero}\\
# \delta_{n,m} & \text{Delta function }\delta_{n,m} = 1\text{ if }x = 0 \text{ else is zero},\; n,m\text{ are integers}\\
# f(x) & \text{function of } x\\
# f'(x)\;f''(x) & \text{First and second derivatives wrt. }x\\
# \dot f,\;\ddot f & \text{First and second derivatives wrt. to time}\\
# \text{O}(x^n) & \text{Big ‘O’ notation. Used in series expansion to indicate that the next}\\& \text{unwritten terms do not grow faster than }x^n\\
# \hline
# \end{array}$$
# 
# Other more complex symbols such as for differentiation are given when their usage is described in the text.

# In[ ]:




