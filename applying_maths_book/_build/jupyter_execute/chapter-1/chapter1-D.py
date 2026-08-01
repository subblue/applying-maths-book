#!/usr/bin/env python
# coding: utf-8

# # 10 Modulo arithmetic, $\delta$ functions, arithmetic and geometric series &  estimating quantities

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 16})  # set font size for plots if needed


# ## 10.1 Modulo arithmetic of integers
# 
# Modulo arithmetic uses integer division to find remainders which themselves are a limited set of integers. We routinely use clock time, which is measured in base $60$ for minutes and base $12$ for hours. For example if you arrange to meet someone at $10:35$ and they are $40$ minutes late, then rather than noting the time as $10:75$ we say it is $11:15$. This is an easy mental calculation expressed mathematically and somewhat confusingly as, 
# 
# $$\displaystyle 35 + 40 \equiv \,15\,(\mathrm{mod}\, 60) $$
# 
# which shows that integer division $75/60$ has an integer remainder of $15$ or that $60$ divides exactly into $75-15$, i.e. $\displaystyle \frac{75-15}{60}=1$ with remainder of zero.  The modulus is $60$. For instance working in modulo $3$ we find
# 
# $$\displaystyle \begin{align}4&\equiv 1(\,\mathrm{mod}\,3),\qquad\displaystyle 3+4 \equiv 1\\
# -4&\equiv 5\,(\mathrm{mod}\,3),\qquad 3-4\equiv 2\\
# 12&\equiv 0\,(\mathrm{mod}\,3),\qquad 3+12 \equiv 0\\
# 112&\equiv 1\,(\mathrm{mod}\,3),\qquad 3+112 \equiv 1\\\end{align}$$
# 
# where the second column shows that addition is equivalent to clockwise and subtraction to anticlockwise rotations around a circle numbered $0 \to 2$, see figure 24. We could also write $3^n+4\equiv 1$ where $n\ge 0$ for the first example, meaning $n$ whole rotations bring us back to zero.
# 
# If the difference between two integers $a$ and $b$ is an exact multiple of integer $m$, or $\displaystyle \frac{(a - b)}{m}$ is a whole number (integer), this means '$a$ and $b$ are *congruent modulo* $m$'; hardly an expression that flows naturally off the tongue. Congruence here means 'in agreement with'. This is the case in our example of clock time, integer division $(35+40-15)/60 = 1$ with no remainder. As an identity, this is written as
# 
# $$\displaystyle a\equiv b\,\mathrm{mod}\,m \qquad \text{or as}\qquad a\equiv b\,(\mathrm{ mod }\,m)$$
# 
# which means that $a$ is *equivalent* to $b\,\mathrm{mod}\,m$ and is called a congruence. More generally if $a=qm+b$ then $a\equiv b\, (\mathrm{mod}\,m)$ where $q$ is an integer. 
# 
# 
# More familiarly, it means that *integer division* $a/m$ produces the *remainder* $b$; thus $75/60 = 1$ with remainder $15$. Integer division measures only how many times one integer goes into another, ignoring any remainder; thus $3/2 = 1,\, 23/4 = 5$, and so forth, but the modulus is different because integer division occurs repeatedly until the remainder is in the range $0,1,\cdots m-1$,  i.e. $23\equiv 3\,(\mathrm{mod}\, 5)$ 
# 
# 
# 
# ![Drawing](chapter1-fig24a.png)
# 
# Figure 24. Showing the effect of modulo $3$ arithmetic, for example $4\equiv 1\,( \mathrm{mod}\,3)$ which means that the division $(4-1)/3$ has no remainder.  Move clockwise to increase values and anti-clockwise to decrease. Each list of equivalent numbers is called a congruence class.
# ____________________________
# 
# The table shows some useful properties of the modulus.
# 
# $\displaystyle \begin{array}{lll}
# \hline
# \text{if }  a=b   \text{ then }  a\equiv b\,(\mathrm{mod}\, n)\\
# a\equiv a \,(\mathrm{mod}\, n)\\
# \text{if } a\equiv b\,(\mathrm{mod}\, n) \text{ then } b\equiv a\,(\mathrm{mod}\, n)\\
# \text{if } a\equiv b\,(\mathrm{mod}\, n) \text{ and } b\equiv c\,(\mathrm{mod}\, n) \text{ then } a\equiv c\,(\mathrm{mod}\, n)\\
# \text{if } a\equiv b\,(\mathrm{mod}\, n) \text{ and } c\equiv d\,(\mathrm{mod}\, n) \text{ then } a+c\equiv (b+d)\,(\mathrm{mod}\, n) \text{ and } ac\equiv bd \,(\mathrm{mod}\, n)\\
# \hline\end{array}$
# 
# 
# However, the syntax in most computer languages is different to that used so far. In Python the modulo calculation is written as 
# 
# $$\displaystyle  \mathtt{a = b} \text{ % }\mathtt{ m}$$
# 
# which means that $a$ is the integer *remainder* of integer division of $b$ by $m$, this is equivalent to $b-(b//m)$ where // means integer division.
# 
# Using Python,
# 
# $$\displaystyle  75 \text{ % }  60 \qquad \text{ produces } 15 $$
# 
# because integer division $75/60 = 1$ with a remainder of $15$. When $b$ is a negative integer, the result is the same as calculating $m - (+b \text{ % } m)$, and as mentioned above, is the same as moving anticlockwise round a circle. For example,
# 
# $$\displaystyle \begin{align} +39 \text{ % } 8 \qquad &\text{ produces } 7\\
# -39 \text{ % } 8 \qquad &\text{ produces } 1 \\
# 8-(+39 \text{ % } 8) \qquad &\text{ produces } 1 \end{align}$$
# 
# 
# ## 10.2 CAS registry number and Check digits
# 
# An important example is found in the CAS registry number, which is used to uniquely identify every chemical compound.  There are at least 35 million compounds known, so uniqueness is important. The last digit in the CAS number is a check digit, used to confirm the uniqueness of the number. The check digit is obtained by multiplying each preceding digit by its position in the number; taken in reverse order, starting by multiplying the last digit by $1$, the second to last by $2$, etc. then adding the result and calculating modulo $10$ of the number produced. The CAS number for naphthalene is $(91 - 20 -3)$; the $3$ is the check digit. This is obtained with the following sum, $0 \times 1 + 2 \times 2 + 1 \times 3 + 9 \times 4 = 43 $ and finally, $43 \text{ mod } 10 = 3$, where the $3$ is the remainder of a whole number of divisions of $10$ into $43$.
# 
# In transmitting digital data, check digit information is added to ensure that the data has not been corrupted and is calculated as for the CAS example .
# 
# ## 10.3 Musical scales
# 
# Musical scales and clock time use circular arithmetic. The equal temperament scale has $12$ frequencies in each octave given by $2^{n/12}$ times the base frequency where $n = 1 \to 12$, this being the scale that most people find is most pleasing to their ears. The next octave, above or below, has exactly the same ratio of frequencies and so follows modulo $12$ arithmetic. 
# 
# ## 10.4 Algorithm converting a decimal number to and from another base
# 
# When writing a numeral such as $123$ we naturally assume it is in decimal. The conversion we do in our heads is so automatic that we never think about it and is the calculation $1\cdot 10^3+2\cdot 10^1+3\cdot 10^0$ where here the base is $10$. If we are using another base, say octal base $8$, then the octal number $123$ is $1\cdot 8^2+2\cdot 8^1+3\cdot 8^0 = 83$ in decimal. 
# 
# If we want to know what numeral 123 in decimal is in octal (or any other base) we can use modulo arithmetic and start with the decimal and cut it into parts. We have to use integer division which mean whole number division and ignoring any fractional value, thus $3//2 = 1$ where $//$ means integer division. In comparison $ 3 \text{ mod } 2 = 1$ which is the remainder.
# 
# An algorithm to do this is 
# 
# **(1)** modulo the number with the base and save it. This produces remainder.
# 
# **(2)** use integer division on the number with the base and make this the new number.
# 
# **(3)** repeat from step 1.
# 
# **(4)** print result in reverse order.
# 
# Modulo $123$ with the base $8$ is $3$ calculated as integer division $123//8 = 15$ then $123 - 8 \times 15 = 3$, i.e remainder after dividing $123$ by $8$.  Next repeat this with $15$ which gives $15\mod\,8 = 7$  and integer division $1$, reversing the order gives $173$. Some code to do this is given below.

# In[2]:


# Convert a decimal number (num) to another base.

def convert_base(base,num):
    if base < 2:
        return
    newn = []
    while num > 0:
        rem = num % base               # modulo
        if rem > 9:
            rem = chr(65 - 10 + rem)   # make into letter if > 9, look up ascii code numbers
        newn.append( rem  )
        num = num // base              # integer division
        pass
    return newn
#------------------------

base = 16                # base 2 is binary, 8 is octal, 10 decimal, 16 hexadecimal
num  = 19701

ans = convert_base(base,num)

print('decimal ',num,' in base ',base,' is ', ans[::-1] )  # [::-1] reverses the order of elements


# To work in the opposite direction, i.e. to start say with $123$ in base $8$ and get the decimal we must find the position of each number and multiply it by 10 to the power of its position. This is most easily achieved on a computer by converting the number to a string and extracting each character, converting it to a number and multiplying this by the base raised to the power. Making a string is important as in hexadecimal, for example, the number may contain a letter and this has to be converted to a number.

# In[3]:


# convert from a base to decimal

def convert_to_decimal(base, anum):
    if base < 2:
        return
    num  = anum[::-1].upper()       # reverse string and make letters capitals
    decn = 0
    for i in range(len(num)):
        temp = (ord(num[i]) - 48 )
        if temp not in [0,1,2,3,4,5,6,7,8,9]:
            temp = (ord(num[i]) - 65 ) + 10   # ascii -65 but A must be 10
        decn = decn + temp*base**i
    return decn
    
#--------------------
base = 16
num  = '4CF5'    # no check is made as to whether the number is possible in the base used.

ans = convert_to_decimal(base,num)
print(num,' in base ',base,' is ', ans, 'in decimal' )


# ## 11 Delta functions,  Krokecker and Dirac
# 
# The Kronecker delta is a function defined with two indices, and has a value of $1$ when these are the same, and $0$ when they are different.
# 
# $$\displaystyle  \delta_{n,m} = 1 \quad\text{ if } n=m,\quad \text{ i.e. } \delta_{n,n} =1,\quad \text{ otherwise } \delta_{n,m} =0$$
# 
# This function is met when calculating the orthogonality of many functions with integer arguments, for instance, $\sin(nx)$ and $\cos(mx)$, see Chapter 9, and is commonly also found in quantum problems where it is often used to pick out one term in a summation;
# 
# $$\displaystyle s_j =\sum_i a_i\delta_{i,j} =a_0\delta_{0,j} +a_1\delta_{1,j}\cdots +a_j\delta_{j,j}+ +\cdots a_n\delta_{n,j} =a_j$$
# 
# The second type of delta function $\delta(x)$ is named after Dirac. This behaves like a normal continuous function but has a positive value only at $x$ and elsewhere is exactly zero. The area under the function, which is its integral, is unity. This means that the function is a spike at position $x$ of infinitesimal width but unit area. Further properties of this function are described in Chapter 9. The Dirac delta can be derived in a number of ways, but the function may be realized by drawing a Gaussian (bell-shaped curve), then making it narrower and narrower until, at the limit, it becomes the $\delta$ function.
# 
# Just as for the Kronecker delta the Dirac delta function can be used to extract values, in this case from an integral,
# 
# $$\displaystyle \int f(x)\delta(x-a)dx=f(a)$$
# 
# which can be seen as the delta function is zero except when $x=a$.
# 
# ## 12 Series: Arithmetic and Geometric progressions
# 
# A series of numbers is $1, 2, 3, 4 \cdots$ or $1, 2, 4, 8, 16 \cdots$. If each term is made from the previous one by adding a constant number, this is an _arithmetical progression_. If, however, each term after the first is multiplied by a constant term, the series is called a _geometrical progression_. Both series may continue to infinity, but the last value may still be finite.
# 
# ### **Arithmetic progression**
# 
# Examples of arithmetic progressions are
# 
# $$\displaystyle 1+2+3+4+\cdots, \qquad   1+3+5+7+9+\cdots$$
# or in general
# 
# $$\displaystyle A = a_1 +a_2 +a_3 +a_4 +\cdots =a_1 +(a_1 +d)+(a_1 +2d)+\cdots+(a_1 +(n-1)d)$$
# 
# where the constant additional term is $d$. The sum of the progression can be obtained by regrouping as
# 
# $$\displaystyle \sum A = na_1+ \sum (n-1)d$$
# 
# and then recalling that the sum of $m$ consecutive numbers, $1,2,3$... is $m(m+1)/2$ thus $\sum (n-1)= n(n-1)/2$  making
# 
#  $$\displaystyle \sum A= \frac{n}{2}\left( 2a_1+(n-1)d\right)$$
# 
# ### **Geometric progression**
# 
# In the geometric progressions below, in $G_1$, each term after the first is multiplied by $1/2$, in $G_2$ the multiplier is $4$ and is $x$ in $G_x$.
# 
# $$\displaystyle \begin{align}
# G_1 &=1+1/2+1/4+1/8+\cdots \\
# G_2 &=1+4+16+64+\cdots \\
# G_x &=a+ax+ax^2 +ax^3 +\cdots +ax^{n-1} +\cdots =a(1+x+x^2 +x^3 +\cdots + x^{n-1} +\cdots)\end{align}$$
# 
# The first of these three series, $G_1$, converges to a finite value even after an infinite number of terms, and the sum of the terms to infinity is also finite. These series are discussed further in chapter 5.
# 
# The spectrum of atomic hydrogen and other atoms with one outer electron show an emission spectrum that converges to a limit. In the ultraviolet part of the spectrum, the lines are called the Lyman series and have the form
# 
# $$\displaystyle \bar \upsilon = R\left( 1-\frac{1}{n^2} \right) \qquad \mathrm{cm^{-1}}$$
# 
# where $n$ is an integer greater than $1$ and $R$ is the Rydberg constant, $109678$ wavenumbers for hydrogen. The positions of the lines are sketched in figure 25. The symbol $\bar \upsilon$ represents the frequency in wavenumbers or $\mathrm{cm^{-1}}$. These are habitually used by spectroscopists rather than $\mathrm{s^{-1}}$ (or Hertz) because the numbers are smaller, a few thousands of wavenumbers correspond to $\approx 10^{13} \to 10^{15} \mathrm{s^{-1}}$, where $ 1 \mathrm{cm}^{-1} \equiv 3 \cdot 10^{10} \mathrm{s^{-1}}$.
# 
# ![Drawing](chapter1-fig25.png)
# 
# Figure 25. The Lyman series converges to a finite value.
# 
# ____

# ## 13 Estimation
# 
# Estimating quantities is sometimes relatively easy to do and can often be used to help design experiments. An accurate number is not sought; just an order of magnitude estimate is often sufficient. A straightforward example is to estimate the minimum useful concentration of a sample knowing the sensitivity of an instrument. This can be put the other way around. At the maximum allowable sample concentration, we can estimate what the size of the signal will be and whether the method proposed will work or whether another more sensitive instrument should be used. Other forms of estimation can be used to obtain quantities that would otherwise seem impossible, such as calculating the weight of a mountain or the weight of the atmosphere. In these types of problems, we make some simplifying, but not unreasonable assumptions to be able to reach a sensible conclusion; the mountain could reasonably be approximated by a cone or a hemisphere depending on its shape. Several interesting estimation examples are given by Adam (2003).
# 
# ### 13.1 X-rays
# In an X-ray experiment on crystalline Be, suppose that the detector (e.g.a CCD type camera) can respond to single X-ray photons; estimate the least number of electrons that there must be in the crystal's scattering plane for a signal to be detected. What does this imply for the size of the crystal? (Assume that the scattering from $n$ electrons is $n$ times that from a single one and the following data is assumed to be known.)
# The fractional intensity, $I/I_0$, of X-ray scattering by a single electron is approximately given by
# 
# $$\displaystyle \frac{I}{I_0}= \frac{e^4}{(4\pi\epsilon _0)^2m^2c^4}\frac{1}{R^2}\frac{(1+\cos^2(\phi))}{2}$$
# 
# where $I_0 = 10^8$ is the initial number of X-ray photons per second, $R$ the distance from the centre of scattering to a detector, and $\phi$ the angle through which the X-ray is scattered. The detector is $1$ m from the sample and at an angle $\phi = 20^{\mathrm{o}}$.
# 
# In this example, although calculating $I/I_0$ is straightforward it is not if you use a calculator because the powers produced are too large. The calculation is done by first adding the powers, then the remaining numbers are easily evaluated.
# 
# $$\displaystyle \begin{align}\frac{I}{I_0}&=\frac{(1.6\cdot 10^{-19})^4}{(4\pi \cdot 8.8\cdot 10^{-12})^2(9.1\cdot 10^{-31})^2(3\cdot 10^8)^4\cdot 1  }\left( \frac{1.8}{2}  \right)\\
# &=\frac{16^4\cdot 10^{-4}}{4^2\pi^2\cdot 8.8^2 \cdot 9.1^2\cdot 3^4\cdot 2} 10^{-76+24+62-32}\cdot 1.8\\[10px] &\approx 10^{-29}\end{align}$$
# 
# If the initial X-ray intensity is $10^8$ photons/sec then $\approx 10^{-21}$ X-rays are detected from each electron, which means that $10^{21}$ electrons have to be present to detect one X-ray. As Be contains four electrons, $(1/4) \cdot 10^{21}$ atoms are needed, which is $10^{21}/(4 \times 6 \cdot 10^{23}) \approx 4\cdot 10^{-4}$ moles. The molar mass is $9$ g mol$^{-1}$, thus, to detect 1 photon/second, the sample has to weigh at least $0.0036$ g and as the density of Be is $1.8\;\mathrm{g\; cm^{-3}}$ this is a cube of volume $0.0036/1.8$ cm$^3$ which has a side of $\approx 0.13$ cm which is not large.
# 
# ### 13.2 Mass of atmosphere
# Sometimes, a particular estimation seems to be very hard but it turn out not to be. To calculate the mass of the atmosphere would appear to be difficult because of the changing density with increasing altitude. However, if atmospheric pressure is taken at sea level to be $1\; \mathrm{atm} = 101325$ Pa all that is needed is the surface area of the earth and the fact that a pascal is a N m$^{-2}$ which is force/area. Assuming that the earth is a sphere of radius $6378$ km, its surface area is 
# 
# $$\displaystyle 4\pi \times 63782 \times 1000^2 = 5.1 \cdot 10^{14} \mathrm{m}^2$$
# 
# and the total force on its surface due to the atmosphere is 
# 
# $$\displaystyle 101325(\mathrm{Pa}) \times 5.1 \cdot 10^{14} (\mathrm{m}^2) \approx 5.18 \cdot 10^{19}\;\mathrm{ N (\;kg\, m\, s^{-2}})$$ 
# 
# The atmospheric mass is therefore $\approx 5.18 \cdot 10^{19}/g = 5.27 \cdot 10^{18}$ kg where $g$ is the acceleration due to gravity $9.81$ m s$^{-2}$.
# 
# ### 13.3 Benjamin Franklin and the pond on Clapham Common
# 
# The American Benjamin Franklin is famously remembered for flying a kite with an attached wire into a thunderstorm in an attempt to capture lightning and store it in a Leyden jar (Bernal 1973). Luckily, he survived. He is less well known for changing the florid opening of the American Constitution to read 'We hold these truths to be self - evident, that all men are created equal'. He travelled to England in 1757 to help with the tax situation of Pennsylvanians, and while sailing noticed that oil on water calmed waves, something that had been noticed since the time of the Greeks. Being interested in surface phenomena, when again in London in 1770, he poured no more than a teaspoon of oil on a pond on Clapham Common and observed that the oil spread of its own volition over the pond, sweeping leaves and other debris out of the way as it did so. He estimated that the oil covered about half an acre of the pond ($2023\;\mathrm{m^2}$ ), and he repeated this experiment everywhere he went from then on. He also recorded seeing colours due to thin film interference effects but could not have explained this, nor could he have realized that a monolayer of molecules was formed because the idea of atoms and molecules was not clearly understood at that time. Surprisingly, he appears not to have worked out the thickness of the oil layer. 
# 
# To calculate the monolayer thickness, a teaspoon holds approximately 4.5 ml and half an acre is $2023\;\mathrm{ m^2}$. From the amount of oil used, the thickness of the layer was $4.5\cdot 10^{-6}\; \mathrm{m}^3/2023\; \mathrm{m}^2 =2.2$ nm. This is approximately the thickness of a close-packed, one-molecule thick layer of a long chain fatty acid monolayer, and gives a simple and direct estimate of the length of the oil molecules used. Oleic acid is a major constituent of olive oil and is a by-product of making candles; it has an 18-carbon chain so when fully extended would be about 2.8 nm long, remarkably similar to our estimate.
# 
# ### 13.4 Size of atom
# If the density of Li is $0.534\,\mathrm{ g\, ml^{-1}}$, what is average size of an atom? A mole contains $N=6.022 \cdot 10^{23}$ molecules, therefore $6.94$ g (the molar mass) contains this number of atoms each on average occupying a volume of
# 
# $$\frac{0.00694\,\mathrm{kg\,mol^{-1}})}{534\, \mathrm{ kg \,m}^{-3}N\mathrm{mol^{-1}}} = 2.16 \cdot 10^{-29} \mathrm{m}^3$$
# 
# or $2.16 \cdot 10^{-2}\, \mathrm{nm}^3$. Assuming spherical atoms, each has a radius $0.17$ nm which is not a bad estimation considering the approximate nature of the calculation. The similar calculation for Pb produces $0.19$ nm.

# In[ ]:




