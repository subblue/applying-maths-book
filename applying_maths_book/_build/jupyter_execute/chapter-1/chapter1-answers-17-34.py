#!/usr/bin/env python
# coding: utf-8

# # Solutions Q17 - 34

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
init_printing()                      # allows printing of SymPy results in typeset maths format
plt.rcParams.update({'font.size': 16})  # set font size for plots


# ## Q17 answer
# There are $5! = 120$ ways of arranging the numbers. They could all be listed, starting with $12345$, and ending at $54321$ as the largest and counted off. This is possible, but tedious. Instead, think of the numbers as five groups of four digits each, containing $4! = 24$ elements. If $x$ is a digit, the first ordered group of $24$ numbers starts with $1xxxx$, the second, $2xxxx$ starts at the $25^{th}$ number in the list and so forth. The $33^{rd}$ number is therefore in the second group. Next, split the numbers $xxxx$ into four groups of three digits making $3! = 6$ numbers each, and call them $yyy$. The first sets of numbers in group $2$ are $21yyy, 23yyy, 24yyy$ etc. The $33^{rd}$ can be found now just by enumerating, although the $yyy$ could be split into three groups of $2!$ numbers each producing $213zz, 231zz, 241zz$ etc. Putting in the values for $zz$ starting at the $25^{th}$ number gives
# 
# $$\begin{array}\\
# 21345,&  21354,&  21435,&  21453,&  21534,&  21543,\\ 23145,& 23154,& 23415, & 23451, & 23514, & 23541\\
# \end{array}$$
# 
# of which $23415$ is number $33$ in the list. This method could be made into an algorithm and used to find any given number in an ordered list of say $20!$ numbers. All these numbers could not possibly be listed, as each is $19$ digits long; it would also be quite a task to sort them into increasing order.
# 
# ## Q18 answer
# 
# (a) The fraction of $^{13}$C is $(12.011 - 12)/12 = 0.011$ or approximately 1/91. 
# 
# (b) The chance that both carbons are $^{13}$C is $(1/91)(1/91) = 1/8281$. 
# 
# (c) One $^{13}$C appears with a chance $1/91$ and there is a $1 - 1/91= 90/91$ chance that it does not appear. Two $^{13}$C have a chance of $(90/91)^2$ that they do not occur and therefore $1 - (90/91)^2 = 181/8281$ is the chance that a $^{13}$C appears at least once. Alternatively, we can argue that if at least one carbon is $^{13}$C, the chance of this occurring is the sum of the chance that carbon $1$ is $^{13}$C or atom $2$ is $^{13}$C less the chance that both carbons are $^{13}$C.
# 
# This is $\displaystyle \frac{1}{91}+\frac{1}{91}-\frac{1}{91^2}=\frac{181}{8281}\approx 0.0219 $. 
# 
# The relative size of the NMR signal is this result multiplied by the cube of the ratio of magnetogyric ratios, which is $\approx 3.5 \cdot 10^{-4}$ making it clear why $^{13}$C spectra are more difficult to obtain than $^1$H. Furthermore, as the signal to noise increases only as $\sqrt{n}$ for $n$ scans, this makes obtaining good quality $^{13}$C spectra very time consuming.
# 
# ## Q19 answer
# (a) This is effectively the same problem as tossing two coins. The four possible outcomes are 'AA', 'Aa', 'aA',and 'aa' of which the middle two are the same, therefore, the incidences are $p^2,\; 2p(1 - p)$, and $(1 - p)^2$ respectively.
# 
# (b) The chance that an individual is 'aa' is is the fraction 
# 
# $$\displaystyle \frac{(1-p)^2}{p^2+2p(1-p)+(1-p)^2}=(1-p)^2$$
# 
# and the joint chance that an individual has both 'A' and 'a' genes  $2p(1 - p)$, then, by equation 24, the probability of not being 'aa' but having an 'a' gene is $2p(1 - p)/(1 - p)^2$.
# 
# ## Q20 answer
# (a) $p=V_A/(V_A +V_B)$. 
# 
# (b) For $N$ molecules,the chance of all molecules being in volume $V_A$ becomes a product of individual ones since one molecule does not influence any other. The result is $\left(V_A/(V_A + V_B)\right)^N$. For equal volumes and only $100$ molecules, the probability is extremely small $p = 1/2^{100} \approx 8 \cdot 10^{-31}$. For $1000$ molecules, this probability becomes infinitesimal $\approx 10^{-301}$. If a mole of gas molecules were present this chance is so small that it has been effectively zero since the universe began and is likely to be so until it ends. 
# 
# (c) In this question the initial volume is $V_A$ and expansion is into the total volume, thus the chance is the inverse of that in (a). The entropy change is positive and is 
# 
# $$\displaystyle S=Nk_B\ln\left( \frac{V_A + V_B}{V_A} \right)$$
# 
# i.e. the gas expands spontaneously to fill the whole volume. 
# 
# ## Q21 answer
# (a) $4^3 = 64$ possible amino acids. 
# 
# (b) $20^{100} \approx 1.3 \cdot 10^{30}$ proteins.
# 
# ## Q22 answer
# As the objects are distinguishable, label them $A$ and $B$. The arrangements are,
# 
# $(AB,-,-),\; (-AB-),\; (-,-,AB),\; (A,B,-),\; (A,-,B), \;(-,A,B),\;(B,A,-), \;(B,-,A),\; (-,B,A).$
# 
# There are nine of these, six if the particles are indistinguishable bosons, and three if fermions.
# 
# ## Q23 answer
# The number of spin states is $6$ for p and $10$ for d orbitals because electrons are fermions. The number for the N atom is the combination $C(6, 3) = 6!/(3!3!) = 20$ and $C(10, 6) = 10!/(6!4!) = 210$ for the Fe atom.
# 
# ## Q24 answer
# There are $n=5$ degenerate orbitals making $10$ possible spin orbitals allowing for 'spin up' and 'spin down', and $p = 1 \to 10$ electrons so the calculation is 
# 
# $$\displaystyle C(2n, p) = \frac{(2n)!}{p!(2n-p)!}$$
# 
# and the number of microstates is shown in the table..
# 
# $$\displaystyle \begin{array}{cccc}
# \hline 
# \text{configuration } p= &0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10\\[5px]
# \text{microstates}      & 1 & 10 & 10!/(2!8!)=45 & 120 & 210 & 252 & 210 & 120 & 45 & 10 & 1\\
# \hline \end{array}$$
# 
# It makes sense that, with one electron there are $10$ ways of placing it in the orbitals, and similarly there is only one microstate when all orbitals are empty or filled; d$^0$ and d$^{10}$. The symmetry in the number of microstates is also clear; we can count electrons up to d$^5$ or equivalently 'holes' from d$^5$ to d$^{10}$. Using Python the number can be calculated as shown below

# In[2]:


def fact(n):                # recursively calculate factorial
    if n == 0 or n==1:
        return 1
    else:
        return n*fact(n-1)
#-------------
n = 5
for p in range(2*n+1):      # 1 added as range is otherwise 0 to 2n-1
    print( int( fact(2*n)/(fact(p)*fact(2*n-p)) ) ,end='  ')  # int makes integer result.


# ## Q25 answer
# The vibrational quanta are bosons because a vibration can have any number of quanta (see Section 9.4; the quantum numbers are $n = 0, 1, 2, 3 \cdots $ etc.) The number of states a doubly degenerate vibration produces is $C(2 + 5 - 1, 5) = 6$, which can be labelled as $(5,0),\; (0,5), \;(4,1),\; (1,4), \;(3,2), \;(2,3)$, and in the triply degenerate one $C(3 + 5 - 1, 5) = 42$.
# 
# ## Q26 answer
# The left-hand state is produced  $5!/(4!1!)=5$ ways, (see Section 9.2), the middle distribution in $5!/(2!1!1!1!) = 60$ ways, and the right-hand one in $5! = 120$ ways so, the probabilities are in the ratio $1 : 12 : 24$. The empty levels do not change the result because $0! = 1$ and are not shown in the calculation.
# 
# ## Q27 answer
# (a) The fraction of isotopes in the product is the same because each molecule is distinguishable therefore the equilibrium constant is 
# 
# $$\displaystyle K=\frac{(f_Hf_D)(f_Hf_D)}{(f_H^2)(f_D^2)} = 1$$ 
# 
# (b) The fraction $\mathrm{^{35} Cl }$ in $\mathrm{P^{35}Cl_3}$ is $f_{35}$ and $f_{37}$ in $\mathrm{P^{37}Cl_3}$. In the product each can have three ways of arranging the isotopes; for example $35-35-37,\, 35-37-35,\, 37-35-35$. Hence the fraction is $3f^2_{35}f_{37}$ and $3f^2_{37}f_{35}$ and the equilibrium constants is 
# 
# $$\displaystyle K=\frac{(3f^2_{37}f_{35})(3f_{35}^2f_{37})}{(f_{35}^3)(f_{37}^3)} = 9$$
# 
# (c) In this case, the product molecules each have six ways of distributing the isotopic atoms, or $4!/(2!2!)$, and the equilibrium constant 
# 
# $$\displaystyle K=\frac{(6f^2_{37}f^2_{35})(6f^2_{35}f^2_{37})}{(f_{35}^4)(f_{37}^4)} = 36$$
# 
# If you are not convinced, label the chlorine isotopes A and B and work out the number of arrangements of 2A and 2B atoms such as AABB, etc.
# 
# ## Q28 answer
# The important words here are 'at least' which means that we look for the chance that the event does not happen. The chance of _not_ observing one six is $5/6$ and in four throws, as they are independent, is $(5/6)^4$. Therefore, the chance of observing at least one six in four throws is $1 - (5/6)^4 = 0.518$ , which is slightly better than  evens or $50$% chance of winning. 
# 
# On the other hand, the chance of observing two sixes in $24$ throws is $1 - (35/36)^{24} = 0.491$ and, as there is less than a $50$% chance of winning, money will inevitably be lost. Had the Chevalier insisted instead on gambling that two sixes would be produced in $25$ throws he would have, perhaps, not lost his money. He is remembered because he asked Pascal to help work out out the odds of success and failure and so helped to found probability theory.
# 
# ## Q29 answer
# (a) The chance of at least one club being chosen can be found using equation 23 and is the chance of obtaining a club from each group less the joint chance which is $\displaystyle \frac{3}{5}+\frac{1}{4}-\frac{3}{5}\times\frac{1}{4}=\frac{7}{10}$.
# 
# (b) To obtain at least one ace the chance is $\displaystyle \frac{1}{5}+\frac{1}{4}-\frac{1}{5}\times\frac{1}{4}=\frac{8}{10}$.
# 
# (c) The chance of one card less than $4$ (assuming ace is high) is $\displaystyle \frac{4}{5}+\frac{3}{4}-\frac{4}{5}\times\frac{3}{4}=\frac{19}{20}$.
# 
# ## Q30 answer
# (a) This is effectively the problem of tossing two coins many times and looking for heads and tails. The Punnett square has the form
# 
# $$\displaystyle \begin{array}{lll}
# \hline 
#  & B & b\\
# \hline 
# B & BB & Bb \\
# b & Bb & bb \\
# \hline  \end{array}$$
# 
# which means that $3/4$ can taste the PTC because at least one allele in the offspring is B in $3$ of the $4$ outcomes. The ratio of dominant to recessive is $3:1$. 
# 
# (b) The chance that each of the first four children are non-tasters is $(1/4)^4 = 1/256$. All children $1/4$. 
# 
# (c) If the children are both B then the next generation are all tasters as both parents carry this gene. If the parents are heterozygous,B and b, the result is the same as in (a),and if both parents are b, all offspring are non-tasters.
# 
# ## Q31 answer
# Use the binomial distribution and the chance of any one digit being chosen is $1/10$; the chance of a run of zeros is the same as a run of any other digit. 
# 
# (a) The chance of choosing five zeros one after the other is 
# 
# $$\displaystyle p(10,5,1/10)= \frac{10!}{5!5!}\left(\frac{1}{10} \right)^5\left(\frac{9}{10} \right)^5$$
# 
# which is relatively easy to calculate, giving $3720087 / 2500000000 = 0.00148$. 
# 
# (b) The chance this time is 
# 
# $$\displaystyle p(100,50,1/10)= \frac{100!}{50!50!}\left(\frac{1}{10} \right)^{50}\left(\frac{9}{10} \right)^{50}$$
# 
# and is hard to calculate, but is clearly very small. Taking logs so as to use the Stirling approximation, 
# 
# $$\displaystyle \ln(n!) = n \ln(n) - n$$
# 
# gives 
# 
# $$\displaystyle \begin{align}\ln(p)&=\ln(100!)-2\ln(50!)+50\ln\left(\frac{1}{10}\right)+50\ln\left(\frac{9}{10}\right)\\&=100\left(\ln(100)-\ln(50) \right)+50\left(\ln\left(\frac{1}{10}\right)+\ln\left(\frac{9}{10}\right) \right)\end{align}$$
# 
# which, when the exponential is taken, gives a chance of $6.53 \cdot 10^{-23}$, which is miniscule. The exact calculation gives $5.2 \cdot 10^{-24}$ and, as might be expected, is different because $n!$ is not approximated.
# 
# (c) In this case, the Stirling approximation is needed, and gives $1.4 \cdot 10^{-222}$, which is definitely a zero chance in practice of finding a run of five hundred zeros as might be expected if the numbers are random. The exact calculation gives $3.5 \cdot 10^{-224}$.
# 
# ## Q32 answer
# (a) Using the binomial distribution, the chance or fraction is 
# 
# $$\displaystyle p(6,4,1/2)=\frac{6!}{4!2!}\left(\frac{1}{2} \right)^4\left(\frac{1}{2} \right)^2= \frac{15}{64}\approx 0.23$$
# 
# and the chance of not having four girls is $49/64$. The odds are therefore $15:49$ against having four girls in a family of six children. 
# 
# (b) If the odds are no longer 1/2 but 3/5, the fraction is now 
# 
# $$\displaystyle p(6,4,1/2)=\frac{6!}{4!2!}\left(\frac{3}{5} \right)^4\left(\frac{2}{5} \right)^2= \frac{927}{3125}\approx 0.31$$
# 
# ## Q33 answer
# (a) Start by evaluating both sides separately and the identity is proved.
# 
# $$\displaystyle q\binom{n}{q}=q\frac{n!}{q!(n-q)!}=\frac{n!}{(q-1)!(n-q)!}\\
# q\binom{n-1}{q-1}=q\frac{n(n-1)!}{(q-1)!(n-q)!}=\frac{n!}{(q-1)!(n-q)!}$$
# 
# (b) The right-hand side is 
# 
# $$\displaystyle  \binom{n-1}{q-1}+\binom{n-1}{q}=\frac{(n-1)!}{(q-1)!(n-q)!}+\frac{(n-1)!}{q!(n-1-q)!}$$
# 
# Next, as $n! = n(n - 1)!$, substituting $n \to n - q$ gives $(n - q)! = (n-q)(n-q-1)!$. Substituting $(n - 1 - q)! $ into the equation gives
# 
# $$\displaystyle  \binom{n-1}{q-1}+\binom{n-1}{q}=\frac{q(n-1)!}{q!(n-q)!}+\frac{(n-q)(n-1)!}{q!(n-q)!} = \frac{n!}{q!(n-q)!}$$
# 
# ## Q34 answer
# Using the method in the text, $100759-86$ has the sum $1 \times 6 + 2 \times 8 + 3 \times 9 + 4 \times 5 + 5 \times 7 + 8 \times 1 = 112$ and $112\,\mathrm{mod}\, 10 = 2$ which is the checksum. The CAS number $116-31$ has a sum $34$ and so a check digit of $4$ and $103-30$ has the sum $20$ and check digit $0$. The alternative registry number for pheophytin-A is $603-17-8$
