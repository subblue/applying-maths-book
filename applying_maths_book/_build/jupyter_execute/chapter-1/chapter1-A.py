#!/usr/bin/env python
# coding: utf-8

# # Numbers to Algorithms

# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
init_printing()                      # allows printing of SymPy results in typeset maths format
plt.rcParams.update({'font.size': 16})  # set font size for plots


# 
# ## 1 Symbols and basics
# 
# When you learn mathematics, it is normal to use $x$ and $y$ to do most of the algebraic manipulations. In science, $x$ and $y$ are almost never used and this simple change can make a simple equation look complicated. This is something that you just have to get used to, but is made easier if you think what the variables mean; then perhaps you will be confident, say, to differentiate with respect to temperature $T$ rather than feeling that this can only be done with variable $x$. How ingrained using $x$ and $y$ are, is not, however, to be underestimated, for even experienced practitioners will on occasion revert to these when faced with a difficult calculation.
# 
# ### 1.1 Equations
# 
# There is one fundamental rule for manipulating an equation and that is to make sure that whatever is done to one side is done to the other. The equality$^1$ sign = means that the left-hand side of any equation is always equal to the right; for example the ideal gas law equation
# 
# $$\displaystyle pV = nRT \tag{1}$$
# 
# means that the product of pressure $p$ and volume $V$ is always equal to the number of moles multiplied by the gas constant $R$ and the temperature $T$. To isolate the pressure on the left-hand side, both sides of the equation must be divided by the volume,
# 
# $$\displaystyle \frac{pV}{V} = \frac{nRT}{V}$$
# 
# which is then canceled through on the left giving $p = nRT/V$. To avoid making mistakes, the best approach is to do any manipulations step by step even if they seem trivial. Repeating a calculation because a slip has been made in an earlier step takes far longer than it does being thorough at each stage.
# 
# $^1$ The = sign was probably first used by Robert Recorde in an algebra textbook of 1557.
# 
# ### 1.2 Multiplication and division
# 
# Slightly different notations are used when mathematics is written by hand, printed in a book, or interpreted by a computer language. One instance is the different symbols used to indicate multiplication; for example, multiplying together three quantities can be written as $pqr,\; p \times q \times r,\; p \cdot q \cdot r$ or $p. q. r$. In the last two examples, the point could be confused with a decimal point if numbers were involved of if vectors as the dot product, see chapter 6. 
# 
# In most computer languages, including Python used in this text, multiplication is indicated by the symbol, *, for example, $\mathtt{a*b}$. Raising a number to a power is written as $\mathtt{p**3}$ or $\mathtt{p**(1/3)}$  for the cube root.
# 
# In mathematics, the brackets (), {}, [ ]  are each used to indicate multiplication, and if an equation dictates that brackets must be nested then different ones may be used for clarity. In a computer language, different brackets are usually used to indicate _different types_ of arrays or functions, but exactly which is used depends on the language.
# 
# Each term inside a bracket, is multiplied by any terms outside; for example in mathematical notation, 
# 
# $$\displaystyle a(b - c) = ab - ac \quad \text{and}\quad (a - d)(b - c) = ab - ac - db + dc$$ 
# 
# If a bracketed expression is multiplied by itself then this is written as, for example, $(a-d)^2 =(a-d)(a-d)=a^2 - 2ad  + d^2$.
# 
# In Python and other languages, $a(b - c)$ would be written as $\mathtt{a*(b - c)}$. The reason for the multiplication sign is that one or more letters, such as $a$, followed by a bracket could indicate a function, array, vector, or matrix. Which one depends on how the name, in this case, $a$, was defined.
# 
# In printed equations, division is sometimes ambiguous because it is not clear what is to be divided; for example, $a/bc$ can mean $\displaystyle \frac{a}{bc}$ or $\displaystyle \frac{a}{b} c$, which is $(a/b)c$; it depends on the convention being used as to which operation takes precedence. The safest way, although a little clumsy, is to write $a/(bc)$, making it clear that $a$ is divided by $bc$. In Python and other computer languages, $a/b*c$ means $(a/b)*c$, so adding brackets is essential to get the calculation you expect.

# ### 1.3 Approximation: never do this in the middle of a calculation
# 
# Truncating numbers in  the middle of a calculation is an easy way to introduce numerical errors, the final result only should be rounded and if it related to an experimental value then the extent of this will be determined by the standard deviation or uncertainty in the value. In some calculations a similar type of approximation can be made algebraically, but this is also risky.   
# 
# Sometimes when one variable is small compared to another it can be ignored. The expansion of the exponential is one such case where this is legitimate; 
# 
# $$\displaystyle e^{-x}=1 - x + \frac{x^2}{2!}-\frac{x^3}{3!}\cdots$$ 
# 
# The terms $2!$ and $3!$ are factorials; $3!=3\times 3\times 1$ and similarly for other numbers.  Suppose that $x$ is small compared to  $1$ then $x^2$ is very small and $x^3$ tiny, then it is possible to approximate $\displaystyle e^{-x} \approx 1 - x$ and this is valid. However, to do this in the middle of a calculation is invariably fatal. Take a simple example, in the expression 
# 
# $$\displaystyle y = x+\frac{N-x^2}{2x}$$
# 
# suppose that $x\lt N$. It would be wrong to make this $\displaystyle y = x+\frac{N}{2x}$ by arguing that as $x^2$ is much less than $N$, it can be ignored, because when it is expanded out the expression becomes 
# 
# $$\displaystyle y=\frac{1}{2}\left(  x+ \frac{N}{x}\right)$$
# 
# and no comparison of the relative sizes of $x$ and $N$ is possible.
# 
# ### 1.4 Assignments in Python and other computer languages.
# 
# In Python when something has to be calculated, it can be typed directly for example, $\mathtt{2.5*6}$; or $\mathtt{np.sin( np.pi/6 )}$ where the prefix $\mathtt{np.}$ tells Python to use the numpy numerical library. (This has to be loaded initially, see the header to this page or the appendix on using Python ). The result can be see by printing to the screen as $\mathtt{print(np.sin( np.pi/6 ) )}$.  
# 
# However, if the result is needed later on, then the calculation must be assigned a name. This name could be almost anything but is usually a letter and numbers, $\mathtt{f01,\, result3}$ and so forth, but not a name such as $sin,\, log$ etc. as these are reserved names.  The assignment operation is the equal symbol = , so, $\mathtt{thesine= np.sin(np.pi/6)}$ would produce $\mathrm{\sin(\pi/6)}$ every time $\mathtt{thesine}$ is typed; thus, $\mathtt{6*thesine}$ is the same as $\mathrm{6 \sin(\pi /6)}$. The assignment does not produce an equation; it makes the _name_ on the left of the = sign produce the calculation shown on the right. Thus a statement such as
# 
# $$\displaystyle \mathtt{eqn1 = x**2 + 3*x - 2}$$
# 
# is acceptable but not $\mathtt{eqn1 = x**2 + 3*x = 2}$.  Details of Python syntax is shown in Appendix 1. Because many different symbols are used in maths, and some only infrequently, these are listed in the Glossary with their meanings.
# 
# ### 1.5 Functions
# 
# The equation $y = 1/x$ may be written in functional form as $f(x) = 1/x$, thus, if $x = 2$ the function has the value of 1/2, or $f(2) = 1/2$. The function operates on its argument $x$, which here is $2$, to produce its reciprocal. The $x$ in the function, either as its argument or in the body of the function is, however, simply a symbol, and is arbitrary; it is the algebraic expression or operation itself that determines what happens. Therefore, writing $f(w) = 1/w$ is entirely the same as $f(x) = 1/x$; if a function is evaluated with a series of numbers then it does not matter whether $w$ or $x$ was used to represent the mathematical operation; the result is the same. Similarly, as $x$ is only symbolic, it is possible to write a composite function where the argument is itself a function; e.g. $f (x^2) = 1/x^2$ or $f (e^x) = 1/e^x$, which means that the function $f$ _operates_ on $x^2$ or $e^x$ to produce its reciprocal. Formally, this would be written as $f[g(x)] = 1/g(x)$ where $g(x)$ is the 'inside' function.
# 
# You can consider a function to be a _rule_ that converts $x$ or $w$, or whatever symbol is convenient to use, into some other mathematical form. You can also consider it as a _mapping_ that transforms $x$ into a new form. The function 'log' follows the rules to make the log of a number, and is written as $f(x) = \ln(x)$, and formally the range $x \ge 0$ should be added, but usually this is assumed to be known. A different example is illustrated by the functions used to find the real and imaginary part of a complex number. Complex numbers $z$ have two parts; the 'real' and the 'imaginary', which is multiplied by $i =\sqrt{-1}$; for example, $z = 2 + 3i$, see Chapter 2. The real function is written $f(z) = Re(z)$ and the imaginary function $f (z) = Im(z)$ and if $z = 2 + 3i$ then $Re(z) = 2$ and $Im(z) = 3$.
# 
# While some functions are 'even' and others 'odd', many are neither. 'Even' functions have the property that they do not change sign when their argument changes sign, $f(x) = f (-x)$, but 'odd' functions do change sign $f (x) = -f(-x)$; $x^2$ is an even, $x^3$ an odd function. A function's odd or even properties are very useful in evaluating integrals.
# 
# Inverse functions also exist, which means that if a function $f(x)$ changes $x$ to something else then the inverse function $f^{-1}(x)$ changes it back. Suppose that the function $T_F$ changes temperature in degrees centigrade ($^{\text{o}}$C) to Fahrenheit ($^{\text{o}}$F), $T_F(C) = 9C/5 + 32$; the inverse, $T_C$, takes $^{\text{o}}$F and converts it into $^{\text{o}}$C as $T_C(F) = 5(F - 32)/9$. This last function could be written as $T_F^{-1}$(F) because it is the inverse function. Finding the inverse function can be quite involved and is described in Section 1.6.
# 
# To summarize, a function transforms its argument into some other mathematical object, which can be of almost every possible type. The principle is the same whether the function produces the reciprocal, or square or an integral or a series and so forth. The function may also be called a transform, an operator, or a mapping of the argument ($x$) onto some other object.
# 

# ### 1.6 Functions in Python
# 
# In Python and other languages, a function has a particular syntax that is related to that used in mathematics, but with some important differences. A function is first defined using an assignment statement, then, when used it has a pair of brackets added to it. The function $f(x)= 2x+3x^2$ is defined in Python as
# 
# $$\displaystyle f = \text{ lambda x}: 2*x+3*x**2$$
# 
# and this is called a 'lambda' function and used as when the code can be placed on a single line. It is used as  f(2) which returns a value of 16 etc. If the function has two arguments then this is written as for example $f = \text{lambda x,y}: 2*x+3*y**2$.
# 
# The other type of 'function' is a sort of subroutine, to give it an old fashioned name, and has the form <br>
# 
# $ \mathtt{def\;\; afunc( x, y ):}\\
#   \quad  \mathtt{a= 2*x+3*y**2}\\
#    \quad \mathtt{return\;\; a}$
#    
# and is used as 
# 
# $\mathtt{result= afunc(2,3) }$
# 
# ### 1.7 Algorithms
# 
# There is no one definition of an algorithm. We can consider this as a logical, fixed set of rules for solving a problem that may or may not be mathematical. Starting with a known set of initial conditions, an algorithm proceeds to some other known set of conditions that ends the process and does so in a finite number of steps. A game could be considered to follow an algorithm in that it follows a fixed set of rules and starts and ends in a definite way. An algorithm could be the sequence of operations undergone to determine the rotation, reflection, and other symmetry properties of a molecule, or it could simply be the steps producing an algebraic or numerical calculation. This leads to a more restrictive and more common definition of an algorithm as a series of instruction with which to perform a calculation on a computer.
# 
# 
# ### 1.8 Numbers
# 
# So familiar are numbers and counting that we hardly give them much thought and automatically use different types of numbers and ways of counting as the situation dictates. Integer numbers are either prime, when the number is only divisible by itself and 1, or composite and the product of two other integers. Besides the positive and negative integers, there are the rational numbers or fractions, generated from their ratios, $n/m$, such as $1/2,\, 2/3$, etc. A fraction is called proper when the numerator is less than the denominator, $2/3$ for instance, and a fraction improper if the numerator is greater.
# 
# It is common to express fractions as decimal expansions such as $0.500$, or $1.36348 \cdots$ etc., which are called real numbers. In some countries, real numbers are expressed with a comma $0,500$ or $1,36348$ instead of a decimal point, and the comma is the international system of units (SI) recommended symbol, although it is never used in the English-speaking world.
# 
# In fractions, such as $98/77$, there are repeats of the sequence of digits; this fraction has a $2$ digit repeat and is $1.272727 \cdots$ while $98/78 = 1.256410256410 \cdots$ has a 6 digit repeat. The irrational fractions can also be expanded in decimal notation, $\sqrt{2} \approx 1.4142135 \cdots$ , however, the sequence of digits does not repeat itself.
# 
# The decimal numbers, as the name implies, use a base of $10$ for counting, however, we are also quite happy using other counting systems; base $60$ to count time, with $60$ seconds in a minute and $60$ minutes in an hour, as well as base $12$ or $24$ for counting the hours. Angles such as latitude and longitude, essential for navigation, are based on the $360^\text{o}$ of a circle with the degrees each split into $60$ minutes and then into $60$ seconds of arc. Engineers used to use _grads_, which divide the circle into $400$ parts, and perhaps some still do. In measuring weights, the old Imperial units, pounds (lb) and ounces (oz), where $16$ oz = $1$ lb and $14$ lb = $1$ stone are still used, although this is being replaced by the decimal units based kg and g. Distances in the UK and in North America are measured in miles, an arbitrary measure of $5280$ ft, based on a yard of $3$ ft with each foot containing $12$ inches, although the km, m, and cm are replacing these older measurements. Eight km is approximately $5$ miles and $1$ inch $\equiv 25.4$ mm. A distance of $3$ yards, $1$ ft and $4$ inches is often written as $3,\, 1'\, 4''$ and a similar notation used for degrees, minutes, and seconds of arc, $25^\text{o}\, 12''\, 3''$.
# 
# Because computers are now so fast in performing calculations, it is almost never necessary to write code at the binary level, which is base two. Binary means calculating at the bit and byte level; the numbers used are $0$ and $1$ only. In octal, numbers based on $0$ to $7$ only are used and in hexadecimal numbers are on base $16$. These are chosen by convention to have the numbers and letters, $0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, F$. Thus $10$ in decimal is $A$ in hexadecimal, $15$ in decimal is $F$ in hex and $16$ in decimal is $10$, pronounced as '1 zero', in hex. 
# 
# It is not difficult to do arithmetic in hexadecimal, just awkward because we are so used to decimals and often change to decimal to do the calculation and convert back, this being easier than learning the hexadecimal multiplication table. Hexadecimal arithmetic is used only rarely now as programming a computer is usually done at a high level. Once 'hex' was necessary not only to get speed in a calculation on a computer using words $16$ bits wide, but also to 'talk' to an external instrument via pulses generated by repeatedly sending a $1$ or $0$ to a certain address that is then mapped onto the pins of an output socket on the computer. The table shows a comparison of different number systems in the order top to bottom, decimal, octal, hexadecimal, and binary. In section 10 describing modulo arithmetic an algorithm for converting from decimal to other bases is described.
# 
# ![drawing](chapter1-fig0.png)

# ## 2 Integers, real, and irrational numbers
# 
# Integers are the numbers that are colloquially called whole numbers, $-2, 0, 3$, etc. and they extend from minus infinity to plus infinity; $-\infty$ to $+\infty$ and although there is an infinity of them they are what mathematicians call countable. Integers are clearly not the only numbers as we are familiar with the real numbers such as $1.2,\, 3.56$, and so on, of which there is also an infinite number, but many, many more than there are integers. If two different real numbers are chosen, an infinite number of others can be squeezed in between them simply by adding more decimal places; this and other interesting discussions on numbers and algorithms are explained clearly in 'The Emperor's New Mind' (R.Penrose publ. 1990 Vintage OUP). Surprisingly, real numbers can usually be reduced to a rational fraction which is a ratio of two integers, e.g. $0.751 = 326/533$, but, for some numbers, $\pi,\, e, \sqrt{2}$ there is no ratio of integers that will accurately form the number and they are called _irrational_, i.e. not being logical! You may recall that $\pi \approx 22/7$ , which would seem to contradict this statement, but $22/7$ is a very poor approximation to $\pi$, only accurate to three digits, $355/113$ is better but only to seven digits.
# 
# When you use a computer to perform calculations, a distinction is made whether a real or integer number is used. This is a common feature of programming languages and also in computer algebra, for example, the integer division $2//3$ is exact and in Python (notice the //), it remains in the calculation as such. In other languages integer division can result in the answer $2/3 = 0$ rather than $0.666^.$ or remaining as an exact ratio so this has to be checked carefully.
# 
# We look next at some of the irrational numbers as their calculation has fascinated mathematicians over the centuries and they allow us to use some interesting geometry and to write some algorithms.
# 
# ### 2.1 Pi ($\pi$ ) and an algorithmic way of calculating its value
# 
# Whole books have been devoted to the number $\pi \approx 3.1415926$, such as The Joy of Pi (Blatner 1999), and calculations performed to study the pattern of its digits, which now extend to many thousands of millions of decimal places. As far as is known, the pattern of digits does not repeat itself; if they did it would mean that $\pi$ could be written as a fraction of two whole numbers and would no longer be an irrational number. $\pi$ is also a transcendental number, which is one that is not the solution (we say root) of any polynomial equation, and it is not considered to be algebraic; it transcends algebra and this guarantees that it is irrational.
# 
# Approximations to $\pi$ are $22/7,\; 355/113, \; 52163/16604$ and many others. The scientist James Jeans used a sentence to remember the first 15 terms; 'How I want a drink, alcoholic of course, after heavy chapters involving quantum mechanics' where the length of each word represents a digit.
# 
# $\pi$ appears naturally in many algebraic formulae that have no obvious geometric interpretation. Some estimations involve summing series, others dropping needles across a square grid (Buffon, 1707-1788), and yet others, going back at least to Archimedes, measure either the length of one polygon that is just larger, or one that is just smaller, than a circle. The proof of this method is very elegant and leads us in a direction that is not obvious before coming to its conclusion. We shall follow it through and then write an algorithm with which to calculate $\pi$. We consider ourselves to be advanced and modern, but looking at the subtlety in this and other proofs, we soon realize how brilliant were those who initially worked them out.
# 
# 
# ![drawing](chapter1-fig1.png)
# 
# Figure 1. The hexagon has a side of length DB = $a_1$ the dodecagon CB = $a_2$.
# ____
# 
# To calculate $\pi$, the idea is first to draw a circle and then place a square inside this touching its circumference. Next, a pentagon, hexagon, and 'n-agons' are drawn in the same way, and as the number of sides gets larger, the closer the straight-sided polygon will get to the circumference of the circle. And, if another polygon is drawn just outside the circle, then the two values of the polygons' circumferences define upper and lower bounds to the value of $\pi$. However, actually drawing these polygons on a piece of paper, or the computer, is not necessary as geometry and algebra can be used. This was how the first calculation of an accurate value of $\pi$ was achieved, and furthermore, to 15 decimal places, and was described by the Iranian astronomer Jamshid al-Kashi in 1424 in his book 'A Treatise on the Circumference'.
# 
# The calculation starts by sketching a hexagon in the circle, then a 12-sided figure (a dodecagon). Just one segment of each is shown in Figure 1. The drawing need not be very accurate and only represents the shapes. Geometry is used to assert their true shapes.
# 
# Figure 1 shows the straight edge of one piece of the hexagon, DB = $a_1$ and a dodecagon CB = $a_2$. The plan al-Kashi had was to find $a_1$ and then to determine $a_2$ in terms of $a_1$ and then continue to a '24-gon' and find the length of one of its segments, calling this $a_3$, which is known in terms of $a_2$ and so on. By adding up the lengths of polygons with more and more sides, better and better estimates of $\pi$ are obtained.
# 
# It has been known since the time of Euclid, that for a triangle drawn in a circle the angle at the centre of a circle is twice that at the circumference; therefore, if $\angle$ DAB is $\theta$, then $\angle$ DOB is $2\theta$. An angle at the centre and circumference is shown in Figure 2:
# 
# ![drawing](chapter1-fig2.png)
# 
# Figure 2. The angle at the centre is twice that at the circumference.
# ______
# 
# Using this theorem also means that $\angle$ADB is a right angle, since if $2\theta = 180^\text{o}$ then $\theta = 90^\text{o}$ and $\angle$ACB is therefore also a right angle. You may also notice in Figure 1 that AD is parallel to OC, that $\angle$ADO is also $\theta$ and that OC and DB are at right angles since the arc DC = CB and therefore OC bisects DB.
# 
# Now we can start the calculation of $\pi$. Remember that we have to find $a_2$ in terms of $a_1$. To find $a_1$ we have to find the length of one side of a hexagon, BD. But in a hexagon, the total length of the six sides is easily found since each segment, such as ODB is an equilateral triangle, $2\theta = 60^\text{o}$, and if the circle has a radius of $1$, then OA and OB = $1$, since OC, OD and BD $\equiv a_1$ thus, $a_1 =1$.
# 
# The total length of the hexagon, which approximates $2\pi$ the circumference of the circle, is $6$ which produces the rather unimpressive value $\pi = 3$; this can be vastly improved on. Al-Kashc's brilliant reasoning is now outlined. 
# 
# His method was to find a relationship between $x_1$ (which is AD) and $x_2$ (AC), and use this to find $a_2$ (CB) knowing that $a_1 = 1$ and to repeat this process for polygons with progressively larger numbers of sides. To find $a_2$, two lines CZ and OY, are drawn that are, respectively, perpendicular to AB and AD as shown in Figure 3.
# 
# Next, convince yourself, for example by drawing them out, that the triangles ABC and ACZ are similar because their angles are similar. The angle ACB is $90^\text{o}$ because an angle at the circumference is half that at the centre; see Figure 2, and AZC is also a right angle. Therefore the ratio of the sides AZ/AC is the same as AC/AB or as an equation AZ/$x_2 = x_2$/AB from which
# 
# $$\displaystyle  x_2 =\mathrm{AZ \times AB} =2(1+\text{OZ}), \tag{2}$$
# 
# the length AB being $2$ because the radius of the circle is $1$. Our task now is to find OZ. 
# 
# The angles in triangles AOY and OZC are the same, see Figure 1, and the triangles are also the same size as both have the radius as their hypotenuse, AO and OC. Notice also that $\angle$AOY = $90 -\theta$ and therefore bisects the angle $\angle$AOD making AY = YD and AY = $x_1/2$ = OZ. Substituting for OZ produces
# 
# $$\displaystyle x_2^2 = 2 + x_1\tag{3}$$ 
# 
# A relationship between the length $x_1$ or $x_2$ and $a_2$ has to be found. Triangle ADB is right angled and as the hypotenuse is 2 (the diameter), using Pythagoras's theorem is 
# 
# $$\displaystyle  a_1^2 = 4 - x_1^2$$
# 
# and because $a_1 = 1$ then $x_1^2 = 3$. Using equation 3 produces $x_2$ as 
# 
# $$\displaystyle x_2 = 2 +  \sqrt{3} = 3.7320$$
# 
# The dodecahedron value for $\pi$ is calculated by using the right-angled triangle ACB to obtain $a_2^22 = 4 - x_2^2$. Hence $a_2 = 0.5177$, and as there are 12 sides to the dodecahedron and the circumference is $2\pi$, we obtain $2\pi = na_2$ or $\pi = 12/2 \times 0.5177 = 3.106$. Not too bad a result and accurate to one decimal place, however, the calculation can be repeated, doubling the number of sides in each step.
# 
# Using pairs of equations, the parameter $a_n$ becomes the approximation to $\pi$, when multiplied by $n$ the number of sides, e.g. using the value for $x_2$ which is based on $x_1$, then
# 
# $$\displaystyle  a_3=\frac{24}{2}\sqrt{4-x_3^2}\qquad \text{with }\qquad x_3^2 = 2+x_2 $$
# 
# The next approximation is
# 
# $$\displaystyle  a_4=\frac{49}{2}\sqrt{4-x_4^2}\qquad \text{with }\qquad x_4^2 = 2+x_3 $$
# 
# and so on making the $i^\text{th}$ step, with a n-sided polygon
# 
# $$\displaystyle  a_{n+1}=\frac{n_i}{2}\sqrt{4-x_{i+1}^2}\qquad \text{with }\qquad x_{i+1}^2 = 2+x_i $$
# 
# This calculation of $\pi$ obviously means that Al-Kashi knew how to evaluate square roots. His method is written in Python as Algorithm 1, if you are not familiar with the syntax then look at the Python appendix now.
# 
# 
# ![drawing](chapter1-fig3.png)
# 
# Figure 3. Geometric construction to calculate $\pi$. (Not drawn to scale). 
# 
# _____
# 
# Any algorithm is a process or 'machine' used to arrive at an answer and starts by defining the parameters needed, then setting their initial values. A termination criterion has also to be defined to stop the calculation. The calculation is then performed step by step, repeating steps as necessary and the results printed as the calculation proceeds or just at the end as necessary. If the calculation involves repeated evaluations, as in this case, then a loop is needed to do this.
# 
# In the Python code, note that the equal symbol = while looking like an equality sign in mathematics does indicate an equation but rather that an _assignment_ is being made; for example, the statement m= 20: means 'set aside some space in the computer's memory, give it the name $m$, and when that name is called it will produce the number 20'. 
# 
# The instruction $\mathtt{for \;i\; in \;range(m):}$  starts a  loop that repeats the calculation $m$ times starting at zero, automatically incrementing the value of $i$. In this example, this is done 20 times, which corresponds to using a $6$ by $226$-gon which has $402,653,184$ sides! 
# 
# The results are reproduced here, but it is clear that the true value of $\pi$ is being approached and the last approximate value listed is accurate to $12$ decimal places.

# In[2]:


# Algorithm. Al-Kashi's method of calculating pi

m = 20                              # number of iterations
xa = np.sqrt(3.0)                   # initial guess for x
na = 6.0                            # number of sided on polygon
print('{:d} {:s}{:10.8f}'.format( 0,'first guess ',3*np.sqrt(4 - xa**2 ) )  )
for i in range(m):
    xb = np.sqrt( 2 + xa )          # new value of x 
    nb = 2.0*na
    ab = nb/2.0*np.sqrt(4 - xb**2 ) # new estimate of pi
    print( i+1, ab)
    xa = xb                         # replace new for old
    na = nb
    pass                            # end loop


# An accurate value is $\pi= 3.141592653589793$ and you can see that our algorithm approaches this value but then deviates and becomes incorrect with iterations above 15. Precision is always a consideration when calculating to many decimal places it does not mean that the algorithm is failing but that the precision in the arithmetic used by Python is limited to $9 \to 10$ significant figures. There are ways to overcome this such as using the mpmath library to say $50$ decimal place precision. 
# 
# Al-Kashi repeated his calculation of $\pi$ until its error was the same ratio as the width of a hair is to the size of the universe. From historians' estimates of what was considered to be the size of the universe in the thirteenth century, this means that he calculated $\pi$ to $15$ decimal places, which is a genuinely extraordinary calculation and better in accuracy than our answer.
# 
# Such has been the fascination of $\pi$ to mathematicians that there are many algebraic formulae which contain $\pi$ and others with which to calculate it. Some examples are shown below.
# 
# $$ e^{i\pi}=-1, \qquad \int_0^\infty \frac{\ln(x)^2}{1+x^2}=\left(\frac{\pi}{2} \right)^2 ,\qquad \frac{\pi^6}{6}=\frac{1}{1^2}+\frac{1}{2^2}+\frac{1}{3^2}\cdots=\sum_{n=1}^\infty\frac{1}{n^2}+$$
# 
# In these and many other formulae producing $\pi$, it is difficult to appreciate why $\pi$ appears; some are far more efficient at calculating than the method described, but do not have the appeal of the geometric approach.
# 

# ### 2.2 Square Roots
# 
# The square root is familiar as the solution to an equation such as $n^2 = 3$. The square root is also the length of the diagonal of a square or rectangle and therefore has a geometric interpretation. The Pythagoreans discovered that the length of the diagonal of a square of unit side, i.e. length one, is $\sqrt{2}$; see Figure 4. Although it is hard to imagine this today, this calculation caused a scandal since this number was not 'pure' in the sense that it was not the ratio of two whole numbers. And the scandal still echoes down the centuries. Numbers that are not the ratio of two whole numbers are called irrational or 'un-ratio-able', the Latin word ratio coming from the Greek for 'logo' which means ratio. We use the word logic to symbolize reason and it is still a pejorative statement to describe someone as being irrational.
# 
# The right-angled triangle does not of itself produce the value of $\sqrt{2}$ very accurately, since the length of the line would have to be measured; it is more of a geometrical definition. The first calculation of a square root was made by the Babylonians (modern-day Iraqis), about four thousand years ago, and is inscribed in a clay tablet preserved at Yale University. A geometric method was used that could be converted into an algorithm. The result on the tablet was accurate to five decimal places and the calculation shows the Babylonians knew Pythagoras's theorem about a century before Pythagoras discovered it. This result is remarkable, and it is possible that a formula was used similar to what is now called Heron's method after the Alexandrian mathematician and engineer who lived about two thousand years ago; his dates vary from $150 \to 250 AD.
# 
# ![drawing](chapter1-fig4.png)
# 
# Figure 4 Unit right-angled triangle.
# ____
# 
# Suppose that $N$ is the number for which the square root is sought, then let $a$ be an estimate of this number and therefore we want an algorithm with which to find $a$. Suppose that $a_2$ differs from the true value $N$ by an error $\epsilon$; hence
# 
# $$\displaystyle \epsilon = N-a^2 \tag{4}$$
# 
# The next step is to find a better approximation and instead of just randomly guessing a new value for $a$, a small positive or negative number $c$ is added. Now we suppose that $a + c$ is the new and better approximation to the square root, therefore
# 
# $$\displaystyle \epsilon = N-(a+c)^2 \tag{5}$$
# 
# Assuming that adding $c$ makes the error we call $\epsilon_c$  zero, by expanding the bracket, substituting
# for $N - a^2$ and rearranging produces $\epsilon = 2ac+c^2$. However, as $c$ is small, then $c^2$ is even smaller,$^2$ and we will suppose we can ignore it and then $c = \epsilon /2a$. Next we suppose that the approximation to the square root is $a_1$ where $a_1 = a+c$. Hence
# 
# $$\displaystyle  a_1=a+\frac{\epsilon}{2a}= a +\frac{(N-a)^2}{2a}=\frac{1}{2}\left( a+\frac{N}{a} \right) \tag{6}$$
# 
# Repeating the procedure gives $a_2 = (a_1 + N/a_1)/2$ and then $a_3,\; a_4 \cdots$  and so forth, and and will eventually produce $\sqrt{N}$ after $n$ iterations. It is implicitly assumed that this sequence will converge to a real number as the error $\epsilon$ gets smaller.
# 
# To calculate $\sqrt{2}$, then $N = 2$, and starting with a poor initial guess of $a = 1/2$, the steps where $\epsilon$ is explicitly calculated are
# 
# $$\displaystyle \begin{array}{lll}
# \text{step} 1: & \epsilon =N-a^2 =7/4, & a_1 =1/2+7/4 = 9/4\\
# \text{step} 2:&\epsilon_1 =N-a_1^2 =2-81/16=-3.0625,& a_2 = a_1 +\epsilon_1/2a_1 =1.569\\
# \text{step} 3:&\epsilon_2 =N-a_2^2 =2-(1.569)^2 =-0.4618,&  a_3 = a_2 +\epsilon_2/2a_2 =1.4218\\
# \text{step} 4:& \epsilon_3 = 2 - (1.4218)^2 = -0.0215,& a_4 = a_3 + \epsilon_3/2a_3 = 1.4142
# \end{array} $$
# 
# After four steps, a value accurate to three decimal places is produced; the convergence is very rapid because of the $a^2$ term in the equation.
# 
# Heron's formula can also be described as an average, because if $a$ is an estimate of $\sqrt{N}$ then $(a + N/a)/2$ is the average of the number $\sqrt{N}$ plus a small increment $a$ and the value $N/a$, which is less than $\sqrt{N}$. This result is also produced from Newton's method for approximating the roots of the equation, $N + a^2 = 0$, see Chapter 3.
# 
# $^2$ Making approximations is something that should not be done in the middle of a calculation. In this case its iterative nature effectively corrects for this, i.e. it is justified by leading to a formula that iteratively approximates the correct result.
# 
# The Babylonian root estimation process can be calculated as an algorithm. A 'for' loop is used to repeat the calculation, in this case seven times, although this number can easily be increased. The initial value is calculated as 'aa' (equation 6) and then updated in the loop as 'ab' Each time a the latest estimate of the root 'aa' replaces the older one 'ab'. The accurate value is $\sqrt{2}= 1.414213562373095$ and you can see that the algorithm rapidly converges to this value. Precision is not so critical in this calculation because unlike the calculation of $\pi$ no functions such as the square root are needed, just simple arithmetic. 
# 

# In[3]:


# Algorithm. Heron's or Babylonian method to find a square root.

N = 2                                       # find sqrt of N
a0= 1/2                                     # initial guess
m = 9                                       # number oif iterations
print('{:s} {:d} {:s} {:12.8f}'.format('square root of ', N, 'initial guess', a0) )
aa = a0                                     # 1st value
for i in range(m):                          # do iteration
    ab = aa +( N - aa**2 )/( 2*aa )         # equation 6
    print('{:d} {:20.15f}'.format( i,  ab) )
    aa = ab                                 # replace new value with old 
    pass


# If you try this algorithm with other values of $N$, you will soon realize that from a very wide range of initial values, convergence takes only a few iterations; it is far more efficient than the algorithm calculating $\pi$. You will need to increase $m$ a little to get more iterations with larger $N$.
# 
# ### 2.3 Golden ratio 
# 
# The golden ratio $1.618033 \cdots$ was thought by the ancient Greeks to be the perfect ratio of width to the height of a picture, but to me it appears a little too wide for its height. The golden ratio has been written about extensively and has somewhat of a cult following. In _ca_ 300 BC Euclid divided the line below, figure 5, so that AC is to AB as AB is to BC. This is the ratio of the total length and of the larger section, to the ratio of the larger to small sections. As an equation  $\displaystyle \phi=\frac{AC}{AB}=\frac{AB}{BC}$ , then
# 
#  $$\displaystyle  \frac{\phi +1}{\phi} = \phi \tag{7}$$
#  
#  ![drawing](chapter1-fig5.png)
#  
#  Figure 5. Defining the golden ratio.
#  
#  ____
#  
#  Rearranging produces $\phi^2-\phi-1=0$ which has the solution $\displaystyle \phi = \frac{1\pm \sqrt{5}}{2} = \pm 1.6180339\cdots$ and the Golden ratio $\phi$ is the positive root. Interestingly $1\phi = 0.6180339\cdots$ and as $\phi$ is the solution to an algebraic equation it is an algebraic not a transcendental number.
#  
# The golden ratio appears in unusual places, such as in rectangles and pentagons, as well as the coordinates of the edges of an icosahedron. These can be found from the values $(0, \pm 1, \pm \phi)$ and making a circular shift of the values and is formed only of triangles. Boron suboxide, $\mathrm{B_6O}$ forms particles of icosahedral symmetry.
# 
# Fullerenes, such as $\mathrm{C_{60}}$, and soccer balls, have a truncated icosahedral structure in which the vertices of the icosahedron are cut away to reveal a regular solid with sides of pentagons and hexagons. The truncated icosahedron is one of the Archimedean solids. In both cases the coordinates of the vertices are related by the golden ratio. The C$_{60}$ is formed of hexagons and pentagons and also has vertices given in terms of the golden ratio.

# ![drawing](chapter1-fig6.png) ![drawing](chapter1-fig6a.png)
# 
# Figure 6. An icosahedron (left) and a truncated icosahedron (right) which is the shape of a football and C$_{60}$ molecules.
# ____
# 
# A practical use of the golden ratio is that it provides the most efficient way of dividing a curve when trying to find its minimum numerically. This can be used when trying to find the minimum difference between a set of data $y_{ expt}$ and its fit to a theoretical expression $y_{calc}$. The minimum difference thus produces the best fit of the theoretical model to the experimental data. Suppose that the function $y_{calc}$ describes the first-order decay of a chemical species with time $t$, the theoretical equation is $\displaystyle y_{calc} = e^{-kt}$ where $k$ is the rate constant for the reaction we want to determine by fitting to experimental data. 
# 
# The parameter whose minimum is sought is the square of the difference between the experimental data points and the theoretical equation calculated at the same $t$ values but with different values of $k$. This is called the _residual_ and is the sum $\displaystyle R_k = \sum_{i=0}^n (y_{expt_i} - y_{calc_i})^2$ where points are labelled with index $i$. The curve whose minimum values is sought is then $R$ vs $k$ for a range of $k$ values; recall that $y_{calc}$ depends on $k$. One tedious way would be to start with some very large or very small value of $k$, far away from the true value, and calculate $R_k$ by changing $k$ by small amounts and comparing values of $R$ until a larger value of the residual than its predecessor is found. This method is called an exhaustive search, and this would take a huge number of computations to complete.
# 
# ![drawing](chapter1-fig7.png)
# 
# Figure 7. The first two steps in finding the minimum of a curve $R$ vs. $k$ using the golden ratio. The second step follows the labels in brackets.
# 
# ____
# 
# A smarter way, Figure 7, chooses two points A and B, known to straddle the true value of $k$, and places two new points in the interval A-B in the ratio $1/\phi$ and $1 - 1/\phi$ as a first step to determining where the minimum might lie. One point $k_a$ is therefore placed at $\approx 38$% along the interval, the point $k_b$ at $ 
# \approx 62$%, and the values at $k_a$ and $k_b$ are compared. If, as is the case in Figure 7, the value at $k_a$ is smaller than $k_b$ the minimum is in the region A to $k_b$, otherwise it is in $k_a$ to B.
# 
# The rules of this 'game' define the algorithm and are:
# 
# >**(i)** If the value of the function $R$ at $k_a \lt k_b$, then B is moved to $k_b$ to reduce the interval over which the minimum is to be found; the point $k_a$ becomes $k_b$ and a new $k_a$ is calculated in the original region A to $k_b$ and is placed at $1/\phi$ along this interval, labelled in the diagram as $k_a$.
# 
# >**(ii)** If the value of the curve at $k_a \gt k_b$ then labels A and B are swapped around, meaning that A is moved to $x_a$ to reduce the search interval; $k_b$ becomes $k_a$ and a new $k_b$ is placed at $1 - 1/\phi$ along the original region $k_a$ to B.
# 
# These processes are repeated until a fixed number of iterations have occurred, or the minimum difference in two consecutive $k$ values found to within a certain error whose value you must decide.
# 
# The next algorithm calculates the minimum of the function $f(x)=x^3 - 5x^2$, which occurs in the range $0 \to 6$. We shall suppose, for the purposes of testing the algorithm, that this represents the shape of the residual $R$ between some measured data and a calculated curve. The minimum value set between successive estimations of $k$ is $0.03$ and is used to terminate the calculation. The results are printed as the calculation proceeds.

# In[4]:


# Algorithm. Golden Section Search

f= lambda x:  x**3 - 5*x**2           # define a function

a = 0.0                               # set limits a and b
b = 6.0
N = 20                                # number of iterations 
Lmt= 0.02                             # smallest b-a allowed this is for you to determine
g = (np.sqrt(5.0)-1.0)/2.0               # golden ratio
ka = g*a+(1-g)*b                      # define start points
kb = g*b+(1-g)*a
fa = f(ka)                            # function at start pos’ns
fb = f(kb)
print('   i    ka    kb     fa       fb    b-a') 

i = 0
while abs(b-a)> Lmt and i < N-1 :
    i = i+1
    if fa < fb :                       # rule(i)
        b = kb
        kb= ka
        fb= fa 
        ka= g*a + (1.0 - g)*b
        fa= f(ka)
    else:                              # rule(ii)
        a = ka
        ka= kb
        fa= fb
        kb= g*b +(1.0 - g)*a
        fb= f(kb)
    print('{:4d} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f}'.format( i, ka, kb, fa, fb, b-a ) )
    pass                               # end while


# The table of results shows how the search range is reduced as iteration proceeds. The minimum of the function is known by differentiation, which is how we can check that the calculation is correct, and is at $x = 10/3 = 3.33 \cdots$ with a value $-18.519$; effectively the same as found by the Golden Section iteration.
# 
# 
# ## 3 The exponential $e$ and $e^x$
# 
# The final number we consider is the exponential $e = 2.7182818284 \cdots$ familiar from Boltzmann's equation, the growth of populations of bacteria, or the decay of a compound by its first-order reaction. It also describes the largest rate of increase of compound interest achievable in your savings account or mortgage.
# 
# The number $e$ is transcendental, just like $\pi$ or $\sqrt{2}$, in that it cannot be expressed as the solution to an algebraic equation; it transcends algebra and its value is obtained by expanding an algebraic series or as a limit of the expression;
# 
# $$\displaystyle  e = \lim_{k\to \infty}\left(1 + \frac{1}{k}\right)^k \quad \text{ or } \quad  e^x = \lim_{k\to\infty}\left(1 + \frac{x}{k}\right)^k$$
# 
# The limit is found by trying progressively larger values of $k$; and the first expression tends towards the constant value, $e$.
# 
# $$\displaystyle \begin{array}\\
#  k & \text{limit}\\
# 10         &2.593742 \\
# 1000       & 2.716924\\
# 100 000    & 2.718268\\
# 100 000 00 & 2.712828\\
# \end{array}$$  
# 
# In a savings account you receive interest on your capital sum. If this is an amount £$N$ and if the interest is at an annual rate of $r$% which is 'compounded' $k$ times per year, then at the end of a year the capital has grown to
# 
# $$\displaystyle N\left(1+\frac{0.01r}{k}  \right)^k$$
# 
# Starting with £3000, if this is compounded annually at 5%, you should receive $\displaystyle 3000\left(1+\frac{0.05}{1}  \right)^1 = £3150$ at the end of the first year. However, if the interest is compounded quarterly this rises $\displaystyle 3000\left(1+\frac{0.05}{4}  \right)^4 = £3152.84$ also at the end of the first year. If compounded daily ($k = 365$) for a year the interest would only rise to $£3153.80$ and, eventually, if continuously compounded, $k$ would tend to infinity giving the exponential. The maximum, therefore, that you could ever obtain would be $£3000\times e^{0.05} = £3153.81$.
# 
# The exponential is defined as a series expansion as was first discovered by Euler in 1748. The series converges absolutely for any finite value of $x$ even though it extends to infinity:
# 
# $$\displaystyle e^x = 1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\frac{x^4}{4!}+\cdots = \sum_{n=0}^\infty \frac{x^n}{n!} \tag{8}$$
# 
# The $\sum$ sign at the end of the series is a shorthand and indicates summation in this case from $n = 0$ to infinity. The factorials $2!, 3!$, etc. are products, such that,
# 
# $$\displaystyle 0!=1,\quad 1!=1,\quad 2!=2\times 1, \quad3!=3\times 2\times 1, \\ n!=n\times(n−1)\times(n−2)\times(n−3)\times\cdots \times 2\times 1$$  
# 
# and so on. Factorials are calculated at positive integer values only (see Section 7). The value of the exponential series with $x = 1$ is
# 
# $$\displaystyle e = 1+1+\frac{1}{2!}+\frac{1}{3!}+\frac{1}{4!}+\cdots = 2.71828$$
# 
# Consider now two series, 
# 
# $$\displaystyle  f(x)=  1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\frac{x^4}{4!}+\cdots \qquad \text{and} \qquad  1+y+\frac{y^2}{2!}+\frac{y^3}{3!}+\frac{y^4}{4!}+\cdots $$
# 
# By multiplying term by term and rearranging, it is found that $f(x)f(y) = f(x + y)$ , which means that the familiar property, that powers of products of numbers add, has been confirmed because,
# 
# $$\displaystyle  e^xe^y= e^{x+y}; \qquad \text{and} \qquad e^xe^ye^z= e^{x+y+z}\tag{9}$$
# 
# You can try calculating the exponential series by hand and easily see how the accuracy improves term by term. A quick examination of the series shows that you can use a previous value to obtain the next; starting at the second term, the third is calculated by multiplying the second by $x/3$, the fourth by multiplying the third by $x/4$, and so on. This type of process is called _recursion_ and the calculation of $\pi$ and of square roots described in Sections 1 and 2 are both recursive. The Python code to calculate $e^x$ recursively is shown below; you can add more terms in the for loop to get a better answer, which is the sum of terms $s$.

# In[5]:


# Algorithm. Series (recursive) calculation of an exponetial exp(x).
p = 1.0 
s = 1.0
x = 1.0                    # exponential value to calculate, if 2 calculates e^2
for i in range(1,12):      # increase number if does not comverge
    p = p*x/i 
    s = s + p 
    print(i,s)
    pass


# There are numerous examples of the exponential function in chemistry and physics. One of the most important is Boltzmann's equation, which relates the population of two energy levels, an upper one $n_2$ to that of a lower one $n_1$ and which are separated by energy $\Delta E$ at absolute temperature $T$,
# 
# $$\displaystyle \frac{n_2}{n_1}=e^{-\Delta E/(k_BT)} $$
# 
# Boltzmann's constant has the value $k_B = 1.38065 \times 10^{-23}\, \mathrm{J\, K^{-1}}$.
# 
# ## 4 Logarithms
# 
# The 'inverse' of the exponential is the logarithm, because if $y = e^x$ then a function, the logarithm, can be defined as $\ln(e^x) = x$, and therefore $\ln(y) = x$. More generally put, the logarithm of $x$ is the power by which a base number $b$ must be raised to give $x$. Normally we use base 10 or base $e$ and the use the symbols $\log(x)$ or $\log_{10}(x)$ for base 10 and $\ln(x)$ for base $e$ which are also called the natural logs.  Note, however, that this convention is not always followed in computer languages.
# 
# As an equation logs are defined as 
# 
# $$\displaystyle  x = b^{\log_b(x)}$$
# 
# The value of $x$ must be greater than zero for the log to be a real number. If $x\lt 0$,then a complex number is obtained, see Chapter 2.
# 
# A series of increasing powers of $2$, familiar in computing where $2$ kilobytes $\equiv 2048$ bytes, is shown on the top row of the table; the bottom row is the power with which to raise $2$ to obtain the number, e.g. $2^3 = 8$.
# 
# $$\displaystyle \begin{array}{c|cc}
# \hline
# n & -2 & -1 & 0 & 1 & 2& 3& 4& 5& 6& 7& 8\\
# \hline\\
# 2^n &1/4 & 1/2 &1 &2 &4 &8 &16& 32 &64 &128 &256\\
# \hline
# \end{array}$$
# 
# Calculating $1/4$ multiplied by $128$ can be done directly of course, but we want instead to avoid multiplication. As $1/4$ has a power $n = -2$, and $128$ has a power of $n = 7$, adding the powers and looking up the answer under $5$, we get $32$. This is written as $128/4 = 2^7/2^2 = 2^7 \times 2^{-2} = 2^5 = 32$. 
# 
# The idea behind the logarithm, therefore, is to 
# 
# $$\displaystyle \text{"use addition to multiply and subtraction to divide"}$$
# 
# and this is because logs relate to powers of numbers. The calculation $128/4$ done with logs is
# 
# $$\displaystyle \log_2\left(\frac{1}{4}\right) +\log_2(128) =-\log_2(4) +\log_2(128)=5$$
# 
# and as $5 = \log_2(32))$ the result is $32$.
# 
# If $a\gt 0$ and $ b> 0 $ the first two laws of logs are
# 
# $$\displaystyle \log(a) + \log(b) = \log(a \times b)  \tag{10}$$
# 
# $$\displaystyle \log(a) - \log(b) = \log\left( \frac{a}{b}\right)  \tag{11}$$
# 
# and these relationships are true no matter what the base is. 
# 
# Note that this rule does not apply to $\displaystyle \frac{\log(a)}{\log(b)}$  , which cannot be simplified.
# 
# The _third law_ of logs is:
# 
# $$\displaystyle \log(x^n) = n \log(x)\tag{12}$$
# 
# but note that $\log(x)^n$  means that the log of the number is raised to $n$; for example, $\log(x)^3 = \log(x)\log(x)\log(x)$, whereas $\log(x^3) = 3\log(x)$.
# 
# The number $2^8 = 256$ written in logarithmic form is $8 = \log_2(256)$, which means that '8 equals log base 2 of 256' and 'base 2' means that we are raising numbers to powers of $2$. The general formulae relating a number $N$ with base $b$ and a power $p$ are:
# 
# $$\displaystyle N = b^p \qquad \text{ or }\qquad  p = \log_b(N) \tag{13}$$
# 
# Natural (Napierian) logs, usually written as $\ln$, or $\log_e$, use $e$ as the base number with which we raise to a power, and $\log$ or $\log_{10}$ use 10 as the base, for example because $10^3 = 1000$ we can write this as $3 = \log_{10}(1000)$. The number $e^3 = 20.086$ can be written as $3 = \ln(20.086)$. The series of powers of 10 and $e$ are shown below:
# 
# $$\displaystyle \begin{array}{c|cc}
# \hline
# n & -2 & -1 & 0 & 1 & 2& 3& 4\\
# \hline\\
# 10^n &0.01 & 0.1 &1 &10 &100 &1000 &10000\\[10pt]
# \hline
# e^n & e^{-2}=0.135 & 0.368 & e^0=1 & e^1=2.718 & 7.389 & 20.086 & 54.598\\
# \hline
# \end{array}$$
# 
# 
# The change in population of a chemical species $c(t)$ decaying by a first-order process follows an exponential law, 
# 
# $$\displaystyle c(t) = c_0e^{-kt}$$
# 
# This is more easily analysed to find the rate constant $k$ by taking logs of both sides to give 
# 
# $$\displaystyle \ln\left(\frac{c(t)}{c_0}\right) = -kt$$
# 
# because a plot of the $\ln$ vs time  $t$ is a straight line of slope $-k$. Dividing the concentration $c(t)$ by that initially present $c_0$ makes the log dimensionless.
# 
# When Napier invented logarithms in c.1614, he used bones inscribed with numbers. The old-fashioned slide rule uses the principle of addition and subtraction to do multiplication and division with lines marked on rulers that slide past one another. The slide rule is quite easy to use with practice, and not that much slower, but less accurate, than a hand calculator.
# 
# ### 4.2 Summary of logs and powers: definition $x=b^{\log_b(x)}$
# 
# $$\displaystyle \begin{array}{lll}\\
# \hline
# a^0 = 1& \log(0)=-\infty& \log(1)=0,\quad \log(\infty)=\infty\\[10pt]
# a^{n+m}=a^ma^n & \log(a)+\log(b)=\log(a\cdot b) & \text{ if } a \gt 0 ; b\gt 0\\[10pt]
# a^{n-m}=a^na^{-m}=\frac{a^n}{a^m} & \displaystyle \log(a)-\log(b)=\log\left(\frac{a}{b}\right) & \text{ if } a \gt 0 ; b\gt 0\\[10pt]
# \left(a^m \right)^n=a^{m\cdot n} & \log(a^n)=n\log(a)& \text{ if } a \gt 0 ; n\ne 0\\[10pt]
# \text{ change base } &\log_a(x)=\log_a(b)\log_b(x)\\[10pt]
# \hline
# \end{array}$$
#     
# ### 4.3 Comparison of log and exponential 
# 
# The graph in Figure 8 shows the exponential and log functions; $e^x,\; e^{-x}$, and $\ln(x)$. Notice how the exponential and log are symmetrical about the line $y = x$. The curve that would be symmetrical about $y=-x$ and $e^{-x}$ is $\ln(|x|)$ where $|x|$ means taking the absolute value, which if $x$ is real means changing $-x$ into $x$.
# 
# The log is normally defined only over the range of positive $x$ values $0 \lt x \lt \infty,\; \ln(1) = 0$ and $\ln(0) = -\infty$. The log with negative $x$ values is a complex number; see Chapter 2.8. The exponential is defined over all values of $x$ and has the values $e^0 = 1,\; e^{-\infty} = 0$ and $e^\infty = \infty$. 
# 
# ### 4.4 Log as a series 
# 
# Just as with the exponential, the log can be written as a series expansion, but only if $|x|\lt 1$:
# 
# $$\displaystyle \begin{align}\ln(1+x)&=x-\frac{x^2}{2}+\frac{x^3}{3}-\frac{x^4}{4}+\cdots =\sum_{n=1}^\infty (-1)^{n+1}\frac{x^n}{n}  ,\\
# \ln(1-x)&=-x-\frac{x^2}{2}-\frac{x^3}{3}-\frac{x^4}{4}-\cdots =-\sum_{n=1}^\infty \frac{x^n}{n} \end{align} $$
# 
# and where the $(-1)^{n+1}$  makes the even-valued terms in $n$ negative. If we want to calculate $\ln(q)$, we can make $q = 1 + x$ and substitute $x = q - 1$ into the series. This and other series are described in Chapter 5.
# 
# ![drawing](chapter1-fig8.png)
# 
# Figure 8 Graph of exponential and log functions with the straight line $y = x$. The symmetry between $e^x$ and $\ln(x)$ is clear.

# In[ ]:





# In[ ]:




