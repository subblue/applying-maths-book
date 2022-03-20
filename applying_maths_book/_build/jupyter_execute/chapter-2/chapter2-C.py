#!/usr/bin/env python
# coding: utf-8

# ##  Euler's theorem
# 
# The exponential series is $\displaystyle e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$, and similarly a series can be formed in the complex number $w$, 
# 
# $$\displaystyle e^w = 1 + w + \frac{w^2}{2!} + \frac{w^3}{3!} + \cdots$$
# 
# Now suppose that $w = i\theta$, where $\theta$ is real, then rearrange into real and imaginary terms;
# 
# $$\begin{align}\displaystyle e^{i\theta} &= 1 + i\theta + \frac{i^2\theta^2}{2!} + \frac{i^3\theta^3}{3!} +\frac{i^4\theta^4}{4!} +\cdots \\
# &= 1 + i\theta - \frac{\theta^2}{2!} - i\frac{\theta^3}{3!}+\frac{\theta^4}{4!} +\cdots\\
# &=\left(1 -  \frac{\theta^2}{2!} +\frac{\theta^4}{4!} -\cdots \right)
#  +i\left( \theta  - \frac{\theta^3}{3!} +\frac{\theta^5}{5!} -\cdots\right)\end{align} $$
# 
# The real and imaginary parts are expansions of the cosine and sine functions respectively,
# therefore, if $z$ is a complex number
# 
# $$\displaystyle z = e^{i\theta} = \cos(\theta) + i \sin(\theta)\tag{16}$$
# 
# Figure 4 shows the relationship in diagrammatic form. This equation was discovered in 1748 by the Swiss mathematician Euler, and is extremely important as it crops up everywhere from quantum mechanics to X-ray diffraction in crystals and other phenomena connected with waves.
# 
# Changing $\theta \to -\theta$ produces
# 
# $$\displaystyle z = e^{-i\theta} = \cos(\theta) - i \sin(\theta)$$
# 
# because $\sin(-\theta)=-\sin(\theta)$ and $\cos(-\theta)=\cos(\theta)$ and therefore, for a general complex number with (modulus) $r$ as a real number,
# 
# $$\displaystyle re^z = re^{i\theta} = r\left(\cos(\theta) + i \sin(\theta)\right)$$
# 
# De Moivre's theorem can be derived from these equations: the power of a complex number $w$ is
# 
# $$\displaystyle w^n = r^ne^{in\theta} = r^n\left(\cos(n\theta) + i \sin(n\theta)\right) \tag{18}$$
# 
# Adding and subtracting $\displaystyle e^{\pm i\theta}$ gives
# 
# $$\displaystyle  \cos(\theta) = \frac{e^{i\theta}+e^{- i\theta}}{2} ;\qquad   \sin(\theta) = \frac{e^{i\theta}-e^{- i\theta}}{2i} $$
# 
# Calculating $\displaystyle e^{i\theta}$ with $\theta = \pi$ and $r = 1$ produces 
# 
# $$\displaystyle  e^{i\pi} =-1 \qquad  \text{ or } \qquad  e^{i\pi} +1=0 \tag{19}$$
# 
# which some consider the most beautiful equation in mathematics, as it connects the most important numbers of mathematics $(0, 1, i, e$, and $\pi)$ and uses the most important operations (multiplication, exponentiation, negation, and addition). Furthermore, an integer is produced by raising an irrational number $\pi$ times the imaginary unit $i$ to the power of another irrational number, $e$. It is not at all obvious why this connection exists from an arithmetical standpoint, but from a geometrical one it is clearer. Consider a circle of unit radius on an Argand diagram; as the angle $\theta$ increases from $0 \to 2\pi$, the modulus (radius) is $1$ when $\theta = 0$, and is $i$ when $\theta = \pi/2$, and $-1$ when the angle is $\pi$ and so on; see figure 4.
# 
# ### 4.1 Roots of unity - continued
# 
# The $n$ complex roots of unity are also easily calculated by defining
# 
# $$\displaystyle w_n=e^{2\pi i /n}\tag{20}$$
# 
# and the roots are obtained by raising this to integer powers $(w_n^j\equiv (w_n)^j)$
# 
# $$\displaystyle w_n^j=e^{2\pi ij /n},\quad j=0,1\cdots n-1$$
# 
# for example if $n = 4$ then $w_4^0 = e^{2\pi i 0/4},\; w_4^1= e^{2\pi i/4},\; w_4^2 = e^{2\pi i 2/4},\;w_4^3=e^{2\pi i3/4}$ and evaluating gives
# 
# $$\begin{align}w_4^0=&\;1,\\ w_4^1 = & \;\cos(\pi/2)+i\sin(\pi/2)=i,\\w_4^2 = & \;\cos(\pi)+i\sin(\pi)=-1,\\ w_4^3 = & \;\cos(3\pi/2)+i\sin(3\pi/2) = -i  \end{align}$$
# 
# With the definition of eqn 20, $(w_n)^n\equiv w_n^n=e^{2\pi i}=1 =w_n^0$. if we add $n$ to any of the roots for example $w_n^{j+n}$ then  $w_n^{j+n}=w_n^nw_n^j=w_n^j$ which shows that the roots are cyclic $j$ being an integer and have a period of $n$. 
# 
# #### Sum and product of the roots of unity
# Figure 5 shows the five roots of unity. The sum of these roots, provide there are two or more, is zero, which can be intuitively seen by looking at the image. A geometric argument is that each root can be considered as a vector based at $(0,0i)$ and their sum will be zero as each is equally spaced from its neighbour.
# 
# The sum is 
# 
# $$\displaystyle S=\sum_{k=0}^{n-1} e^{2\pi i k/n}$$
# 
# and we know from the first part of chapter 5 that the sum 
# 
# $$\displaystyle \sum_{k=0}^{n-1} x^k=\frac{1-x^n}{1-x}$$
# 
# making the substitution $x=e^{2\pi i/n}$ produces 
# 
# $$\displaystyle S=\frac{1-e^{2\pi i}}{1-e^{2\pi i/n}}=0$$
# 
# because we know from eqn. 20 that $(w_n)^n=e^{2\pi i}=1$ the sum is zero.
# 
# The product is 
# 
# $$\displaystyle \prod_{k=0}^{n-1} x^k= x^{n(n-1)/2}$$
# 
# where the sum of numbers from $0 \to n-1$ is $n(n-1)/2)$ as first worked out by Gauss when a schoolboy. The product is therefore 
# 
# $$\displaystyle e^{2\pi i (n-1)/2}=e^{i\pi n}e^{-i\pi}$$
# 
# As $e^{-i\pi} = -1$ and the other term is $\pm 1$ depending on whether $n$ is odd or even therefore the product is always $ 1$ if $n$ is even and $-1$ if odd, i.e $(-1)^n$.

# ### 4.2 Examples
# 
# Euler's formula is important in science, because it permits the description of a sinusoidally varying real quantity by means of complex exponentials as in Fourier Transforms described in Chapter 9. This change simplifies equations, because it is far easier to manipulate exponentials than trig functions. For example, the general form of a sinusoidally varying quantity, such as a plane wave, is $f (t) = a_0\cos(\omega t - \theta)$, where $a_0$ is the amplitude, $\omega$ the frequency, and $\theta$ the phase. These are all constants, and $t$ is time and is a real variable. The equivalent complex function is
# 
# $$\displaystyle g(t) = a_0e^{i(\theta-\omega t)} = a_0\left(\cos(\omega t - \theta) - i \sin(\omega t - \theta)\right)$$
# 
# therefore $f(t) = Re(g(t))$. Very often in chemistry and physics, the complex form is used without explicitly stating that it is only the real part that represents the waveform. Figure 7 compares these waveforms.
# 
# As an example of using Euler's equation, we will evaluate $w = \ln(-1)$ even though it doesn't exist as a pure real number, then calculate $w = \ln(i)$ and $w = \ln(z/3)$, where $z$ is any complex number. The strategy in problems of this type is to convert the number $-1$, or $i$, or whatever it is into an exponential form using Euler's theorem.
# 
# **(i)** In the first example, $w = \ln(-1)$ or $e^w = -1$ and $w$ has to be found to solve this equation. A general complex number can always be written as $z = re^{i\theta}$, therefore to find $w$, let $w = i\theta$. The absolute value (modulus) $r$ of $e^w$ is $e^{i\theta}e^{-i\theta} = 1$. Because $e^{i\theta} = \cos(\theta) + i \sin(\theta)$, when $\theta = \pi$, $e^{i\theta} = -1$ making the principal value of $\ln(-1) = \ln(1e^{i\pi}) = i\pi$, which, obviously, is a complex number. Note that there are other values of $\theta$ separated by $2k\pi i$, where $k$ is an integer because $e^{i\theta}$ is a cyclic function.
# 
# **(ii)** Suppose $w=\ln(i)$ or $e^w =i$. Let $w=i\theta$. As $e^{i\theta} =\cos(\theta)+i\sin(\theta)$,when $\theta = \pi/2$ this equation produces $e^{i\pi/2} = i$ or $\ln(i) = i\pi/2$.
# 
# **(iii)** If $w = \ln(z/3)$, then $3e^w = z$, and if $z$ is any complex number then we look for a value of $\theta$ such that $3e^{i\theta} = z$. Generally a complex number is represented by $z = re^{i\theta}$, then in this example $w = \ln(z) = \ln(3e^{i\theta}) = \ln(3) + i(\theta + 2\pi k)$ and $2\pi k$ is added because the function is cyclic and $k$ is any integer; recall that the Euler equation can be put into a cosine and sine form, so it is a repetitive function. The principal value occurs when $k = 0$.
# 
# Returning to example (i), $w = \ln(-1)$, if the $-1$ is treated as a complex number with an imaginary part that is zero, then the answer can be written down directly as $w = \ln(-1) = \ln(re^{i\theta}) = \ln(1) + i(\pi + 2\pi k)$
# and, since $r = 1$ and $\ln(1) = 0$, this gives the same result as in (i) $\ln(-1) = i\pi$ for the principal value.
# 
# ![Drawing](chapter2-fig7.png )
# 
# Figure 7. Visualizations of the complex number $e^{i\theta} = \cos(\theta) + i \sin(\theta)$ illustrate that it has a wavelike form.
