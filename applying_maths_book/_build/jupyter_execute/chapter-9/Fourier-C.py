#!/usr/bin/env python
# coding: utf-8

# In[1]:


Godfrey Beddard 'Applying Maths in the Chemical & Biomolecular Sciences an example-based approach' Chapter 9


# In[1]:


# import all python add-ons etc that will be needed later on
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy.integrate import quad
init_printing()                         # allows printing of SymPy results in typeset maths format
plt.rcParams.update({'font.size': 14})  # set font size for plots


# # 5 Fourier Transforms 
# 
# ## 5.1 Motivation and concept
# 
# Fourier transforms are of fundamental importance in the analysis of signals from many types of instruments; these range from infra-red spectroscopy, to x-ray crystallography, to MRI and CT scan imaging and to seismology. Even in everyday life, Fourier transforms are important because they are used to produce the images observed in a digital television and in most other forms of digital information processing. Every scientist is familiar with the interference pattern produced by light passing through a pair of slits; this is the spatial Fourier transform of the two slits. 
# 
# Usually, the data, which might be a string of values taken at many sequential times, is transformed to allow the frequencies present be displayed and analysed. More fundamentally, the instruments used to measure infrared and NMR spectra produce data that is itself the Fourier transform of the spectrum, and similarly, in X-ray crystallography, the image of spots produced on the detector is the Fourier transform of the gaps between the planes of atoms in a crystal. 
# 
# Although we concentrate on Fourier transforms, they are only one in a class of _integral_ transforms. The Abel transform is an integral transform that is used to recover the three-dimensional information from its two-dimensional image. It is used in such diverse areas as astronomy and the study of the photo-dissociation pathways of molecules. In photo-dissociation experiments, the fragments (atoms, ions, electrons) are spatially dispersed depending on where the breaking bond is pointing at the instant of dissociation. Their image is captured on a camera as 2D information and by transforming this, the geometry of the dissociation process can be determined (Whittaker 2007). Other transforms are the Hilbert, used in signal processing, and the Laplace, used to solve differential equations.
# 
# Folklore has it that Fourier transforms are formidably difficult and abstruse things. We know that they form the basis of the FTIR and NMR instrument, but secretly hope that nobody asks us how or why. In fact, Fourier transforms are quite straightforward but must be treated with respect. We are used to seeing the NMR or IR spectrum as a set of lines at different fixed frequencies and feel comfortable with this, but the raw data produced is a wiggly signal in which the information needed is almost totally obscured. This makes the process of unravelling this in a Fourier transform seems mysterious: ‘I cannot understand the data so where does the spectrum come from? Contrariwise, we are used to interpreting speech and music that are oscillating signals in time, and would not easily understand either of them if Fourier transformed and viewed or heard as a continuously changing spectrum of frequencies.
# 
# We shall come back to this shortly but, briefly, a Fourier transform is an integral and therefore it can be evaluated by any of the methods used to solve integrals. The Fourier transform integral is one of several types of integral transforms that have the general form
# 
# $$\displaystyle g(k)=\int f(x)G(k,x)dx      \tag{23}$$
# 
# The 'transformed' function is $g$ and the function being transformed is $f$. The algebraic expression $G$ is called the _kernel_ and this changes depending on the type of transform, Fourier, Abel, etc. The exact form of the kernel is also described later. However, whatever form the transform takes it always occurs between pairs of conjugate variables, which are $x$ and $k$ in equation (23). Often these conjugate pairs are time (seconds) and frequency (1/seconds), or distance and 1/distance. The reciprocal relationship between variables is why the transform converts time into frequency, changing, for example, an oscillating time profile into a spectrum. 
# 
# A second property is that these integral transforms are reversible, also called _invertible_, which means that $f$ can be changed into $g$ and $g$ can be changed into $f$ depending on which one we start with.
# 
# Solving the Fourier transform integral both algebraically and numerically will be described starting in Section 5.5, but first the role of the Fourier transform in FTIR and NMR experiments, and in X-ray crystallography is outlined.

# ### 5.2 The FTIR instrument
# 
# The Fourier transform infra-red (FTIR) spectrometer directly generates the Fourier transform of the spectrum by mechanically moving one mirror of a Michelson interferometer and measuring the signal generated by the interference of the two beams on the detector. Fig. 11 shows a (simulated) example of the raw data from the instrument and the IR spectrum produced after this is transformed. After transformation, the displacement from the centre of the interference signal is changed into reciprocal distance or wavenumbers, cm$^{-1}$, which is proportional to the IR transition frequency.
# 
# The FTIR spectrometer is an interferometer, therefore, the waves that have travelled down each of its arms are combined on the detector and this measures the intensity or the square of the wave's amplitude, Fig. 12. Constructive interference occurs when the path length in both arms differ by zero or a whole number of wavelengths; destructive interference occurs when they are exactly out of phase and the difference in length is an odd multiple of half a wavelength. If only one wavelength is present, changing the path-length $\Delta$ would make the signal on the detector change sinusoidally. The 'coherent' broadband infrared 'light' from the source contains many wavelengths, and at a given path-length, some constructively and some destructively interfere, but the signal is greatest when both paths are the same. The relative path-length of the two arms of the interferometer can be changed by mechanically moving one mirror; the full interference pattern is mapped out as a function of path-length and this pattern decreases in an oscillatory manner to some constant, but not zero value, as the difference in path length increases. This is shown in the left of Fig. 11. Because changing either path's length has the same effect, the signal is symmetrical about zero path difference.
# 
# When the sample is placed in the beam, it absorbs only some frequencies depending on the particular nature of the sample, which results in a change in the signal size on the detector. When this interference signal is subtracted from that obtained without the sample and is transformed, the infrared absorption spectrum is produced. The distance the mirrors are moved is accurately determined by using a visible laser that follows the same path in the interferometer, but does not pass through the sample. This laser produces an interference pattern on a second (photodiode) detector; the number of fringes passed as the arm of the interferometer moves is counted, and this is used to determine how far one mirror has moved relative to the other.
# 
# The FTIR spectrometer has the multiplex (Fellgett) advantage over a wavelength scanning instrument, because all wavelengths are simultaneously measured on the detector, which also receives a large and virtually noise free signal. Both of these factors improve the signal to noise ratio. In comparison, in a scanning instrument, the radiation is detected through a narrow slit and the wavelength is changed by rotating a diffraction grating. In such an instrument the narrow slit, necessary for high resolution, is responsible for a poor signal to noise ratio because only a little light can reach the detector at any given wavelength. Scanning the wavelength also makes the experiment lengthy.
# 
# <img src='fourier-fig11.png' alt='Drawing' style='width:500px;'/>
# Figure 11. Left: A simulated Fourier transform as might be produced directly by an FTIR spectrometer. Right: The IR spectrum after Fourier transforming and converting into transmittance.
# 
# ____
# 
# <img src="Fourier fig9-12.png" alt='Drawing' style='width:400px;'/>
# 
# Figure 12. Schematic of an FTIR spectrometer as an interferometer. The laser is used to measure the relative distance of the two arms and does not pass through the sample.
# 
# _____
# 
# 
# ## 5.3 NMR
# 
# Possibly the most important analytical technique for the synthetic chemist is NMR spectroscopy. In an NMR experiment, the nuclear magnetization, which is the vector sum of the individual nuclear spins, is tipped from its equilibrium direction, which is along the direction of the huge permanent magnetic field $B$, by a relatively weak RF pulse of short duration. By applying this short pulse along the $x$- or $y$-axis, and therefore at $90^\text{o}$ to the permanent field, the magnetization is tipped away from the z-direction and experiences a torque and starts to precess. After the RF pulse has ended, the nuclear magnetization, and hence individual nuclear spins, undergoes a free induction decay (FID) by continuing to precess about the permanent magnetic field $B$. The rotating magnetization, Fig. 13, is measured by the detecting coil in the x-y plane as an oscillating and decaying signal, which, when Fourier transformed, produces the NMR spectrum.
# 
# In this experiment, the oscillating and decaying signal is converted into reciprocal time or frequency, which is ultimately displayed as a frequency shift $\delta$ in ppm from a standard compound, such as tetramethylsilane. In a classical sense, it is possible imagine the rotating nuclear magnetization repeatedly passing in front of the detection coil, thereby inducing a current to flow in it as it does so, and which will cause the output signal to rise and fall. Many such magnetizations from the many groups of nuclear spins in different chemical environments produce many signals, resulting in a complicated oscillating FID. Figure 14 shows a synthesized NMR free induction decay of two spins with a frequency of $10$ and $11$ MHz, and the corresponding real and imaginary parts of the transform, which we will suppose is the NMR spectrum of two lines separated by $1$ MHz. The RF pulse used to tip the magnetization contains many frequencies, as may be seen from the Fourier series of a square pulse, and simultaneously excites the nuclear spins in different magnetic environments in the molecule. The analysis of the spectrum provides information about the structure of the molecule, but not bond distances or angles unless sophisticated multiple pulse methods are used (Sanders & Hunter 1987; Levitt 2001).
# 
# In an NMR experiment, the data is obtained as an FID rather than directly as a spectrum because this increases the speed of data acquisition and, more importantly, increases the signal to noise ratio over an instrument where the magnetic field is continuously changing in strength. In the FID, all frequencies are measured simultaneously, as in the FTIR instrument, giving the measurement a multiplex or Fellgett advantage. There are other reasons for measuring the FID, which is that the instrument now operates in real time; this allows multiples of RF pulses to be applied to the sample, and these allow the magnetization to be manipulated via multi-quantum processes.
# 
# 
# <img src="Fourier-9-13.png" alt='Drawing' style='width:400px;'/>
# Figure 13. The sequence of the magnetization and the FID produced during a basic NMR experiment.
# 
# ____
# 
# <img src='fourier-fig14.png' alt='Drawing' style='width:950px;'/>
# 
# Figure 14. A simulated FID of two NMR transitions showing its real and 'imaginary' parts and the phase. The real part is the absorption spectrum or the normal NMR spectrum, the imaginary part the dispersion. The vertical dashed lines show the frequencies used in the calculation of the FID.
# 
# ## 5.4 X-ray diffraction
# 
# In FTIR and NMR, a conscious choice is made to perform a transform type of experiment. This is not so in X-ray diffraction, for the very nature of the experiment removes any choice. In X-ray crystallography, the three-dimensional diffraction pattern produced by the X-rays that scatter off the electrons in the many different planes of atoms in the crystal is projected onto the two-dimensional detector surface and is measured as a pattern of bright spots. This image is Fourier transformed and produces the distances between lattice planes from which the molecule's structure can be determined. 
# 
# Scattering of the X-rays occurs because they interact with electrons and cause them to re-radiate, which they do so in all directions. Only when waves originate from planes of atoms that satisfy the Bragg law, $n\lambda = 2d\sin(\theta)$, is there constructive interference, and an X-ray is detected on the detector, usually a CCD. Everywhere else, there is destructive interference and no waves exist. The CCD detector is similar in nature to the one in a digital camera or mobile phone and the brightness of a spot is proportional to the amplitude squared (intensity) of the X-ray waves arriving at that point. 
# 
# The atoms in a crystal form repeating unit cells and each set of planes of atoms, in principle, will produce one spot on the detector and in a position proportional to the reciprocal of the lattice spacing between planes. Sometimes a crystal's symmetry may cause extra interference between X-rays from different planes, which produces systematic absences in the X-ray image and these can be use to distinguish one particular type of crystal lattice from another. 
# 
# It is important to note that it is not the positions of the spots on the detector that ultimately produces the molecular structure, these positions are determined by the reciprocal planes, but the _intensity_ of the spots. 
# 
# <img src='fourier-fig14a.png' alt='Drawing' style='width:300px;'/>
# 
# Figure 14a. The phase of scattered x-rays is given by $2\pi$ times the ratio of the perpendicular distance from the origin $R_{hkl}$ to an atom at $(x,y,z)$ to the separation of the hkl planes $d_{hkl}$ i.e $\phi=2\pi R_{hkl}/d_{hkl}$. The hkl plane is perpendicular to the plane of the diagram.
# 
# _____
# 
# The two-dimensional image on the detector has to be Fourier transformed into a representation of the crystal structure but, because the absolute value of the transform rather than its amplitude is produced on the detector, phase information is lost and this makes the interpretation of the image very much more difficult than it would otherwise be. This is the origin of the 'phase' problem and ingenious methods have had to be devised to overcome this (McKie & McKie 1992; Giacovazzo et al. 1992). The summation of several waves particular to this problem is described in chapter 1.
# 
# Fourier transforms are widely used in other areas, such as image processing, for example from star fields, MRI images, X-ray CT scans, information processing, and in solving many types of differential equations such as those describing molecular diffusion or heat flow. These technologies show that it is essential to be familiar with Fourier transforms whether you are a chemist, physicist, biologist, or a clinician.

# ## 5.5 Linear transforms
# 
# The next few sections describe the Fourier transform in detail, but first some jargon has to be explained. Formally, a Fourier transform is defined as a linear integral transform of one function or set of data into another; see equation (23). The transform is reversible, or invertible, enabling the original function or data to be retrieved after an inverse transform. These last two sentences are in 'math-speak', so what do they really mean?
# 
# Integral simply means that the transform involves an integration as shown in equation (23). The word 'linear', in 'linear transform, means that the transform $T$ has the property, when operating on two regular functions $f_1$ and $f_ 2$, that $T( f_1 + f_2) = T( f_1) + T( f_2)$. This means that the transform of the sum of $f_ 1$ and $f_ 2$ is the same as transforming $f_ 1$, and then transforming $f_2$ and adding the result. In addition, the linear transform has the property $T(cf_1) = cT( f_1)$ if $c$ is a constant.
# 
# Reversible, or invertible, means that a reverse transform exists that reforms the initial function from the transform; formally this can be written as $f = T^{-1}[T[f]]$ if $T^{-1}$ is the inverse transform. Put another way, if a function $f$ is transformed to form a new function $g$, as $T[f] = g$, then the inverse transform takes $g$ and reforms $f$ as $f = T^{-1}[g]$. This might seem to be rather abstract, but is, in fact, very common. 
# 
# A straightforward example is the log and exponential functions, as they are convertible into one another as an operator pair: If $T$ is the exponential operator $e^{( )}$, and $x^2$ is the 'function', then $\displaystyle T[x^2] = e^{x^2}$. The inverse operator $T^{-1}$ reproduces the original function: $T^{-1}[T[x^2]] = x^2$ or, by substitution, it is true that $\displaystyle T^{-1}[e^{x^2}] = x^2$ if $T^{-1}$ is the logarithmic operator $\ln( )$ because $\displaystyle \ln(e^{x^2}) = x^2$. The Fourier transform is only a
# more complicated version of an operator than is $\ln( )$ or $e^{( )}$.
# 
# The Fourier transform can be thought of as changing or 'mapping' the initial function $f$ to another function $g$, but in a systematic way. The new function may not look like the original, but however one might modify the transformed function $g$, when transformed back to $f$, it is as if $f$ itself had been modified. Although it is common to use the word 'transform', the word 'operator' could equally well be used although this is not usual in this context. Conversely, a matrix when acting on another matrix or a vector, performs a linear transform, however, a matrix is usually called a linear operator.

# ## 5.6 The Transform
# 
# The Fourier transform is used either because a problem is most easily solved in 'transform space', or, because of the way an experiment is performed, the data is produced in transform space and has then to be transformed back into 'real space'. This 'real space' is usually either time or distance; the transform space is then frequency (as inverse time) or inverse distance. The time-to-frequency and the distance-to-inverse distance are both _conjugate pairs_ of variables between which the Fourier transform operates. In practice, there are two 'flavours' of Fourier transforms. The simpler is the mathematical transformation of a function, such as a sine wave or exponential decay, the other is, effectively, the same process, but performed on real experimental data presented as a list of numbers. The latter is called the Discrete Fourier Transform (DFT). Because the transform is in reciprocal space, values near to zero on its abscissa correspond either to large values of frequency or  reciprocal distance depending on whether the conjugate variable is time or distance respectively. 
# 
# The Fourier transform is always between pairs of conjugate variables, time $\leftrightharpoons$ frequency, so that $\Delta t\Delta v = 1$ or distance $\leftrightharpoons$ 1/distance. As the transform changes one variable into its conjugate, it is possible in simple cases to visualize what the spectrum will look like without actually doing the calculation. A sine wave that has a single frequency has a Fourier transform that is a single line at the frequency of the wave. If there are two waves of different frequencies superimposed on one another, two lines will appear after transforming. 
# 
# So far, so good, but the length of the waves is not specified. Are they of finite length and so contain only a finite number of oscillations, or are they of infinite extent? If a sine wave is infinitely long, then only one line is observed in the transform, and will be of infinitesimal width and occur at the frequency of the sine wave. This line is a delta function. If the waves are turned on at some point and off again at another, then there are discontinuities at these points, and some additional frequencies must be associated with turning the signal on and off, which will appear in the transformed spectrum as _new_ frequencies. Think of how a waveform is made up of a sum of sine waves of different frequency, see Fig. 1. If a waveform is to be zero in some regions and not in others, then many waves have to be present to cancel out one another as necessary and these are the new frequencies needed. A broadening of the lines also occurs, because $\Delta t\Delta ν = 1$ and if $\Delta t$, the length of the whole sine wave is finite, then $\Delta v$ has a width associated with it. This is observed in FTIR and NMR spectra, but the software provided with many instruments can be set to remove as much of this broadening as possible by apodizing the lines (Sanders & Hunter 1987). This means multiplying the function with a decreasing function such as $\displaystyle e^{-x}$ before transforming.
# 
# The effect of Fourier transforming a short and a long rectangular pulse is shown below. The right-hand plots show the real part of the transform, which is a _sinc_ (or Cardinal Sine) function, $\mathrm{sinc}(ax) \equiv \sin(ax)/ax$. The result of transforming is mathematically the same for both long and sort pulses, of course, but in a fixed frequency range the effect appears to be different. The short pulse has a wide central band set at zero and widely spaced side bands, which decay rapidly at frequencies away from zero and extend to infinity. The longer pulse has a narrower central band, also centred at zero, and higher frequency side bands than in the short pulse case; the results conform to $\Delta t\Delta v = 1$, i.e. short $\Delta t$ with wide $\Delta v$ and _vice versa_.
# 
# If a pulse is turned on and off, as shown in Fig. 15, the transform must have frequencies associated with these changes. Again, think of the pulse being made of many terms in a Fourier series. Fig. 1 shows a few of the terms, but the more of these there are each with a different frequency, the better is a sharp edge or pulse defined. The oscillations in the transform of Fig. 15 arise from the many terms needed to describe the rectangular pulse. In fact, to reproduce the original pulse exactly by reverse transforming, an infinite frequency range is needed. If the transform of Fig. 15 _exactly_ as shown in the right-hand side were reverse transformed, the rectangular pulse shown on the left of the figure would not be produced, because on the plot the transform has a limited frequency range. 
# 
# The reciprocal nature of the function and its transform  is also clear in these plots. The wider the function the narrower the transform and vice versa, this leads to an 'uncertainty principle' in which it is not possible to measure, at the same time, both the function and its transform with unlimited precision. This is described in detail later on. In Quantum mechanics this leads to the Heisenberg Uncertainty Principle. 
# 
# <img src="fourier-fig15.png" alt='Drawing' style='width:500px;'/>
# 
# Figure 15. Example of the Fourier transform of a short and a long rectangular pulse each centred about zero and of total width $a$. Only the real part of the transform is shown, and is the sinc function, $\sin(ax)/ax$. The transform extends to $\pm \infty$ and $a=2$ in the top plots and 4 in the lower ones. The transform crosses zero at equally spaced points which are $\pm n/a$ where $n=1,2\cdots$.
# _____
# 
# What is the transform of a cosine wave of finite length? The result is shown in Fig. 16 and is somewhat similar to that of the square pulse except that the transform frequency cannot be centred at zero because the cosine has a finite frequency. The main peak is almost at the cosine's frequency, and the many other sidebands are needed to account for the fact that the wave is suddenly turned off. Now suppose that the cosine is damped by an exponential function  and smoothly decreases in amplitude, then these extra frequencies disappear, because at the end of the cosine wave there is no discontinuity; the exponential makes the cosine gently approach zero. The result is a widening of the feature at the frequency of the cosine wave, Fig. 17. The effect of the exponential decay is to _apodise_ the transform.
# 
# <img src="fourier-fig16.png" alt='Drawing' style='width:500px;'/>
# 
# Figure 16. Left: A truncated cosine wave of frequency 1/2, starting at zero and of length of 3.5 cycles. Right: The real part of its Fourier transform. The value of the wave’s frequency is marked with a vertical line.
# 
# <img src="fourier-fig17.png" alt='Drawing' style='width:500px;'/>
# Figure 17. The same cosine wave but now apodised by multiplying by $\displaystyle e^{-t/2}$, which makes the cosine diminish at long times. In the transform (right), one peak is found at the frequency of the wave (small vertical line). All the frequencies associated with suddenly ending the cosine are effectively removed.
# 
# _____

# ## 5.7 Fourier series and transforms
# 
# The connection between the Fourier series and the Fourier transform is important, and it should not be ignored. To produce the Fourier series such as that which describes a rectangular pulse, infinitely many terms in the Fourier series will be needed, and of ever increasing frequency. The Fourier transform allows us to see these frequencies by transforming to frequency space, so that each frequency in the Fourier series appears as a feature.
# 
# In an NMR experiment, a square pulse of RF radiation is used to excite the nuclear spin states in the sample and, as has been seen, the Fourier transform of such a pulse illustrates that it has many frequencies contained within it. In an experiment, the pulse is made of sufficient duration to contain all those frequencies needed to excite the nuclear spins. Of course, these frequencies are not made by the transform, but are there all the time, because to form the pulse in the first place many different sine or cosine waves each of different frequency are added together in the electronic circuitry.
# 
# To illustrate this further, consider a laser pulse with the duration of a few femtoseconds. Such pulses are made by the process of mode-locking. For a laser to work, the light waves in the cavity must fit exactly into its length no matter what the colour of the light, and a node must occur at each of the mirrors; the restriction is that $n$ half wavelengths must equal the cavity length, $n\lambda/2 = L$. If these waves, which have different frequencies for each $n$, can be forced to be in phase with one another, a pulse results; mode-locking is the process by which this is achieved. Making the phase the same means ensuring that each of the waves has a maximum in the same place, no matter what their frequency is. A pulse results because waves of different frequency must eventually fall out of step with one another away from zero or $\pm n\pi$, where they are in phase. Figure 18 shows that a pulse can only result from the addition of many different frequencies if they are in phase. The pulse is normalized to a maximum of $\pm 1$ in the figure and shows the amplitude, a photodiode or CCD detector measure the intensity which is the square of this signal is always positive. In a mode-locked laser, $\approx 10^6$ waves may be added together rather than the few shown; consequently, the laser pulse is far better defined.
# 
# <img src="fourier-fig18.png" alt='Drawing' style='width:400px;'/>
# 
# Figure 18. Left: Eleven cosine waves and their sum show that pulses can only be made by adding waves of different frequency together but only if they have the same phase. Right: One possible sum when the waves are added with random phases. The waves are $\cos(nx/2)$ where $n$ is an odd integer. The effect is more pronounced if more waves are used; the pulse becomes shorter and the random noise (right) becomes smaller in amplitude. The lower two plots show the square of the signals in the upper ones, plotted on the same scale. The square is important because if the waves correspond to the photons electric field, the intensity measured is the square of this. The pulses and random noise are both clear.
# ____
# 
# To realize mode-locking, a laser must have a broad emission spectrum and nowadays titanium sapphire is often used as the gain medium to produce femtosecond duration pulses, dye-lasers are sometimes still used to produce picosecond pulses. The Ti$^{3+}$ ions have many different sites in the sapphire ($\mathrm{Al_2O_3}$) crystal lattice and therefore have a broad emission spectrum, which is in the far-red part of the visible spectrum and centred around $850$ nm. The molecules or ions used to produce the fluorescence/luminescence which give rise to lasing, have a certain wavelength range caused by the nature of their potential energy surfaces and by the inhomogeneity of the host material a glass or liquid, for example, which shifts energy levels up and down. The coating on the mirrors, and perhaps added optical elements such as gratings, interference or birefringent (Lyot) filters, restrict the wavelengths over which the laser can operate, and this is done to enable the wavelength to be changed. However, if a short pulse is to be produced, the wavelength range has to be so wide that no filters are wanted, quite the opposite, as little restriction as possible on the wavelength range is desirable, as the product $\Delta v\Delta t$ has a constant value. This means that a wide frequency (or wavelength) range is necessary if $\Delta t$ is to be small. This is entirely consistent with the observation that many waves of different frequencies are needed to make a pulse. ( In practice is possible to produce  femtosecond pulses centred at different the wavelenghts as the spread in wavelength needed to produce the pulse is less than the possible wavelength range of the emission.)

# ## 6 The Fourier Transform equations
# 
# The derivation of the transform equations is now sketched out by starting with the Fourier series. Butkov (1968) gives the full derivation. The Fourier series, considered in Section 1, are all formed from periodic functions, but suppose that the function is thought of as having an infinite period, or to put it another way, if the limits are $-L \to L$ then what happens when $L \to \infty $? It is easier here to use the complex exponential form of the series, equations (7), and write
# 
# $$\displaystyle f(x) = \sum_{n=-\infty}^{\infty}c_ne^{+in\pi x/L}    \tag{24} $$
# 
# with coefficients
# 
# $$\displaystyle c_n= \frac{1}{2L}\int_{-L}^L f(x)e^{-in\pi x/L}    \tag{25} $$
# 
# where $n$ is an integer specifying the position in the series, therefore, $c_n$ is one of a series of numbers that could be plotted on a graph $c_n$ vs $n$. To simplify (24), we define $k = n\pi /L$, which gives $\Delta k = (\pi/L)\Delta n$ for a small change in $k$, and clearly, as $L$ gets larger, $k$ gets smaller. However, there is a problem here, for when $L\to \infty$ it looks as though all values of $c_n$, equation (25), will go to zero, because $L$ is in the denominator. 
# 
# Instead of immediately taking the limit, suppose that the values of $n$ describe adjacent points on a graph of $c_n$ vs $n$, and because adjacent points are the smallest differences that $n$ can have, then $\Delta n  = 1$ and so $\Delta k = \pi /L$ or $(L/\pi )\Delta k = 1$. Equation (24) can now be multiplied by this factor without difficulty because it is $1$, giving
# 
# $$\displaystyle f(x)=\sum_{n=-\infty}^{\infty}\frac{L}{\pi}c_ne^{+ikx} \Delta k   \tag{26} $$
# 
# and $c_n$ is given by equation (25). The limit $L\to \infty$ also means that $\Delta k \to  0$, which makes $k$ into a continuous variable, and the coefficients $c_n$ can now be written as a function of $k$, i.e. as $c(k)$ instead of the discrete values $c_n$. Taking this limit also changes $f(x)$ to an integral, because $\Delta k \to  0$,
# 
# $$\displaystyle f(x)=\lim_{L \rightarrow \infty}\sum_{n=-\infty}^{\infty}\frac{L}{\pi}c_ce^{in\pi x/L}\Delta k =\int_{-\infty}^\infty c(k)e^{ikx} dk$$
# 
# and $c(k) = Lc_n/\pi $ but from eqn. 25 $c(k)$ is
# 
# $$c(k)= \frac{1}{2\pi}\int_{-\infty}^\infty f(x)e^{-ikx}dx  $$
# 
# This equation is conventionally rewritten by defining a new function $g(k)$, where $g(k) = c(k)\sqrt{2π}$. This function is the _forward transform_ and is defined as 
# 
# $$\displaystyle  g(k) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty f(x)e^{-ikx}dx \qquad\text{       forward transform} \tag{27}$$
# 
# and the reverse transform is 
# 
# $$\displaystyle  f(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty g(k)e^{+ikx}dk \qquad\text{       reverse transform} \tag{28}$$
# 
# notice how the $x$ and $k$ and the signs in the exponential change.
# 
# The two functions form a Fourier transform pair; the function $f(x)$ with a positive exponential is the 'reverse' or 'inverse' transform, and $g(k)$, equation (27), with a negative exponential, the 'forward' transform because it converts the measured or known function $f(x)$, where $x$ might be distance, into the transformed space $k$ which is inverse distance. Alternatively, if $x$ represents time then $k$ represents frequency.
# 
# There are some other points to note.
# 
# **(i)** These equations give the value of the transform at one point only. To obtain the full transform, $k$ has to be varied in principle from $-\infty$ to +$\infty$, but, in practice, a value of $k$ which is far less than infinity can be used because the transform often has an infinitesimal amplitude at large $k$; see Fig. 19 for an example.
# 
# **(ii)** Because the integration involves a complex number, the result might be complex or it might be real; this just depends on what the function is and it might therefore be necessary to plot the real, imaginary, and absolute value of the transform.
# 
# **(iii)** There are different forms of Fourier transform pairs that differ from one another by normalization constants, $1/ 2\pi$ in our notation. This can lead to confusion when comparing one calculation with another.
# 
# **(iv)** Finally, note that some authors, engineers in particular, often define the forward transform with a positive sign in the exponential and negative in the reverse, which is a change of phase with respect to our notation. They also often use $j$ instead of $i$ to mean $\sqrt{-1}$.
# 
# ### 6.1 Plotting transforms
# 
# Because the transform is normally a complex quantity, it has a real and imaginary part. In plotting the transform three graphs can be produced; one for each of the real and the imaginary components of the whole transform and one of the square of the absolute value, which is usually called the power or transform spectrum and is $g(k)^*g(k) = |g(k)|^2$, the asterisk indicating the complex conjugate.
# 
# ### 6.2 What functions can be transformed?
# 
# To perform the transform, $f(x)$ must be integrable and must converge when the integration limits are infinity; this generally means that $f(x) \to$ 0 as $x \to \pm \infty$: a sufficient condition is that $\int_{-\infty}^{\infty} |f(x)|dx$ exists.
# 
# 
# ### 6.3 How to calculate and plot a Fourier transform
# 
# As an illustration, the Fourier transform of a sine wave $f(x) = \sin(\omega x)$ which has an angular frequency $\omega = 2\pi/L$ will be calculated over the range $-L$ to +$L$; this supposes also that the function $f(x)$ is zero everywhere else, see Fig. 19. Choosing the sine function to have the argument $2\pi x/L$ means that it is zero, i.e. has a node, at $x = \pm L$; note that the frequency need not be a multiple of the range of the transform, but the resulting equations are simpler if it is. Because the function is zero outside $\pm L$, so is the integral, and the integration limits become $\pm L$ rather than $\pm \infty$. The forward transform uses eqn. 27
# 
# $$\displaystyle  g(k) = \frac{1}{\sqrt{2\pi}}\int_{-L}^L \sin(2\pi x/L)e^{-ikx}dx $$
# 
# which is easily integrated using the exponential form of the sine. The result is 
# 
# $$\displaystyle  g(k)= -\frac{4i\pi \sin(Lk)}{k^2L^2-4\pi^2}$$

# In[2]:


# check using sympy  1j is the Sympy definition of mathematical 'i' 

L,x,k = symbols( 'L x k',positive =True)

f01 = sin(2*pi*x/L)*exp(-1j*k*x)

g = simplify(integrate(f01,(x,-L,L),conds='none') )
g


# which is the same result after converting the exponentials to the sine. This can be checked also by SymPy using the instructions 'simplify(g.rewrite(sin))'
# 
# The Fourier transform is, in this particular example, wholly the imaginary part of a complex number. When $k  = 0$ and when $Lk = \pm n \pi$, the transform is zero, except when $Lk = \pm 2\pi$ where the maximum or minimum occurs. When $Lk = +2\pi$, the transform has the nominal value of $0/0$, which can be evaluated using l'Hopital's rule (see Chapter 3). Remember to stop differentiating when either the top or bottom of the fraction is not zero, the result is
# 
# $$\displaystyle  \lim_{k \to 2\pi /L} \frac{-4\pi iL\sin(Lk)}{k^2L^2-4\pi^2} \to \frac{-4\pi i L^2\cos(Lk)}{2kL^2} = -\frac{2\pi i}{k} = -iL$$
# 
# which is the minimum value of the transform. The maximum occurs when $kL = -2\pi$, (see Fig. 19), which corresponds to the frequency $k = 2\pi /L \equiv \omega$ in radians, or 1/$L$ in Hz, if $L$ measures time. If $L$ is distance, cm for example, as in an FTIR spectrometer, then 1/$L$ is in wavenumbers or cm$^{-1}$.
# 
# To plot the transform, it is necessary to plot either the imaginary part (fig 19) or its absolute value; there is no real part in this particular example. Notice that there appear to be two frequencies, one at about 0.5 and at -0.5; negative frequencies do not make any sense if the sine wave is a signal from an experiment and for real experimental data, the negative frequencies need to be ignored. If the range $\pm L$ is kept the same, and instead of a sine, a cosine wave of the same frequency transformed, the real frequency part of the Fourier transform would now look like the imaginary part of Fig. 19.
# 
# As a sine wave of infinite extent has a single frequency the extra frequencies seen in Fig. 19 must arise due to the fact that this wave exists only between $\pm L$. The sudden change in value of the function at $\pm L$ corresponds to having several different frequencies present, although they are not apparently there. Put another way, if a Fourier series of this truncated sine wave had to be formed very many sines or cosines of a different frequencies would have to be included. Why several terms? A single sine wave normally extends to infinity, many waves of different frequency are needed to reinforce the values near $k = 0$ and simultaneously to cancel out the part where the amplitude is zero, between $-L \to -\infty$ and $L$ and $\infty$. Although, for practical purposes, these regions in the integration were ignored, this was only because the sine wave is zero here, but this does not mean that waves do not exist to make the amplitude zero. These terms produce the extra frequencies seen in the transform. Put another way, to understand the transform it is necessary to consider all the terms needed to describe the initial truncated function $f(x)$ as a Fourier series because it is exactly these terms that appear as frequencies in the transform.
# 
# <img src="fourier-fig19.png" alt='Drawing' style='width:450px;'/>
# 
# Figure 19. Graphs of $\sin(\omega x)$ from $\pm L$ when $L = 10$; its Fourier transform, the imaginary part (top right), and its spectrum, the square of its absolute value (bottom right). The real part of the transform is zero because $\displaystyle g(k)= -\frac{4i\pi \sin(Lk)}{k^2L^2-4\pi^2}$ has no real part. (The vertical scales are not the same, but the maximum value of the transform and its absolute value is $L$.)
# 

# ### 6.4 How the Fourier transform works
# 
# The transform appears to have the effect of seeking out any repetitive features in a signal $f(x)$. This is true whether it is the discrete transform acting on real data, or the mathematical transform of a sine wave or other function. To understand what the transform does, we must look at eqn 27,
# 
# $$\displaystyle  g(k) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty f(x)e^{-ikx}dx $$
# 
# and recall that this only gives the value at one point $k$. To obtain the transform, $k$ has to vary from -${\infty} \to \infty$ although in practice only a limited range is needed to observe the major features of the transform. In this exponential form, the oscillating nature of the argument is not so apparent, but writing it as
# 
# $$\displaystyle  g(k) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty f(x)(\cos(kx)-i\sin(kx))dx $$
# 
# shows that the function $f$ is multiplied by a sine and cosine and integrated. Because $k$ can take any value, the sine and cosine of all possible frequencies multiply $f$. Most of the time, multiplication results in a highly oscillatory function, with as many positive parts as negative ones, and the integral evaluates to zero or something very close to it. 
# 
# When the period of $f$ is close to, or the same as, that of the sine or cosine, when multiplied together these no longer integrate to give zero. Hence, the particular frequency determined by $k$ gives the transform its value. This effect is pictured in Fig. 20. The left-hand column shows a function $f$ (top image) with a long period compared to a particular frequency of the sine wave in the transform at some value of $k$, middle image left. The bottom left curve shows the product of these two curves. The integral of this product, the area under the curve, almost evaluates to zero, with the positive and negative parts cancelling. 
# 
# The middle graph of the right-hand column shows a different frequency of the sine wave, because $k$ now has a different value in $\sin(kx)$, and the sine wave's period now matches that of the function $f$; their product is now positive and its integral is not zero. The Fourier transform therefore selects this frequency from among all others. Naturally, if there are several frequencies present in the function, these are each picked out in a similar manner as $k$ changes.
# 
# <img src="fourier-fig20.png" alt='Drawing' style='width:500px;'/>
# 
# Figure 20. Left column: The function $f(x)$ has a period that is very different from that of the sine wave $\sin(4kx)$ middle curve. Their product, lowest left-hand curve, oscillates about zero and integrates to zero or very close to it and so appears as an insignificant feature in the transform. <br>
# Right column: The period of the sine wave, (middle curve) which is determined by $k$, is now changed compared to the left-hand figure and now matches the period of the function. The product $f(x)\sin(kx)$ is now only positive, and integrates to a finite number and so appears as a peak in the transform.

# ### 6.5 Phase sensitive detection
# 
# In measuring signals buried in noise, the technique of phase sensitive detection is a very effective way of extracting the data and removing noise. In this method, the input to an experiment is modulated at a fixed frequency and the signal produced by the experiment is measured at this same frequency by a device known as a _lock-in amplifier_. This device illustrates the principle underlying the Fourier transform, although it is not a transform method.
# 
# In using a lock-in amplifier to measure fluorescence the light used to excite the molecules, and so stimulate the fluorescence, is modulated by rotating a slotted disc (chopper) in the exciting light's path. The photomultiplier or photodiode detects the modulated (on - off ) fluorescence signal together with any noise and this signal is passed to the lock-in amplifier. The lock-in also receives a reference signal directly from the chopper and it electronically multiplies this with the fluorescence signal (see fig 20). The schematic of an instrument is shown in Fig.21. 
# 
# Multiples (higher harmonics) of the fundamental reference frequency are filtered away, the resulting signal is integrated over many periods of the fundamental frequency, and a DC output signal is produced. As shown in Fig. 20, when the product of reference and signal is integrated, frequencies dissimilar to the reference, $f(x)$ in the figure, will average to something approaching zero.
# 
# If the reference signal is $r = r_0\sin(\omega t)$ and the noise free signal $s = s_0\sin(\omega t + \varphi)$ then the output of the lock-in is
# 
# $$\displaystyle  V_s= \frac{r_0s_0}{T}\int_0^T \sin(\omega t + \varphi)\sin(\omega t)dt$$
# 
# where $T = 2\pi n/\omega$ and $n \gg$ 1 is an integer and $\varphi$ is the phase (time) delay between the reference and the signal and is due to detectors and the amplifiers and other components in the experiment, but can be changed by the user. Expanding the sines and integrating gives
# 
# $$\displaystyle  V_s= \frac{r_0s_0}{T}\int_0^T \cos(\varphi)-\cos(2\omega t+\varphi)dt\\
# =\frac{r_0s_0}{4\omega T}[\sin(\varphi)+2T\omega \cos(\varphi)-\sin(2\omega T+\varphi) ] $$
# 
# The sine at twice the reference frequency is electronically filtered away leaving a signal that is constant because $T$ is the integration time set by the experimentalist and normally ranges from a few milliseconds to a few seconds. The measured signal is
# 
# $$\displaystyle V_s=\frac{r_0s_0}{4\omega T}[\sin(\varphi)+2T\omega \cos(\varphi) ]   \tag{29}$$
# 
# and as the phase $\varphi$ can be adjusted by the user, this signal can be maximized.
# 
# Now consider the situation when noise is present and assume that this has a wide range of frequencies $\omega_{1,2..}$ and amplitudes $n_{1,2..}$. The signal from an instrument is normally noisy and is represented as
# 
# $$\displaystyle s_0\sin(\omega t+\varphi)+n_1\sin(\omega _1t+\varphi_1)+n_2\sin(\omega _2t+\varphi_2)+\cdots$$
# 
# where $s_0, \omega$, and $\varphi$ are respectively the amplitude, frequency, and phase (relative to the reference) of the data. 
# 
# The first term of the signal arises from the information we wish to measure and produces $V_s$ equation (29). We need only consider one noise term, for all the others behave similarly. Multiplying by the reference at frequency $\omega$ but ignoring the phase $\varphi$, as this adds nothing fundamental but makes the equations more complicated, gives the term,
# 
# $$\displaystyle \sin(\omega_1 t)\sin(\omega t)=[\cos((\omega-\omega_1)t)-\cos((\omega+\omega_1)t)]/2 $$
# 
# Integrating produces
# 
# $$\displaystyle V_n= \frac{r_0n_1}{2T} \int_0^T \cos([\omega-\omega_1]t)-\cos([\omega+\omega_1]t) dt\\
# =\frac{r_0n_1}{4T} \left[ \frac{ \sin([\omega_1-\omega]T)}{\omega_1-\omega}  -\frac{\sin([\omega_1+\omega]T)}{\omega_1+\omega}  \right] $$
# 
# The sum frequency term is filtered by the instrument and is removed from the output leaving the term $\displaystyle \frac{ \sin([\omega_1-\omega]T)}{\omega_1-\omega}$ which is the sinc function, see Fig. 15. Suppose that the frequency $\omega_1$ represents white noise that contains all frequencies more or less equally. As these frequencies differ from $\omega$ and the absolute value $| \omega_1-\omega |$ becomes larger the sinc function rapidly becomes very small. This means that the reference sine wave picks out just that frequency containing the signal and rejects almost all of the noise. The total signal is $V + V_n$ and although it still contains noise at the reference frequency $\omega$ it contains very little at other frequencies, and the signal to noise ratio is increased very considerably. Often signals can be extracted from what appears to be completely noisy data.
# 
# As a practical consideration, the reference frequency should always be chosen to be a prime number so that the chance of detecting one of the multiples of electrical mains frequency is reduced. Also, this frequency should be in a region where the inherent noise of the experiment is low and if possible be of a high enough frequency to allow a short time $T$ to be used in the integration step, allowing many separate measurements to be made in a reasonable time.
# 
# <img src="fourier-fig21.png" alt='Drawing' style='width:350px;'/>
# 
# Figure 21. Schematic of phase sensitive detection and a lock-in amplifier.

# ### 6.6 Parseval or Plancherel  theorem.
# 
# This theorem is important because it proves that there is no loss of information when transforming between Fourier transform pairs. This is rather important because otherwise how would it be possible to tell what information has been lost or added? Fortunately, it can be shown that
# 
# $$\displaystyle \int_{-\infty}^{\infty} g^*(k)g(k)dk = \int_{-\infty}^{\infty} f^*(x)f(x)dx  \tag{30}$$
# 
# where the asterisk denotes the complex conjugate. The Fourier transform of $f(x)$ is $g(k)$, which is integrated over its variable $k$, and similarly $f(x)$ is integrated over its variable $x$. As the total integral taken over all coordinate space $x$ and that over its conjugate variable $k$ is the same, all the information in the original function is retained in the transformation, i.e. it looks as if the transform is a different beast but this is only a disguise as it contains exactly the same information. This, of course, means that if something is done to the transform, then, in effect, the same is done to the function.
# 
# The Plancherel theorem (also called Rayleigh's theorem as it was first used by him in the theory of Black-Body radiation) is effectively the same but usually written as 
# 
# $$\displaystyle  \int_{-\infty}^{\infty} |g(k)|^2dk = \int_{-\infty}^{\infty} |f(x)|^2 dx $$
# 
# Graphically it means that the shaded areas are the same. The figure shows the transform of a square wave as in fig 15. The transform is $\displaystyle g(k)=\frac{\sin(ak)}{ak}$ where $a = 4$. The function $f(x) = 1 $ in the range $-2\to 2$ so $\int f(x)^2 dx= 4$. 
# 
# <img src="fourier-fig21a.png" alt='Drawing' style='width:550px;'/>
# 
# Fifure 21a. Illustrating the Parseval or Plancherel theorem. The shaded areas are the same size, although they do not appear to be. The absolute square of the transform $|g(k)|^2$ extends to $\pm \infty$ which makes up the area since the function is always positive.
# 
# This theorem is very important in quantum mechanics. Should $f(x)$ represent a wavefunction that varies as a function of distance $x$, which could be the displacement from equilibrium of an harmonic oscillator, then variable $k$ can be interpreted as the momentum (usually given the letter $p$) making $g(k)$ the wavefunction in 'momentum space'. This means that calculations can be formed either in spatial coordinates, i.e. distance or in 'momentum space' depending upon which is the most convenient mathematically. The change in displacement, $\delta x$, and change in momentum, $\delta p$, are conjugate pairs of variables and are linked by the Heisenberg uncertainty principle $\delta x\delta p \ge \hbar/2$.

# ### 6.6.1 Uncertainty Principle
# 
# It is known from experiment that when an emission line from an atomic or molecular transition has a very broad frequency spread, then the lifetime $\tau$ of the state involved is short lived and vice versa. This is called the 'time-energy' uncertainty relationship $\Delta E\tau \ge \hbar/2$ or equivalently $v\tau\ge 1/2$ as $E=hv$. A similar effect is observed when a time varying signal is measured such as a voltage  on an oscilloscope. The product of the signal's duration and its bandwidth (its spread in frequency) has a certain minimum value. This is a consequence of the variables, time and frequency being related via a fourier transform. Time and frequency are called _conjugate_ variables. 
# 
# To show this can be quite tricky since the variance of the transform has to be calculated and this may not be integrable. The proof is given by Bracewell 'The Fourier Tranfrom and its Applications'. Instead of giving this proof to illustrate the effect some examples of particular cases are described.
# 
# As a measure of the spread in a value its standard deviation can be used, see Chapter 4 Integration eqn. 26. The square of the standard deviation is the variance and is defined as 
# 
# $$\displaystyle \sigma^2 =\langle x^2\rangle - \langle x\rangle^2$$
# 
# where the brackets $\langle \rangle$ indicate the average value. The average value of a function $p(x)$ is defined as 
# 
# $$\displaystyle \langle x_p\rangle= \int xp(x)dx\big/\int p(x)dx$$
# 
# and the average of the square 
# 
# $$\displaystyle \langle x_p^2\rangle= \int x^2p(x)dx\big/\int p(x)dx$$
# 
# The denominator is the normalisation. If the function is symmetrical about zero the mean is also zero and $\langle x\rangle=0$ and can be ignored. All that is necessary is to calculate $\langle f^2\rangle\langle g^2\rangle$ for a function $f$ and its transform $g$ and take the square root to obtain the standard deviation of the product. 
# 
# Often the transform and function may not be good distributions in which case their square is taken as the function to use thus we make the average squared ( also called the energy function) as 
# 
# $$\displaystyle \langle x^2\rangle= \int x^2f^*f dx\big/\int f^*f dx$$
# 
# and a similar equation for the transform $g$. As the function and transform is usually complex the square is obtained using the complex conjugate i.e. $p^2 \to p^*p$.
# 
# This method will be demonstrated below but sometimes the transform integrals become infinity, in this case we choose to take the deviation as the measure from the peak of the transform $g$ to the first zero. 
# 
# #### (a)  A Gaussian shaped pulse. 
# 
# A wavepacket comprising several waves of varying frequency can have an overall profile that is Gaussian in shape. This can apply, for example, to a laser pulse or a summation of harmonic oscillator wavefunctions. The normalised Gaussian function is $\displaystyle f(t)= \frac{e^{-t^2/2\sigma^2}}{\sigma \sqrt{2\pi} } $ where $\sigma$ is the standard deviation, or width of the Gaussian, see figure 4, Chapter 13 'Data Analysis'. The transform of a Gaussian is also a Gaussian but with a different width. 
# 
# $$\displaystyle g(v)= \frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty f(t)e^{-ivt} dt = \frac{1}{\sqrt{2\pi}}e^{-i\sigma^2v^2/2}$$
# 
# As the standard deviation of $f$ is by definition $\sigma$ and by comparison the standard deviation of $g$ must be $1/\sigma$ hence $\sigma_t\sigma_v=\sigma/\sigma=1$.
# 
# Calculating using the method outlined above to find $\langle t^2\rangle$ and $\langle v^2\rangle$ we expect a different result because we now use the square of the function and transform.
# 
# Taking all integrations between $\pm\infty$, the function $\displaystyle  \int f(t)^2dt= \frac{1}{2a\sqrt{\pi}}$ and $\displaystyle \int t^2f(t)^2dt = \frac{a}{4\sqrt{\pi}}$ and their ratio is $\displaystyle \Delta t= \frac{a^2}{2}$.
# 
# The transform has similar integrals$\displaystyle  \int g(v)^2dv= \frac{1}{2a\sqrt{\pi}}$ and $\displaystyle \int v^2g(v)^2dv = \frac{1}{4a^3\sqrt{\pi}}$ and their ratio is $\displaystyle \Delta v= \frac{1}{2a^2}$. 
# 
# The product of uncertainties after remembering that we calculate the square of the values is 
# 
# $$\displaystyle \Delta t\Delta v = 1/2$$ 
# 
# which is the minimum possible value.
# 
# #### (b) Decaying excited state
# 
# An excited electronic state can decay by emitting a photon, fluorescence if the transition is allowed or phosphorescence if from a triplet excited state to a singlet ground state.  The time profile can be measured as can the spectral width of the transition and they demonstrate the time - energy/frequency relationship. The excited state has a lifetime $\tau$ and transition frequency $\omega_0$. A spatial analogue is momentum broadening as a result of collisions, in this case the lifetime is replace by the mean free path.
# 
# The field of the emitted photon is $\displaystyle f(t)=e^{i\omega_0 t-t/2\tau}$ which is that of a plane wave of frequency $\omega_0$ that decays away (or is damped) with a lifetime $\tau$. The detector measure the 'square' of this which is $f(t)^*f(t) = e^{-t/\tau}$. 
# 
# The transform is 
# 
# $$\displaystyle g(\omega)= \frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty f(t)e^{-i\omega t} dt = \sqrt{\frac{2}{\pi}}\left(\frac{1}{2i (\omega-\omega_0)+1/\tau}\right)$$
# 
# making the measured spectral profile, which has the shape of a Lorentzian curve,
# 
# $$\displaystyle g(\omega)^*g(\omega)=\frac{2}{\pi}\left( \frac{1}{4(\omega-\omega_0)^2+1/\tau^2} \right) $$
# 
# The full width at half maximum of this curve is $1/\tau$ which makes $\Delta t\Delta \omega = \tau/\tau=1$.  Figure 54 in the answer to question 7 shows the exponential decay and the Lorentzian curves. As the energy is related to the frequency as $\Delta E =\hbar \Delta \omega$ then $\tau\Delta E = \hbar$
# 
# The calculation using $\displaystyle \int g(\omega)^2 d\omega$ as in the previous example will not work here because this integral is infinite.
# 
# #### (c) Finite wave-train
# 
# If a plane wave of frequency $\omega_0$  exists for a short time $\pm a$ and is zero elsewhere then over this range $\displaystyle f(t)=e^{i\omega t};\quad -a\le t \le a$, see figure 19 where $a=10$.  The spread of the wave $\Delta x = a$.  The 'square' of $f$ is $f(x)^*f(x)=1$
# 
# The transform has the form of a sinc function 
# 
# $$\displaystyle g(\omega)=\int_{-\infty}^\infty f(t)e^{-i\omega t}dt= \sqrt{\frac{2}{\pi}}\frac{\sin\left( a (\omega-\omega_0)\right)}{\omega-\omega_0}$$
# 
# The function $g^2$ cannot be normalised easily as it produces another integral (the Sine integral) but it is clear that the zeros of the function are at the same place in $g$ and its square and so we can take the spread $\Delta \omega$ to be the value from $\omega_0$ to the first minimum. The zeros of the sinc function occur at $\mathrm{sinc}(n\pi)$ where $n$ is an integer greater than zero, thus the first zero occurs at $\pm \pi/a$ from the central frequency.
# 
# If this zero is associated with $\Delta \omega$ then $\Delta \omega =\pi/a$ and the product $\Delta x\Delta \omega= \pi$.
# 
# #### Heisenberg Uncertainty
# 
# The relationship  $\Delta \omega\Delta x \ge 1$ as described in the last example above is not an inherent property of quantum mechanics but is a property of Fourier Transforms. The last example shows that  it is not possible to form a train of electromagnetic  waves for which it is possible to measure, at the same time, the position and wavelength with infinite accuracy. 
# 
# However, when considering quantum mechanics a particle is given wavelike properties via the de-Broglie relation. Such a material particle (such as an electron or a molecule) of energy $E$ and momentum $\pmb p$ is now associated with a wave of angular frequency $\omega=2\pi\nu$ and wavevector $\pmb k$ i.e. $E=\hbar\omega; \pmb p=\hbar \pmb k$ and leads to $\Delta x\Delta p \ge \hbar/2$. The fact that the Schroedinger equation is linear and homogeneous means that for particles a superposition principle applies which gives them wavelike properties. The small value of Planck's constant makes the limitations of the uncertainty principle totally negligible for anything macroscopic, i.e. greater than approx micron size. 

# ### 6.7 Summary of some  Fourier transform properties
# 
# $$\displaystyle 
# \begin{array}{ll}
# \hline
# \text{The transform pair is} &   f (x) \rightleftharpoons g(k)\\[0.15cm] 
# \text{Shift or delay }     &   f (x - x_0)  \rightleftharpoons e^{- ikx_0}g(k)\\[0.15cm]     
# \text{frequency shift} &   f (x)e^{ik_0x} \rightleftharpoons g(k - k_0) \\[0.15cm]
# \text{Reversal }             &   f (-x) \rightleftharpoons g(-k) \\[0.15cm]
# \text{Complex Conjugate }    &   f (x)^* \rightleftharpoons g(k)^* \\[0.15cm]
# \text{Scaling}               & \displaystyle   f(ax)\rightleftharpoons \frac{g(k/a)}{ |a|}\\[0.15cm]
# \text{Derivative}            & \displaystyle   f'(x)\rightleftharpoons 2\pi\,i\,kg(k)\\[0.15cm]
# \hline
# \end{array}
# $$
