#!/usr/bin/env python
# coding: utf-8

# # Questions 10 - 13
# 
# ## Q10 Poisson distribution
# Show that the Poisson distribution is normalised and that the mean and variance of the Poisson distribution are both $\mu$ the average number of events.
# 
# **Strategy** The summation $k^2p(k,\mu)$ is best tackled by splitting into two parts by letting $k^2=k(k-1) +k$. 
# 
# ## Q11 Fit to poisson distribution
# In 1898, Bortkiewicz, a Russian statistician, published data on the number of deaths by horse kicks in Prussian cavalry units accumulated over a 20 year period. He used Poisson's formula to work out how many deaths you would expect to see. The table shows the raw data as the number killed and the number of cavalry units in which this occurred. If this data follows Poisson's distribution what is $\mu$? Calculate the $\chi^2$ and test for significance.
# 
# $$\displaystyle \begin{array}{c | c}
# \text{No. deaths} & \text{No. Units}\\
# \hline
# 0 & 109\\1 & 65\\2 & 22\\3 & 3\\4 & 1\\5&0\\6&0\\ \hline \end{array}$$
# 
# 
# ## Q12 Poisson distribution and bombings
# In June 1944, during WWII, the Germans started sending V1 Flying bombs to London, nicknamed "Doodlebugs" by locals. A doodlebug was a self propelled bomb with wings and it looked like a small aeroplane but one without a pilot. After the war to understand the clustering of where the bombs fell, the statistician R. D. Clarke split the map into a $12 \;\mathrm{km}\times 12\;\mathrm{ km}$ area with a  grid of $500$ m on a side. The number of bombs in each square is recorded in the table. This was done to see if there was a plan behind where bombs landed or whether they were arriving at random. Confirm that the data follows a Poisson distribution, calculate the $\chi^2$ and test for significance.
# 
# $$\displaystyle \begin{array}{c|c} 
# \text{Bombs per square} & \text{No. of squares}\\
# \hline
# 0 & 229\\
# 1 & 211\\
# 2 & 93\\
# 3 & 35\\
# 4 & 7\\
# >= 5 &1\\ \hline \end{array}$$
# 
# ## Q13 Dissociation energy
# The dissociation energy of a molecule can be calculated by adding up all the differences in vibrational energy until the energy gap is zero. However, it is not always possible to obtain data up to the dissociation limit and the quantum number for this has to be estimated which can be done by plotting the energy gaps vs quantum number and fitting the function to a polynomial. The dissociation energy is the area under the curve.
# Using the data for the oxygen molecule's energy levels given in the text (Section 6.2), calculate the dissociation energy of the $^3\sum_u^-$ state.
# 
# **Strategy:** Using the best functional form found in the example used in Section 6.2, calculate the energy differences $G(n + 1) - G(n)$ to find out what polynomial should be used. Plot $\Delta G_{n+1,n}$ vs $n$, and fit this data. Integrate the resulting equation to find the area but only until the quantum number $n$ that makes $\Delta G_{n+1,n} \approx 0$.

# In[ ]:




