#!/usr/bin/env python
# coding: utf-8

# In[1]:





# **Q17** Calculate Fig. 17, and confirm the numbers in the calculation.
# 
# **Strategy:** Use the code and parameters given in the text. Vary the rate constants to get an idea of the sensitivity of the fit to these values.
# 
# **Q18** Use Python's 'solve' routine routine to integrate the SIR equations using the data in Fig. 11.17, but automate the fitting by using a very simple, but inefficient,  grid search to vary $k_2$ and $k_1$ with $10$ values each, incrementing each value and so find the optimum values. Use two do loops to do this, setting back to the initial values one of the rate constants after the inner loop is completed but allowing the other to vary. Estimate the 'goodness' of each fit by calculating the *residual* which is the sum of the square of the difference between each of  the calculated values and the corresponding data point; $\sum_i (calculated_i - data_i )^2$ . Make a contour plot to determine the range where the best parameters fitting the data can be found.
# 
# The square of the difference is taken in the residual because some differences may be  negative. As smaller values of the residual are found, narrow the search area to 'home in' on the best estimates. The grid search is a slow but effective way of searching.
# 
# Using the SIR model, confirm that $k_2$ is in the range $0.001 \to 0.004\,\mathrm{ day^{-1}}$ and $k_1$ is in range $0.2 \to 0.6\,\mathrm{ day^{-1}}$.
# 
# 
# <img src='num-methods-fig18.png' alt='Drawing' style='width:250px;'/>
# 
# Figure 18. grid search.
# _____
# 
# **Q19** Recalculate equation 32 but with initial values $x_0 = 2,\; y_0 = 1$, and $t_0 = 0$ over the range $0 \to 10$ and then with the modified Euler method. The result is unexpected, increase $N$ and the precision of the calculation. Try to explain why the calculation fails. Repeat with Python's built-in routine.
# 
# 
# **Q20** A group of soldiers under $21$ years old who were all camping in the same field and doing the same work contracted influenza. The number infected was recorded on consecutive days, starting at day zero, as follows:
# 
# $1, 5, 25, 28, 10, 18, 15, 7, 8, 3, 2, 2, 1, 0, 0$.
# 
# The infectious period was reported as $3.5$ days. Calculate whether the epidemic follows the SIR model, and if so, estimate the number of susceptible and infected soldiers over a period of $15$ days.
# 
# **Strategy** To fit the data with rate constants $k_1$ and $k_2$ you will have to estimate values by trial and error. The grid search method could also be used; see question Q18. The data is somewhat noisy, so some judgment is needed as to what range of parameters will fit the data.
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




