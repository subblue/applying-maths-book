#!/usr/bin/env python
# coding: utf-8

# # 12. Monte Carlo Methods

# Numerically solving integrals or differential equations using a Monte Carlo method is done by repeatedly guessing the random values of one or more variables, for example, $x$ in $f(x)$ and assessing the result against some criteria. This method differs from other numerical calculations, which proceed by smoothly varying $x$ from start to finish. Although a Monte Carlo method uses random or 'stochastic' values, this is always done according to some algebraic formula or algorithm. However, whatever the exact method used, efficient or not, the final answer is only achieved after averaging many guesses and is therefore only an approximation to the true value.
# 
# The Monte Carlo methods fall into two broad areas; the first is the use of random numbers with a formula to perform integrations or solve differential equations; the second is the use of random numbers to *simulate* physical processes at an elementary or molecular level, usually because these processes are too difficult to solve in any other way.
