# Info:

# Author:
# Date

# Purpose:

# inputs:

# outputs:

# Version control:

#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Plot activation functions

# Create a range of x
x = np.arange(-10, 10)

# linear activation function
a = 1
y = a*x

plt.subplots() # open a new plot
plt.plot(x, y)
plt.title("Linear")
plt.xlabel ("x")
plt.ylabel ("F(x)")

# Sigmoid activation function
y = 1 / (1 + np.exp(-x))

plt.subplots() # open a new plot
plt.plot(x, y)
plt.title("Sigmoid")
plt.xlabel ("x")
plt.ylabel ("F(x)")

# Sigmoid activation function
y = np.tanh(x)

plt.subplots() # open a new plot
plt.plot(x, y)
plt.title("tanh")
plt.xlabel ("x")
plt.ylabel ("F(x)")




