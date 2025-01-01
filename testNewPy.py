#import necessary modules, set up the plotting
import numpy as np
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
import matplotlib; matplotlib.rcParams['figure.figsize'] = (8,6)
from matplotlib import pyplot as plt
import GPy
print("done 1")
#———————————————————————————————————

m = GPy.examples.regression.sparse_GP_regression_1D(plot=False, optimize=False)
# This line creates a sparse GP regression model without plotting or optimizing it initially, providing a starting point for further exploration of model parameters and behaviors.
print(m)
print("done 2")
#———————————————————————————————————

m.rbf.lengthscale = 0.2
print(m)
print("done 3")
#———————————————————————————————————

print("Values of '.*var':", m['.*var'])
# m['.*var']: This line uses a regular expression .*var to index the model m. The pattern .*var matches any parameter whose name contains the substring "var". In this context, it matches parameters related to variance.

#print "variances as a np.array:", m['.*var'].values()
#print "np.array of rbf matches: ", m['.*rbf'].values()
print("done 4")
#———————————————————————————————————

m['.*var'] = 2.
print(m)
m['.*var'] = [2., 3.]
print(m)
print("done 5")
#———————————————————————————————————

print(m[''])
print("done 6")
#———————————————————————————————————

new_params = np.r_[[-4,-2,0,2,4], [.1,2], [.7]]
print(new_params)

m[:] = new_params
print(m)    
print("done 7")
#———————————————————————————————————
m.inducing_inputs[2:, 0] = [1,3,5]
print(m.inducing_inputs)
print("done 8")
#———————————————————————————————————

precision = 1./m.Gaussian_noise.variance
print(precision)
print("done 9")

#———————————————————————————————————

print("all gradients of the model:", m.gradient)
print("gradients of the rbf kernel:", m.rbf.gradient)
print("done 10")
#———————————————————————————————————

m.optimize
print(m.gradient)
print("done 11")
#———————————————————————————————————

m.rbf.variance.unconstrain()
print(m)
print("done 12")
#———————————————————————————————————

m.unconstrain('')
print(m)
print("done 13")
#———————————————————————————————————

m.inducing_inputs[0].fix()
m.rbf.constrain_positive()
print(m)
m.unfix()
print(m)
print("done 14")
#———————————————————————————————————

m.Gaussian_noise.constrain_positive()
m.rbf.constrain_positive()
m.optimize
print("done 15")
#———————————————————————————————————

fig = m.plot()
print("done 16")
#———————————————————————————————————

import numpy as np

# Assuming 'm' is a GP regression model
X = np.linspace(-3, 3, 100)[:, None]  # Example input range
mean, variance = m.predict(X)

# Plot using matplotlib
plt.figure()
plt.plot(X, mean, 'b-', label='Mean prediction')
plt.fill_between(X[:, 0], mean[:, 0] - 1.96 * np.sqrt(variance[:, 0]), mean[:, 0] + 1.96 * np.sqrt(variance[:, 0]), alpha=0.2, label='95% confidence interval')
plt.scatter(m.X, m.Y, c='r', marker='x', label='Data points')
plt.legend()
plt.show()

print("done 17")
#———————————————————————————————————

import GPy
import matplotlib.pyplot as plt

# Assuming 'm' is your model
fig, ax = plt.subplots()
m.plot(ax=ax)
plt.show()
print("done 18")
#———————————————————————————————————

import GPy
import plotly.graph_objects as go
import plotly.express as px

print("done 19")
#———————————————————————————————————

GPy.plotting.change_plotting_library('plotly')
print("done 20")
#———————————————————————————————————

import numpy as np
print("done 21")
#———————————————————————————————————

X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X) + np.random.randn(20,1)*0.05
print("done 22")
#———————————————————————————————————

kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.) 

print("done 23")
#———————————————————————————————————

#type GPy.kern.<tab> here:
GPy.kern.BasisFuncKernel
print("done 24")
#———————————————————————————————————

m = GPy.models.GPRegression(X,Y,kernel)
print("done 25")
#———————————————————————————————————

from IPython.display import display
display(m)
print("done 26")
#———————————————————————————————————

fig = plotly.subplots.make_subplots(rows=1, cols=1)
print(type(fig))

print("done 27")
#———————————————————————————————————

import plotly.graph_objects as go
import plotly.io as pio

# Hypothetical data extraction from a non-Plotly object
x_data = [1, 2, 3]
y_data = [4, 5, 6]

# Create a Plotly figure
fig = go.Figure(data=go.Scatter(x=x_data, y=y_data))

# Display the figure
pio.show(fig)

print("done 28")
#———————————————————————————————————

m.optimize(messages=True)
print("done 29")
#———————————————————————————————————

m.optimize_restarts(num_restarts = 10)
print("done 30")
#———————————————————————————————————

display(m)
fig = m.plot()
GPy.plotting.show(fig, filename = 'basic_gp_regression_notebook_optimized')
#This is the format of your plot grid:
#[ (1,1) x1,y1 ]
print("done 31")
#———————————————————————————————————

display(m)
fig = m.plot(plot_density=True)
GPy.plotting.show(fig, filename='basic_gp_regression_density_notebook_optimized')
print("done 32")