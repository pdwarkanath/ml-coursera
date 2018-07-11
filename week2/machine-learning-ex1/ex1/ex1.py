# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 19:24:12 2018

@author: pdwarkanath
"""

#======================= Part 1: Import Libraries and Load Data =======================
import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt

data = np.matrix(np.loadtxt('ex1data1.txt',delimiter = ','))

X = np.array(data[:,0])
y = np.array(data[:,1])
m = len(y)


#======================= Part 2: Plotting =======================
print('Plotting Data ...\n')

def plotData(X,y):
    plt.scatter(X,y, marker = 'x', color = 'red', s= 20, linewidth = 1)
    plt.ylabel('Profit in $10,000s') # Set the y?axis label
    plt.xlabel('Population of City in 10,000s')  # Set the x?axis label
    plt.show()


plotData(X,y)

input('Program paused. Press enter to continue.')

#======================= Part 3: Cost and Gradient Function =======================

X =  np.mat([np.ones(m),X.flatten()]).T
theta = npm.zeros((2,1))

# Some gradient descent settings

iterations = 1500;
alpha = 0.01;

# Compute Cost Function

def computeCost(X,y, theta):
    m = len(y)
    d = X*theta - y
    J = d.T*d/(2*m)
    return J


print('\nTesting the cost function ...\n')

# compute and display initial cost

J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = {} \n'.format(J))
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function

theta = np.mat('-1 ; 2')
J = computeCost(X, y, theta)
print('\nWith theta = [-1 ; 2]\nCost computed = {} \n'.format(J))
print('Expected cost value (approx) 54.24\n')
input('Program paused. Press enter to continue.\n')
print('\nRunning Gradient Descent ...\n')


# Gradient Descent Function

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    for i in range(num_iters):
        theta = theta - alpha * X.T*(X * theta - y) / m
    return theta

# run gradient descent

theta = gradientDescent(X, y, theta, alpha, iterations)


# print theta to screen
print('Theta found by gradient descent:\n')
print('{} \n'.format(theta))
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')
