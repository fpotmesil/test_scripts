#================================================================
# Fred Potmesil
# Sept 09, 2024
# all code taken from:
# Deep Learning From Scratch
#
#
#================================================================

import numpy as np
from numpy import ndarray

import matplotlib.pyplot as plt
import matplotlib

#
# FJP TODO - move machine learning functions to another file and then import
# FJP TODO - move matplotlib graphing functions to another file and then import
#
# if the file name is ml_functions:
# import ml_functions as ml_funcs
#

#
# nested functions using Callable and List
#
from typing import Callable
from typing import List
from typing import Dict
#------------------------------------------------------------
#
# nested function prelim setup:
#
# 'Array_Function' takes in an ndarray as an argument and 
# produces an ndarray as a result.
#
Array_Function = Callable[[ndarray], ndarray]
# 
# A 'Chain' is a list of 'Array_Function'(s)
#
Chain = List[Array_Function]
#
#------------------------------------------------------------
#------------------------------------------------------------

#------------------------------------------------------------
#
# Evalutes two functions in a row, 'chained' functions
# Length of input chain must be 2
#
#------------------------------------------------------------
def chain_length_2( 
        chain: Chain,
        x: ndarray ) -> ndarray:

    assert len(chain) == 2, \
    "This function requires 'Chain' objects of length 2"
    
    f1 = chain[0]
    f2 = chain[1]

    return f2( f1(x) )
#------------------------------------------------------------
#------------------------------------------------------------

#-----------------------------------------------------------------------
#
# Use the chain rule to compute the derivatives of two nested functions:
# (f2(f1(x))' = f2'(f1(x)) * f1'(x)
#
#-----------------------------------------------------------------------
def chain_deriv_2( 
        chain: Chain,
        input_range: ndarray ) -> ndarray:

    assert len(chain) == 2, \
    "This function requires 'Chain' objects of length 2"

    assert input_range.ndim == 1, \
    "Function requires a 1 dimensional ndarray as input_range"

    f1 = chain[0]
    f2 = chain[1]

    # df1/dx
    f1_of_x = f1(input_range)

    # df1/du
    df1dx = deriv(f1, input_range)
    
    # df2/du(f1(x))
    df2du = deriv(f2, f1(input_range))

    return df1dx * df2du
#------------------------------------------------------------
#------------------------------------------------------------


#-----------------------------------------------------------------------
#
# Plots a chain function - a function made up of multiple consecutive
# ndarray -> ndarray mappings - across the input_range
# 
# argument 'ax': matplotlib Subplot for plotting
#
#-----------------------------------------------------------------------
def plot_chain(ax,
        chain: Chain,
        input_range: ndarray) -> None:

    assert input_range.ndim == 1, \
            "Function requires a 1 dimensional ndarray as input_range"

    output_range = chain_length_2(chain, input_range)
    ax.plot(input_range, output_range)
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
#
# Uses the chain rul to plot the derivative of a function consisting of
# two nested functions.
# 
# argument 'ax': matplotlib Subplot for plotting
#
#-----------------------------------------------------------------------
def plot_chain_deriv(ax,
        chain: Chain,
        input_range: ndarray) -> ndarray:

    output_range = chain_deriv_2(chain, input_range)
    ax.plot(input_range, output_range)
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

'''
---------------------------------------------------------------
Apply the sigmoid function to each element in the input ndarray
---------------------------------------------------------------
'''
def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))


'''
--------------------------------------------
Square each element in the input ndarray
--------------------------------------------
'''
def square(x: ndarray) -> ndarray:
    return np.power(x, 2)

'''
------------------------------------------------------------
Apply "Leaky ReLU" function to each element in input ndarray
------------------------------------------------------------
'''
def leaky_relu(x: ndarray) -> ndarray:
    return np.maximum(0.2 * x, x)


'''
---------------------------------------------------------------
Evaluates the derivative of a function 'func' at every element
in the 'input_' array.
---------------------------------------------------------------
'''
def deriv(
        func: Callable[[ndarray], ndarray],
        input_: ndarray,
        delta: float = 0.0001) -> ndarray:
    return ( func(input_ + delta) - func(input_ - delta) ) / (2 * delta)



print( "Hello, lets play with python list operations!\n" )
a = [1,2,3,4]
b = [5,6,7,8]
c = [9,10,11,12]
d = [13,14,15,16]

try:
    print( "Normal python list adding:\n" )
    print( "a + b: ", a + b )
    print( "\n\n" )
except TypeError:
    print( "normal python lists cannot be added together!" )
    print( "\n\n" )

try:
    print( "Normal python list multiplication:\n" )
    print( "a * b: ", a * b )
    print( "\n\n" )
except TypeError:
    print( "normal python lists cannot be multiplied together!" )
    print( "\n\n" )


print( "OK, that was fun, now lets try with numpy lists!\n" )
arr1 = np.array([1,2,3,4])
arr2 = np.array([5,6,7,8])
print( "Adding numpy arrays a + b: ", arr1 + arr2 )
print( "Multiplying numpy arrays a * b: ", arr2 * arr2 )

arr3 = np.array( [a, b, c] )
print( "3 dimension numpy array:\n", arr3 )

print( 'arr3.sum(axis=0): ', arr3.sum(axis=0) )
print( 'arr3.sum(axis=1): ', arr3.sum(axis=1) )

print( "\n\n" )
arr4 = np.array( d )
print( "1 dimension numpy array:\n", arr4 )

print( "\n\n" )
print( "Adding arr3 and arr4:\n", arr3 + arr4 )

#---------------------------------------------------------------------
#
# graphing tests to show functions.
#
# displays graphs on 2 rows, 1 column
#
#---------------------------------------------------------------------
'''
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 6))  

input_range = np.arange(-2, 2, 0.01)
ax[0].plot(input_range, square(input_range))
ax[0].set_title('Square Function')
ax[0].set_xlabel('X input value')
ax[0].set_ylabel('Y result value')

ax[1].plot(input_range, leaky_relu(input_range))
ax[1].set_title('"ReLU" function')
ax[1].set_xlabel('X input value')
ax[1].set_ylabel('Y result value')
plt.show()
'''
#---------------------------------------------------------------------
#---------------------------------------------------------------------

#---------------------------------------------------------------------
#
# graphing tests to show chain rule and chain derivative functions.
#
# displays graphs on 2 rows, 1 column
#
#---------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16,8)) 

chain_1 = [square, sigmoid]
chain_2 = [sigmoid, square]

PLOT_RANGE = np.arange(-3, 3, 0.01)
plot_chain(ax[0], chain_1, PLOT_RANGE)
plot_chain_deriv(ax[0], chain_1, PLOT_RANGE)

ax[0].legend(["$f(x)$", "$\\frac{df}{dx}$"], loc='lower left')
ax[0].set_title("Function and derivative for\n$f(x) = sigmoid(square(x))$")

plot_chain(ax[1], chain_2, PLOT_RANGE)
plot_chain_deriv(ax[1], chain_2, PLOT_RANGE)

ax[1].legend(["$f(x)$", "$\\frac{df}{dx}$"], loc='lower left')
ax[1].set_title("Function and derivative for\n$f(x) = square(sigmoid(x))$")
plt.show()
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------


print( "\n\n" )
print( "Well, that is all, buh-bye now!\n" )


