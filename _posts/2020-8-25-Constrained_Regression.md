---
layout: post
mathjax: true
title: Constrained Min Norm - Linear Equality Constraints
date: 2020-08-25
category:
  - Blog
---
Recently I have been reading on how to regress with constraints. It has uses in fitting models to data as well as signal reconstruction. In this post we will look at how to solve optimization problems of the following form.

$$\min_{x} ||x||^2, \text{ s.t.: } Cx = d$$

The Lagrange function of this problem is the following

$$L(x,\lambda) = x^T\mathcal{I}x + \sum_{i=0}^n\lambda_i(C_ix-d_i)$$

The Optimality Conditions derived from KKT conditions are

$$\frac{\partial \mathcal{L}}{\partial x_j}(x^{*}, \lambda) = 0, j\in\{0,\dots m\} $$

$$\frac{\partial \mathcal{L}}{\partial \lambda_i}(x^{*}, \lambda) = 0,i\in\{0,\dots n\}$$

By evaluating the partial derivatives, we can see the problem take shape.

$$\frac{\partial \mathcal{L}}{\partial x}(x^{*}, \lambda) = \frac{\partial}{\partial x}\left( x^T\mathcal{I}x + \lambda^T(Cx-d)\right) = 2\mathcal{I}x^* + C^t\lambda = \vec{0}$$
$$\frac{\partial \mathcal{L}}{\partial \lambda}(x^{*}, \lambda) = \frac{\partial}{\partial \lambda}\left( x^T\mathcal{I}x + \lambda^T(Cx-d)\right) = Cx^* = d$$

So solving the optimization problem is the same thing as solving the following linear system!

$$
\begin{align*} 
& 2\mathcal{I}x - C^T\lambda &=  \vec{0} \\\\ 
& Cx  &=  d
\end{align*}
$$

It can be solved by solving the orignial system or some manipulations can be made that rapidly accelterate this process.

Here is the source code for the Naive version in Python 3.7 and numpy.

```python
import numpy

def min_norm_solve_naive(C:numpy.ndarray, d:numpy.ndarray, return_multipliers:bool = True) -> numpy.ndarray:
    # get the shapes of the matrices
    num_x = C.shape[1]
    num_l = C.shape[0]
    
    # Build the system to solve
    A = numpy.block([[2*numpy.eye(num_x), C.T],[C, numpy.zeros((num_l,num_l))]])
    b = numpy.block([[numpy.zeros((num_x,1))],[d]])
    
    # solve linear system with numpy
    solution = numpy.linalg.solve(A, b)
      
    if return_multipliers:
        return solution
    
    return solution[:num_x]
  
```

We can rearange the system as follows

$$2\mathcal{I}x - C^T\lambda =  \vec{0} \rightarrow x = \frac{1}{2}C^T\lambda$$
$$Cx = d \rightarrow \frac{1}{2}CC^T \lambda = d$$
$$x = \frac{1}{2}C^T\lambda \rightarrow C^T(CC^T)^{-1}d$$

We can solve the multipliers system then substitute it back into the expresstion for x.

Here is the source code for the informed version in Python 3.7 and Numpy.

```python
import numpy

def min_norm_informed(C:numpy.ndarray, d:numpy.ndarray, return_multipliers:bool = True) -> numpy.ndarray:
    
    lagrange_multipliers = numpy.linalg.solve(.5*C@C.T, d)
    x = .5*C.T@lagrange_multipliers
    
    if return_multipliers:
        return numpy.block([[x],[lagrange_multipliers]])

    return x

```

This is much faster then the naive version as it solves a much smaller system of equations. For example, with 100 constraints and 1000 dimensions the informed version ran 80x faster then the naive version (.24 mS vs 19.3 ms) 

