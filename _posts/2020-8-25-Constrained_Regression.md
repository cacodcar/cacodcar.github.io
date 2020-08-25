---
layout: post
mathjax: true
title: Constrained Min Norm
date: 2020-08-25
category:
  - Blog
---

# Constrained Min Norm - Linear Equality Constraints

Lately I have been reading on how to regress data with constraints on the fitting parameters, mainly for applications in data science. In this post we will look at how to solve optimization problems of the following form.

$$\min_{x} ||x||^2, \text{ s.t.: } Cx = d$$

The Lagrange function of this problem is the following

$$L(x,\lambda) = x^T\mathcal{I}x + \sum_{i=0}^n\lambda_i(C_ix-d_i)$$

The Optimality Conditions derived from KKT conditions are

$$\frac{\partial \mathcal{L}}{\partial x_j}(x^{*}, \lambda) = 0, \text{where } j\in\{0,\dots m} $$

$$\frac{\partial \mathcal{L}}{\partial \lambda_i}(x^{*}, \lambda) = 0, \text{where } i\in\{0,\dots n}$$

By evaluating the partial derivatives, we can see the problem take shape.

$$\frac{\partial \mathcal{L}}{\partial x}(x^{*}, \lambda) = \frac{\partial}{\partial x}\left( x^T\mathcal{I}x + \lambda^T(Cx-d)\right) = 2\mathcal{I}x^* + C^t\lambda$$
$$\frac{\partial \mathcal{L}}{\partial \lambda}(x^{*}, \lambda) = \frac{\partial}{\partial \lambda}\left( x^T\mathcal{I}x + \lambda^T(Cx-d)\right) = Cx^* = d$$

So solving the optimization problem is the same thing as solving the following linear system!

$$
\begin{align*} 
& 2\mathcal{I}x - C^T\lambda &=  0 \\\\ 
& Cx  &=  d
\end{align*}
$$

Here is how I solved this Python 3.7  with numpy. Numpy is quite fast, a problem with 20 constraints and $x\in\mathcal{R}^{100}$ took  on average 0.15 seconds to solve on my desktop.

```python
import numpy

def min_norm_solve(C:numpy.ndarray, d:numpy.ndarray, return_multipliers:bool = True) -> numpy.ndarray:
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
