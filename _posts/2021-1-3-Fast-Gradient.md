---
layout: post
mathjax: true
title: Optimizing Optimization - Faster Gradients
date: 2021-1-4
category:
  - Blog
---

Most if not all gradient decent algorithms are characterized by the number of times the gradient of the function needs to be evaluated. So, a very practical thing to do is to look at how these are calculated and try to speed them up. 

As a case study I will look at speeding up the gradient for ridge regression. Ridge regression is least squares regression where the magnitude squared of the parameters are penalized in the regression. The following form is the mathematical description of ridge regression.

$$\min_x f(x) = ||y - Ax||^2 - \lambda ||x||^2$$

$$\nabla f(x) = -2A^T(y - Ax)+2\lambda x$$

We can see in the gradient expression there are multiple matrix operations that are dependent on each other. The outer multiplicaition by $A^T$ cannot start before the residual term $y - Ax$ is calculated and for various reasons this quite inefficient. Primarily due to the multiple multiplications by the same matrix in the expression, we are loosing out on data reuse (peices of the matrix get loaded in from main memory only once).  With the fact we are doing multiple operations and have performance degridation from intermediates. There is a way that we can ger around this problem, if we precompute some quantities then we can see a speed up by removing expensive temperaries and poor data use. We just need to factor it out.

$$B = -2A^Ty$$

$$M = 2(A^TA+\lambda \mathcal{I})$$

$$\nabla  f(x) = -2A^T(y - Ax)+2\lambda x$$

$$\nabla  f(x) = -2A^Ty +2(A^TAx+\lambda \mathcal{I}) x$$

$$\nabla  f(x) = B + Mx$$

In python this is written as the following. 

```python

def textbook_grad(x, lambda_):
  return -2*A.T@(y - A@x)+2*lambda*x

def informed_grad(x, lambda_):
  return M@x+B

```

The time complexity of textbook formula is $\mathcal{O}(nm + mn + n + m) \approx \mathcal{O}(2nm)$, where as the improved formula is $\mathcal{O}(m^2 + m) \approx \mathcal{O}(m^2)$, in the case that $m < n$ then we would have an asymptotic speed up of $~2n/m$. But due to the chache structure of the cpu, reducing the memory foodprint of the matrices required to calculate the gradient can massively increase the performance (e.g. everthing fitting into L2 vs L3 cache). 


Now these terms do have to be precomputed at the start but in my experience this is inconsequential. For a matrix A for size n by m, the asymptotic complexity of creating these is $\mathcal{O}(nm^2)$, where n is the number of observations and m is the number of features. While by the asymptotic analysis might look detrimental, for systems of practical size (n < ~20,000 and m <  ~1000) this system makes sense, I will cover this momentarily. These sort of constraints on the systems cover many practical problems in engineering, science, and statistics.

For a ridge regression problem where there are 10,000 observations and 500 features the timing differences between textbook and informed gradient are quite shocking.

```python
%timeit textbook_grad(x, lambda_)
%timeit informed_grad(x, lambda_)
```

```python
textbook_grad -> 11.4 ms ± 211 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
informed_grad -> 11.2 µs ± 978 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
```

We have an almost 1000x speed up over the original form. In a setting where rapidly evaluating the gradient this is a significant improvement. I will show this speed up in action by comparing the times for a Nesterov based gradient decent algorithm with and without fast gradient.

```python 

def nesterov_desc(x,lambda_,  df, L, max_iter = 10000) -> list:
    y_0 = x.copy()
    y_1 = x.copy()
    x_0 = x.copy()
    x_1 = x.copy()
    
    x_list = list()
    
    for i in range(max_iter):
        #compute grad_f
        grad = df(y_0, lambda_)
        
        #update x_{t+1}
        x_1 = y_0 - (1.0/L)*grad
        #update y_{t+1}
        y_1 = 2*x_1 -x_0
        
        #reset momentum if no longer going down
        if grad.T@(x_1 - x_0) > 0:
            y_0 = x_0.copy()
            y_1 = x_0.copy()
            x_0 = x_0.copy()
            x_1 = x_0.copy()
        
        #step forward
        y_0 = y_1
        x_0 = x_1
        
        #add step to list
        x_list.append(x_0)
        
    return x_list
```

We can now benchmark the resulting algorithm. The strength of the informed gradient can be seen here. Clearly the matrix-matrix multiplication is cheap, compared to the time savings we have here (the time to compute M and B are ~10 ms).

```python
%timeit p = nesterov_desc(x_0, 1, textbook_grad, L)
%timeit q = nesterov_desc(x_0, 1, informed_grad, L)
```

```python
textbook_grad -> 12.1 s ± 403 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
informed_grad -> 22.3 ms ± 1.04 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

These sorts of speed ups don't come every day, and we must take hold of them when possible. I should stress, that the evaluated gradients at every point are the same, that the difference we are seeing here is only on the speed up of computing the gradient.





