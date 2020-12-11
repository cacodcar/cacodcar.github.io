---
layout: post
mathjax: true
title: Model Predictive Control - State Space Model 
date: 2020-08-15
category:
  - Blog
---

With model predictive control, a mathematic optimization problem is solved to find the 'optimal' control needed to drive the sustem to a specified set point. A model of the system is enbedded into the constraints of the optimiation problem in the following way.

$$\min \sum_0^nx^T_kQx_k + \sum_0^{n-1}u^T_kRu_k$$

$$\text{s.t.  } x_{k+1} = g(x_k, u_k)$$

$$x_k \in \mathcal{X}$$

$$u_k \in \mathcal{U}$$

However this leads to a problem, if our function 'g' is nonlinear this becomes a much harder problem to solve in general. For most control applications the time it takes to solve the problem is significant constraint. If a process needs an action every second and it takes 5 seconds to solve the optimization problem then this is not helpful. To reduce the time to solve the optimization problem we need to find a simpler model to embedded into the problem.

One of the most popular model types is the State Space model. It is linear is seperable for the state and input parameters and the discrete time inverent form takes the following form.

$$x_{k+1} = Ax_k + Bu_k$$

While this does not, and can not include nonlinearities from the original system; the state space model usually does a good job for many systems. 






