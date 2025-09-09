---
layout: post
title: "Scored-based Generative Models"
date: 2025-09-01
---
$$
\newcommand{\eqref}[1]{<a href="#eq#1">(\ref{#1})</a>}
$$

## Langevin Dynamics
We first start from the discrete Langevin diffusion process for stochastic gradient update, introduced in {% include cite.html key="welling2011bayesian"%}. Suppose we want to sample from a probability density $p(\vecx)$, the following recursion will converge to this distribution

$$
\begin{align*}
\vecx_t &= \vecx_{t-1} + \frac{\epsilon}{2}\nabla_{\vecx}\log{p(\vecx_{t-1})} + \eta_t \tag{1} \\
\eta_t &\sim N(0, \epsilon \vecI)
\end{align*}
$$

If we write $\eta_t = \sqrt{\epsilon}z_t$, and let $\frac{\epsilon}{2}=h$, then we have  

$$
\begin{align*}
\vecx_t &= \vecx_{t-1} + h\nabla_{\vecx}\log{p(\vecx_{t-1})} + \sqrt{2h} z_t  \tag{2} \\
z_t &\sim N(0, \vecI)
\end{align*}
$$  

The above recurision can be see as the discretization of Langevin diffusion, which is defined as the stochastic differential equation(SDE):

<a name="eq3"></a>

$$
d\vecX_t = -\nabla f(\vecX_t)\, dt + \sqrt{2}\, d\vecW_t  \tag{3}
$$

Comparing Equation [(3)](#eq3) with



Note that Langevin dynamics itself was introduced in {% include cite.html key="neal2011mcmc"%}as a type of MCMC technique


## Score Matching
### Score-based Generative Models
Generative Modeling by estimating gradients of the data distribution  
score-based generative modeling through stochastic differential equations

---
{% include bibliography.html keys="neal2011mcmc,welling2011bayesian" %}
