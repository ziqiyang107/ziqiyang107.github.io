---
layout: post
title: "Scored-based Generative Models"
date: 2025-09-01
---

## Langevin Dynamics
We first start from the discrete Langevin diffusion process for stochastic gradient update, introduced in~\cite{welling2011bayesian}. Suppose we want to sample from a probability density $p(\vecx)$, the following recursion will converge to this distribution

$$
\begin{align*}
\vecx_t &= \vecx_{t-1} + \frac{\epsilon}{2} + \eta_t \\
\eta_t &\sim N(0, \epsilon)
\end{align*}
$$

Note that Langevin dynamics itself was introduced in{% include cite.html key="neal2011mcmc" %}as a type of MCMC technique


## Score Matching
### Score-based Generative Models
Generative Modeling by estimating gradients of the data distribution  
score-based generative modeling through stochastic differential equations

---
{% include bibliography.html keys="neal2011mcmc," %}
