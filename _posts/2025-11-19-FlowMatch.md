---
layout: post
title: "Flow Matching Explained"
date: 2025-11-19
---
## Preliminaries
We start with the notations and definitions:

$$
\begin{align*}
\text{Data point } &x=(x^1,x^2,...,x^d) \in \mathbb{R}^d \\
\text{Probability density path } &p: [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}_{>0} \\
\text{Vector field } &v: [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d
\end{align*}
$$

Now we can construct a time-dependent diffeomorphic map using a vector field $v_t$, via ordinary differential equation (ODE):

$$
\begin{align*}
\text{Flow } \phi &: [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d \\
\frac{d}{dt}\phi_t(x) &= v_t(\phi_t(x))  \\
\phi_0(x) &= x
\end{align*}
$$


--------------------------

<div id="eq3">
$$
\begin{align*}
\vecx_t &= \vecx_{t-1} + h\nabla_{\vecx}\log{p_{data}(\vecx_{t-1})} + \sqrt{2h} \vecz_t  \tag{3} \\
\vecz_t &\sim N(0, \vecI)
\end{align*}
$$
</div>
However, $\log{p_{data}(\cdot)}$ term is still unknown, so we would like to estimate it, and one way to estimate is called $\textbf{Score matching}$ {% include cite.html key="hyvarinen2005estimation"%}

distribution, using [(1)](#eq1), which now becomes:

---
{% include bibliography.html keys="chen2014stochastic,chewi2025logconcave,hyvarinen2005estimation,neal2011mcmc,roberts1996exponential,roberts1998optimal,song2019generative,song2020sliced,vincent2011connection,welling2011bayesian" %}




