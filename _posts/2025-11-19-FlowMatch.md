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
\text{Vector field(VF) } &v: [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d
\end{align*}
$$

Notice above $p_t$ and VF $v_t$ are both time-dependent, then we can construct a time-dependent diffeomorphic map using a VF $v_t$, via $\textbf{Ordinary differential equation (ODE)}$:

$$
\begin{align*}
\text{Flow } \phi : [0,1] \times \mathbb{R}^d &\rightarrow \mathbb{R}^d \quad \text{defined via an ODE:}\\
\frac{d}{dt}\phi_t(x) &= v_t(\phi_t(x))  \\
\phi_0(x) &= x
\end{align*}
$$

If we let $\phi_t(x)$ acts upon $x$, where $x \sim p_0$ ($p_0$ is a simple noise distribution), and then let $p_t$ denote the probability density function of $\phi_t(x)$, then this $p_t$ is a probability density path in $t$, we denote the $\textbf{push-forward equation}$:
<div id="eq1">
$$
\begin{align*}
$$
p_t = [\phi_t]_{*}p_0,
\end{align*}
$$
</div>

where the push-forward(a.k.a change of variables) operator $*$ is defined by:
<div id="eq2">
$$
\begin{align*}
[\phi_t]_{*}p_0(x) := p_0(\phi_t^{-1}(x))\text{det}\Big[\frac{\partial \phi_t^{-1}}{\partial x}(x) \Big]
\end{align*}
$$
</div>
We say $\textbf{a VF $v_t$ generates a probability density path $p_t$ if its flow $\phi_t$ satisfies [(1)](#eq1)}$


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

distribution, using [(3)](#eq3), which now becomes:

---
{% include bibliography.html keys="chen2014stochastic,chewi2025logconcave,hyvarinen2005estimation,neal2011mcmc,roberts1996exponential,roberts1998optimal,song2019generative,song2020sliced,vincent2011connection,welling2011bayesian" %}




