---
layout: post
title: "Flow Matching Explained"
date: 2025-11-19
---
## Preliminaries
We start with the notations and definitions:

$$
\begin{align*}
\text{Data point: } &x=(x^1,x^2,...,x^d) \in \mathbb{R}^d \\
\text{Probability density path(time-dependent): } &p: [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}_{>0} \\
\text{Vector field(time-dependent): } &v: [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d
\end{align*}
$$
Now we can construct a time-dependent diffeomorphic map using a vector field $v_t$, via ordinary differential equation (ODE):

$$
\begin{align*}
\text{Flow(time-dependent): } &\phi: [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d \\
\frac{d}{dt}\phi_t(x) &= v_t(\phi_t(x))  \\
\phi_0() &= x
\end{align*}
$$


