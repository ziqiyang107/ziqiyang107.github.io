---
layout: post
title: "Scored-based Generative Models"
date: 2025-09-01
---
## Langevin Dynamics
We first start from the discrete $\textbf{Langevin Diffusion}$ process in the paper, introduced in {% include cite.html key="song2019generative"%}. Suppose we want to sample from a probability density $p(\vecx)$, the following recursion will converge to this distribution


<a id="eq1"></a>
<a id="eq-sde"></a>

$$
\begin{align*}
\vecx_t &= \vecx_{t-1} + \frac{\epsilon}{2}\nabla_{\vecx}\log{p(\vecx_{t-1})} + \veceta_t\\
\veceta_t &\sim N(0, \epsilon \vecI)
\end{align*}
$$

If we write $\veceta_t = \sqrt{\epsilon}\vecz_t$, and let $\frac{\epsilon}{2}=h$, then we have  
<div id="eq2">
$$
\begin{align*}
\vecx_t &= \vecx_{t-1} + h\nabla_{\vecx}\log{p(\vecx_{t-1})} + \sqrt{2h} \vecz_t  \tag{2} \\
\vecz_t &\sim N(0, \vecI)
\end{align*}
$$
</div>
The above recurision can be see as the discretization of Langevin diffusion, which is defined as the $\textbf{Stochastic Differential Equation (SDE)}$
<div id="eq3">
$$
d\vecX_t = -\nabla f(\vecX_t)\, dt + \sqrt{2}\, d\vecW_t  \tag{3}
$$
</div>
The discrete version Equation [(2)](#eq2) can be obtained from continuous version Equation [(3)](#eq3), by setting time to be $t+h$ and $t$, and use the definition of the Brownian motion $\vecW_t$. Note that Langevin dynamics itself was introduced in {% include cite.html key="neal2011mcmc"%}as a type of MCMC technique.

Now it's natural to ask whether the continuous version and the discrete version converge, the proof of convergence of the continuous version can be found in {% include cite.html key="roberts1996exponential"%}Theorem 2.1, which basically requires $e^{-f}$ suitably smooth. The convergence of the discrete vertion [(2)](#eq2) requires an additional Metropolis-adjusted step {% include cite.html key="neal2011mcmc"%}{% include cite.html key="roberts1998optimal"%}, in order to make sure the discreted version will converge to true density $p$, and this is due to the detailed balance convergence condition in Markov chains. However, in practice, due to its insignificant effect on the convergence, and it's costly to compute, people just ignore the Metropolis correction step {% include cite.html key="chen2014stochastic"%}. The stochastic gradient variant of the langevin dynamics can be found in {% include cite.html key="welling2011bayesian"%}, where authors introduced it to avoid the computation of the gradient on full dataset in the Bayesian framework.

Although the unadjusted discrete Langevin diffusion is biased, according to the proof in {% include cite.html key="chewi2025logconcave"%}, when the step size $h$ is small and the total iteration number $T$ is large, we can still conclude that $\vecx_T$ will be asymptotically distributed as probability density $p(\vecx)$. Now write $p(\vecx)$ as $p_{data}(\vecx)$, and if we want to generate data samples from this unknown distribution, for example, some face image distribution, using [(2)](#eq2), which now becomes:
$$
\begin{align*}
\vecx_t &= \vecx_{t-1} + h\nabla_{\vecx}\log{p_{data}(\vecx_{t-1})} + \sqrt{2h} \vecz_t  \tag{2} \\
\vecz_t &\sim N(0, \vecI)
\end{align*}
$$
However, $\log{p_{data}(\cdot)}$ term is still unknown, so we would like to estimate it, and one way to estimate is called $\textbf{Score Matching}$ {% include cite.html key="hyvarinen2005estimation"%}


## Score Matching
Score matching was originally proposed in {% include cite.html key="hyvarinen2005estimation"%}, as a way of estimating non-normalized density models $q(\vecx;\vectheta)$, where $p(vex;\vectheta)=\frac{1}{Z(\vectheta)}q(\vecx;\vectheta)$, and $Z(\vectheta)$ is the normalizing constant. Since $\nabla_{\vecx} \log{p(vex;\vectheta)}=\log{q(vex;\vectheta)}$, the normalizing constant is cancalled, so that people don't need to know it explicitly, since the integral $Z(\vectheta)=\int_{x \in \mathbb{R}^n} q(\vecx; \vectheta)d\vectheta$ is often hard to compute. 



score-based generative modeling through stochastic differential equations

---
{% include bibliography.html keys="chen2014stochastic,chewi2025logconcave,hyvarinen2005estimation,neal2011mcmc,roberts1996exponential,roberts1998optimal,song2019generative,welling2011bayesian" %}
