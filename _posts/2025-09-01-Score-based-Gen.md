---
layout: post
title: "Scored-based Generative Models"
date: 2025-09-01
---
## Langevin Dynamics
We first start from the discrete $\textbf{Langevin diffusion}$ process in the paper, introduced in {% include cite.html key="song2019generative"%}. Suppose we want to sample from a probability density $p(\vecx)$, the following recursion will converge to this distribution


$$
\begin{align*}
\vecx_t &= \vecx_{t-1} + \frac{\epsilon}{2}\nabla_{\vecx}\log{p(\vecx_{t-1})} + \veceta_t\\
\veceta_t &\sim N(0, \epsilon \vecI)
\end{align*}
$$

If we write $\veceta_t = \sqrt{\epsilon}\vecz_t$, and let $\frac{\epsilon}{2}=h$, then we have  
<div id="eq1">
$$
\begin{align*}
\vecx_t &= \vecx_{t-1} + h\nabla_{\vecx}\log{p(\vecx_{t-1})} + \sqrt{2h} \vecz_t  \tag{1} \\
\vecz_t &\sim N(0, \vecI)
\end{align*}
$$
</div>
The above recurision can be see as the discretization of Langevin diffusion, which is defined as the $\textbf{Stochastic differential equation (SDE)}$
<div id="eq2">
$$
d\vecX_t = -\nabla f(\vecX_t)\, dt + \sqrt{2}\, d\vecW_t  \tag{2}
$$
</div>
The discrete version Equation [(1)](#eq1) can be obtained from continuous version Equation [(2)](#eq2), by setting time to be $t+h$ and $t$, and use the definition of the Brownian motion $\vecW_t$. Note that Langevin dynamics itself was introduced in {% include cite.html key="neal2011mcmc"%} as a type of MCMC technique.

Now it's natural to ask whether the continuous version and the discrete version converge, the proof of convergence of the continuous version can be found in {% include cite.html key="roberts1996exponential"%} Theorem 2.1, which basically requires $e^{-f}$ suitably smooth. The convergence of the discrete vertion [(1)](#eq1) requires an additional Metropolis-adjusted step {% include cite.html key="neal2011mcmc"%}{% include cite.html key="roberts1998optimal"%}, in order to make sure the discreted version will converge to true density $p$, and this is due to the detailed balance convergence condition in Markov chains. However, in practice, due to its insignificant effect on the convergence, and it's costly to compute, people just ignore the Metropolis correction step {% include cite.html key="chen2014stochastic"%}. The stochastic gradient variant of the langevin dynamics can be found in {% include cite.html key="welling2011bayesian"%}, where authors introduced it to avoid the computation of the gradient on full dataset in the Bayesian framework.

Although the unadjusted discrete Langevin diffusion is biased, according to the proof in {% include cite.html key="chewi2025logconcave"%}, when the step size $h$ is small and the total iteration number $T$ is large, we can still conclude that $\vecx_T$ will be asymptotically distributed as probability density $p(\vecx)$. Now write $p(\vecx)$ as $p_{data}(\vecx)$, and if we want to generate data samples from this unknown distribution, for example, some face image distribution, using [(1)](#eq1), which now becomes:

<div id="eq3">
$$
\begin{align*}
\vecx_t &= \vecx_{t-1} + h\nabla_{\vecx}\log{p_{data}(\vecx_{t-1})} + \sqrt{2h} \vecz_t  \tag{3} \\
\vecz_t &\sim N(0, \vecI)
\end{align*}
$$
</div>
However, $\log{p_{data}(\cdot)}$ term is still unknown, so we would like to estimate it, and one way to estimate is called $\textbf{Score matching}$ {% include cite.html key="hyvarinen2005estimation"%}


## Score Matching
Score matching was originally proposed in {% include cite.html key="hyvarinen2005estimation"%}, as a way of estimating non-normalized density models $q(\vecx;\vectheta)$, where $p(\vecx;\vectheta)=\frac{1}{Z(\vectheta)}q(\vecx;\vectheta)$, and $Z(\vectheta)$ is the normalizing constant. Since $\nabla_{\vecx} \log{p(\vecx;\vectheta)}=\nabla_{\vecx} \log{q(\vecx;\vectheta)}$, the normalizing constant is cancalled, people don't need to know it explicitly, since the integral $Z(\vectheta)=\int_{x \in \mathbb{R}^n} q(\vecx; \vectheta)d\vectheta$ is often hard to compute. 

The idea of score matching is fairly simple, we seek to minimize the objective:

<div id="eq-star">
$$
\frac{1}{2}\mathbb{E}_{p_{data}\,(\vecx)}\Big[\left\lVert\vecs_{\vectheta}(\vecx)-\nabla_{\vecx}\log{p_{data}(\vecx)}\right\rVert_2^2\Big], \tag{*}
$$
</div>

using a score neural network $\vecs_{\vectheta}(\cdot): \mathbb{R^d} \rightarrow \mathbb{R^d}$ parametrized by $\vectheta$. A simple trick of partial integration {% include cite.html key="hyvarinen2005estimation"%} can be used to show the above objective, which depends on the unknown data density, can be rewritten as follows:

<div id="eq4">
$$
J(\vectheta)=\mathbb{E}_{p_{data}\,(\vecx)}\Big[\text{tr}(\nabla_{\vecx}\vecs_{\vectheta}(\vecx)) + \frac{1}{2}\left\lVert\vecs_{\vectheta}(\vecx)\right\rVert_2^2 \Big]， \tag{4}
$$
</div>

where $\nabla_{\vecx}\vecs_{\vectheta}(\vecx)$ is the Jacobian of the $\vecs_{\vectheta}(\vecx)$. Furthermore, it's shown in {% include cite.html key="hyvarinen2005estimation"%}, that if the parametrized density include the real data density: $p_{data}(\cdot)=p(\cdot \,; \vectheta^{\*})$ for some $\vectheta^{\*}$, and with some other regularity conditions (including $p(\vecx;\vectheta)>0$ for all $\vecx, \vectheta$), then $J(\vectheta) = 0 \Longleftrightarrow \vectheta = \vectheta^{\*}$. This means if our score network $\vecs_{\vectheta}(\cdot)$ is powerful enough to cover the unknown true data density, and if our optimization algorithms can recover the true minimizer, then the optimized parameters will be the true parameters $\vectheta^{\*}$. However, in practice, this may not be true, as there may be several local minimum (see the discussion under Corollary 3 in {% include cite.html key="hyvarinen2005estimation"%} for further details). 

From Formula [(4)](#eq4), we can see it no longer depend on the unknown data density, and we can estimate it using samples from data to replace the expectation. Recall the above regularity conditions that include $p(\vecx;\vectheta)>0$ for all $\vecx, \vectheta$, if these conditions are satisfied, then if we sampled version of Formula [(4)](#eq4), the estimator obtained is consistent, i.e., it converges in probability to the true value of $\vectheta$ when sample size approaches infinity {% include cite.html key="hyvarinen2005estimation"%}. However, there is another problem of directly applying [(4)](#eq4), the term $\text{tr}(\nabla_{\vecx}\vecs_{\vectheta}(\vecx))$, involving a Jacobian term, can be hard to compute when dimension $d$ is large. One natural idea is to use a trace estimator, $E_{p_{\vecv}\, \sim N(\veczero,\, \vecI)}\Big[\vecv^T \nabla_{\vecx}\vecs_{\vectheta}(\vecx) \vecv \Big]$ {% include cite.html key="song2020sliced"%}, to estimate this term, this is known as $\textbf{Hutchinson}$ $\textbf{trace}$ $\textbf{estimator}$. 

## Score-based Generative Models
Now, it looks like we can just use the trace estimator to replace the trace term in loss [(4)](#eq4), and get $\nabla_{\vecx}\vecs_{\vectheta}(\vecx) \approx \nabla_{\vecx}\log p_{data}(\vecx)$, and use discrete Langevin diffusion [(3)](#eq3) to generate samples from the real data distribution, but there are some hidden problems that prevent us from doing this directly. As pointed out in {% include cite.html key="song2019generative"%}, many real world data tend to concentrate on low dimensional manifolds embedded in a high dimensional space (a.k.a, the ambient space), this is called $\textbf{manifold hypothesis}$. So under this hypothesis, the score $\nabla_{\vecx}\log p_{data}(\vecx)$ is the gradient defined in the ambient space, but it is undefined if $\vecx$ resides in a low dimensional manifold, due to the fact that when (i): $\vecx \notin \mathcal{M}$, then $p(\vecx)=0$, and $\log p(\vecx)=-\infty$, so this is non-differentiable, (ii): $\vecx \in \mathcal{M}$, the extension of $p_{data}(\vecx)$ outside $\mathcal{M}$ is zero, so $\nabla_{\vecx}\log p_{data}(\vecx)$ is still undefined. Furthermore, previous consistency result will non longer hold if the data reside on a low dimensional manifold, since the condition, including $p(\vecx;\vectheta)>0$ for all $\vecx, \vectheta$, does not satisty anymore. If a small Gaussian noise is added to the $p_{data}(\vecx)$ to let the data distribution support in the whole space, then the convergence of using sliced score matching {% include cite.html key="song2020sliced"%} will be greatly improved {% include cite.html key="song2019generative"%}. Another reason that may cause the difficulties for score estimation is existence of $\textbf{low data density regions}$, where the score estimation is inaccurate in these regions due to the scarcity of data samples. If modes of the the data distribution are separated by low density regions, using the $\nabla_{\vecx}\log p_{data}(\vecx)$ may ignore the difference between different modes, so the samples obtained will not depend on modes components. In the end, naive discrete Langevin diffusion can produce correct samples in theory, but may require a very small step size and a very large number of steps to mix {% include cite.html key="song2019generative"%}. 

From the above analysis, if we perturb the data distribution using a small Gaussian noise, we can alleviate the issue of manifold hypothesis, score estimation will be more stable, but the samples generated using Langevin diffusion will be the slightly perturbed distribution; if we perturb the data using a relatively large Gaussian noise, the data distribution will be more spread out, then we can alleviate the issue of existence of low density region, but samples generated will deviate from the origin data distribution by a large amount, to leverage the benefits of this insight while avoiding its drawbacks, the authors in {% include cite.html key="song2019generative"%} designed a clever approach: combining denoising score matching and annealing, using the following idea introduced in {% include cite.html key="vincent2011connection"%}.

$\textbf{Denoising score matching}$ {% include cite.html key="vincent2011connection"%} is a variant of the score matching, it completely circumvents the trace term in [(4)](#eq4), and it proposes an objective:
<div id="eq5">
$$
\begin{align*}
\frac{1}{2}\mathbb{E}_{q_{\sigma}(\tilde{\vecx}|\vecx)p_{data}\,(\vecx)} \Big[\left\lVert \vecs_{\vectheta}(\tilde{\vecx})-\nabla_{\tilde{\vecx}}\log q_{\sigma}(\tilde{\vecx}|\vecx) \right\rVert_2^2 \Big],  \tag{5}
\end{align*}
$$
</div>
where $q_{\sigma}(\tilde{\vecx}|\vecx)$ is a noise distribution, and $\tilde{\vecx}$ is the noise perturbed data point. It is shown in {% include cite.html key="vincent2011connection"%} that the above objective is equivalent to original score matching objective [(*)](#eq-star), but with $p_{data}$ replaced by $q_{\sigma}$:

$$
\frac{1}{2}\mathbb{E}_{q_{\sigma}\,(\vecx)}\Big[\left\lVert\vecs_{\vectheta}(\vecx)-\nabla_{\vecx}\log{q_{\sigma}(\vecx)}\right\rVert_2^2\Big],
$$

Now if $q_{\sigma}$ still satisfies regularity conditions discussed under [(4)](#eq4), then we have the minimizer of [(5)](#eq5) satisfies 

$$
\vecs_{\vectheta^{*}}(\tilde{\vecx})=\nabla_{\tilde{\vecx}}\log q_{\sigma}(\tilde{\vecx}) \overset{\Delta}{=} \nabla_{\tilde{\vecx}}\log \int q_{\sigma}(\tilde{\vecx}|\vecx)p_{data}(\vecx)d\vecx
$$

almost surely. Note that $\vecs_{\vectheta^{\*}}(\tilde{\vecx}) \approx \nabla_{\tilde{\vecx}}\log p_{data}(\tilde{\vecx})$ only if the noise $\sigma$ is small enough such that $q_{\sigma}(\tilde{\vecx})\approx p_{data}(\tilde{\vecx})$. 

Suppose we have a sequence of noise levels $\sigma_1 > \sigma_2 > ... >\sigma_L$, and we make the noise $\sigma_1$ large enough to mitigate the effect of manifold hypothesis, and make the noise $\sigma_L$ small enough to let $q_{\sigma_L} \approx p_{data}$, then we will get a sequence of noise-perturbed distributions $q_{\sigma_1}(\vecx), q_{\sigma_2}(\vecx), ..., q_{\sigma_L}(\vecx)$ that converge to true $p_{data}$, this intuition is inspired by simulated annealing. We fit a score network $\vecs_{\vectheta}(\vecx, \sigma)$ which conditions on different noise levels $\sigma_i$'s, and we want to make sure: $$\forall\sigma \in \{\sigma_{i}\}_{i=1}^{L}$$, we have $$\vecs_{\vectheta}(\vecx, \sigma)\approx \nabla_{\vecx}\log q_{\sigma}(\vecx)$$. After we optimize this score neural network, we can first generate samples by only a couple of steps using the optimized conditional score network $\vecs_{\vectheta}(\vecx, \sigma_1)$ using large noise level $\sigma_1$:
<div id="eq3-prime">
$$
\begin{align*}
\vecx_t &= \vecx_{t-1} + h\vecs_{\vectheta}(\vecx_{t-1}, \sigma_1) + \sqrt{2h} \vecz_t  \tag{3'} \\
\vecz_t &\sim N(0, \vecI),
\end{align*}
$$
</div>
since the perturbed score function $\nabla_{\vecx}\log q_{\sigma_1}(\vecx)$ will be estimated more accurately and will be less affected by manifold hypothesis and low data density regions, based on previous score matching regularity conditions discussion. We first get samples from largely perturbed $q_{\sigma_1}$, then we slowly anneal down the noise level, and finally to $q_{\sigma_L}$, which is indistinguishable from $p_{data}$, if we choose $\sigma_L$ sufficiently small. We will elaborate more on this in the later inference phase. 

## Training Score-based Generative Models
Choosing the noise distribution $q_{\sigma}(\tilde{\vecx}|\vecx)=\mathcal{N}(\tilde{\vecx}|\vecx, \sigma^2 \vecI)$，for a given noise level $\sigma$, the denoising score matching objective [(5)](#eq5) becomes:

$$
l(\vectheta; \sigma)=\frac{1}{2} \mathbb{E}_{p_{data}\,(\vecx)}\mathbb{E}_{\tilde{\vecx} \sim \mathcal{N}(\vecx, \sigma^2 \vecI)}\Big[\left\lVert \vecs_{\vectheta}(\tilde{\vecx},\sigma)+\frac{\tilde{\vecx}-\vecx}{\sigma^2} \right\rVert_2^2 \Big],
$$

and if multiple noise level considered, we have a unified objective:

$$
\mathcal{L}(\vectheta; \{\sigma_{i}\}_{i=1}^{L})=\frac{1}{L}\sum_{i=1}^L \lambda(\sigma_i)l(\vectheta; \sigma_i),
$$

where $\lambda(\sigma_i)>0$. Given the sufficient capacity of the score neural network $\vecs_{\vectheta}$, i.e., for any function $$\{f_i: \mathbb{R}^d \rightarrow \mathbb{R}^d\}_{i=1}^L$$, there exists $\vectheta$ with $\vecs_{\vectheta}(\vecx, \sigma_i)=f_i(\vecx)$ for all $i$, then we have $\vectheta^*$ is a global minimizer of above unified objective **if and only if** $$\vecs_{\vectheta^{*}}(\vecx, \sigma_i)=\nabla_{\vecx} \log q_{\sigma_i}(\vecx)$$ almost surely for all $$i \in \{1,2,...,L\}$$. The "if" part is trivial, for the "only if" part, we can prove by the sufficient capacity assumption, and ensure the existence of $\tilde{\vectheta}$ such that $l(\tilde{\vectheta};\sigma_i)=0$ for all $i$, and then $$0 \leq \mathcal{L}(\vectheta^{*};\{\sigma_i\}_{i=1}^L) \leq \mathcal{L}(\tilde{\vectheta};\{\sigma_i\}_{i=1}^L) = 0$$, thus $$\mathcal{L}(\vectheta^{*};\{\sigma_i\}_{i=1}^L)=0=\frac{1}{L}\sum_{i=1}^L \lambda(\sigma_i)l(\vectheta^{*}; \sigma_i)$$, and finally all the $$l(\vectheta^{*};\sigma_i)=0$$, because of the positivity of the $\lambda(\sigma_i)$ and the nonnegativity of the $$l(\vectheta^{*};\sigma_i)$$ for all $i$. One possible choice of $\lambda$'s, suggested in {% include cite.html key="song2019generative"%}, is motivated by values of $\lambda(\sigma_i)l(\vectheta;\sigma_i)$ being roughly the same order of magnitude, which eventually gives us $\lambda(\sigma)=\sigma^2$.

## Annealed Langevin Dynamics Inference
Given the sigma noise level $\sigma_i$'s, a positve $\epsilon$, and sampling steps $T$, the annealed Langevin dynamics inference is the following,
1. Initialize $\tilde{\mathbf{x}}_0$  $\quad\quad\triangleright$ from some prior distribution, e.g., uniform distribution
2. **for** $i \leftarrow 1$ to $L$ **do**
3. $\quad \alpha_i \leftarrow \epsilon \cdot \sigma_i^2/\sigma_L^2$ $\quad\quad\triangleright$ $\alpha_i$ is the step size.
4. $\quad$ **for** $t \leftarrow 1$ to $T$ **do**
5. $\quad\quad$ Draw $\mathbf{z}_t \sim \mathcal{N}(0, I)$
6. $\quad\quad$ $$\tilde{\mathbf{x}}_t \leftarrow \tilde{\mathbf{x}}_{t-1} + \frac{\alpha_i}{2}\mathbf{s}_\theta(\tilde{\mathbf{x}}_{t-1}, \sigma_i) + \sqrt{\alpha_i}\mathbf{z}_t$$
7. $\quad$ **end for**
8. $\quad \tilde{\mathbf{x}}_0 \leftarrow \tilde{\mathbf{x}}_T$
9. **end for**  
**return** $\tilde{\mathbf{x}}_T$

We can see from the above algorithm, we sample from the sequence of noise-perturbed distributions $q_{\sigma_1}(\vecx), q_{\sigma_2}(\vecx), ..., q_{\sigma_L}(\vecx)$ that converge to true $p_{data}(\vecx)$, with each distribution $q_{\sigma_i}(\vecx)$ being sampled for only $T$ steps and using gradually reduced step size $\alpha_i$. For the final $\sigma_L \approx 0$, we have $q_{\sigma_L}(\vecx) \approx p_{data}(\vecx)$. The choice of selecting $\alpha_i$'s, suggested in {% include cite.html key="song2019generative"%}, is $\alpha_i \propto \sigma_i^2$, motivated by fixing the $l_2$-norm of the "signal-to-noise" ratio $\frac{\alpha_i \vecs_{\vectheta}(\vecx, \sigma_i)}{2\sqrt{\alpha_i} \vecz}$ in Langevin dynamics (see the paper for details). 

The intution of above inference procedure is that since $\sigma_1$ is large enough to mitigate the effects of $\text{manifold hypothesis}$ and $\textbf{low data density regions}$, score estimation $\vecs_{\vectheta}(\vecx, \sigma_1)$ will be an accurate estimation of $q_{\sigma_1}(\vecx)$, and thus sampling loop for noise level $\sigma_1$ will be faster. Because samples produced using $\sigma_1$ are good samples for $q_{\sigma_1}(\vecx)$, are more likely to be around high density regions for $q_{\sigma_2}(\vecx)$, since $q_{\sigma_1}(\vecx)$ and $q_{\sigma_2}(\vecx)$ are quite similar to each other, these samples will be good initial samples for second noise level $\sigma_2$ loop, and again generate good samples for $q_{\sigma_2}(\vecx)$, and eventually we have good samples from $q_{\sigma_L}(\vecx)$, which is indistinguishable from $p_{data}(\vecx)$




## Score-based Generative Modeling through SDEs

---
{% include bibliography.html keys="chen2014stochastic,chewi2025logconcave,hyvarinen2005estimation,neal2011mcmc,roberts1996exponential,roberts1998optimal,song2019generative,song2020sliced,vincent2011connection,welling2011bayesian" %}
