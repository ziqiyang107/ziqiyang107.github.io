---
layout: post
title: "Neural Ordinary Differential Equations Explained"
date: 2025-11-11
---
## Preliminaries

<div class="theorem-env theorem">
<div class="theorem-header">Theorem (Total derivative-Jacobian)</div>
<div class="theorem-content">
设 $f: \mathbb{R}^n \to \mathbb{R}^m$ 是可微函数，则其全导数可以表示为 Jacobian 矩阵：
$$
Df(\mathbf{x}) = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$
</div>
</div>



## Neural ODEs

The variational inference technique is used in many places in deep learning and statistics, e.g., famous $\textbf{Variational}$ $\textbf{autoencoder (VAE)}$ and as an extension of $\textbf{Expectation-Maximization}$ $\textbf{(EM) algorithm}$, it serves as an approximation of posterior distribution or is used in deriving the lower bound of the marginal log-likelihood of the observed data. We will give the basic setup for obtaining the lower bound of a marginal log-likelihood $\log p_{\vectheta}(\vecx)$:

<div id="eq1">
$$
\begin{align*}
\log p_{\vectheta}(\vecx) &= \log \int p_{\vectheta}(\vecx|\vecz) p(\vecz) d\vecz \\
&= \log \int \frac{q_{\vecphi}(\vecz|\vecx)}{q_{\vecphi}(\vecz|\vecx)} p_{\vectheta}(\vecx|\vecz) p(\vecz) d\vecz \\
&\geq \int q_{\vecphi}(\vecz|\vecx) \log\frac{p_{\vectheta}(\vecx, \vecz)}{q_{\vecphi}(\vecz|\vecx)} =: \text{ELBO}   \tag{1}
\end{align*}
$$
</div>

During inference, we first sample from prior $\vecz_K \sim p(\vecz_K)=\mathcal{N}(\veczero, \vecI)$, then sample from the trained likelihood/generative model $\vecx \sim p_{\vectheta}(\vecx\|\vecz_K)$. An annealed version of $-$ELBO multiplies a $\beta_t=\min(1, 0.01+t/10000) \in [0,1]$ term in front of the $\log p_{\vectheta}(\vecx\|\vecz_K)$ term, and this modification is said to perform better in {% include cite.html key="rezende2015variational"%}.

<!--
{: .theorem-env .lemma}
> **Lemma (Schwarz's Theorem)**
> 
> 如果二阶偏导数连续，则混合偏导数与求导顺序无关：
> $$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$$
-->


---
{% include bibliography.html keys="rezende2015variational,silver2016mastering," %}
