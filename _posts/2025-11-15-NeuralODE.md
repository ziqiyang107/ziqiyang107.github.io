---
layout: post
title: "Neural Ordinary Differential Equations Explained"
date: 2025-11-11
---
## Preliminaries

不带证明版：
{% capture def_content %}
函数 $f: \mathbb{R} \to \mathbb{R}$ 在点 $x_0$ 处连续，当且仅当：
$$
\lim_{x \to x_0} f(x) = f(x_0)
$$
{% endcapture %}

{% include theorem.html 
   type="definition" 
   title="Definition" 
   name="连续性"
   content=def_content 
%}



带证明版：

{% capture theorem_content %}
设 $f: \mathbb{R}^n \to \mathbb{R}^m$ 是可微函数，则其全导数可以表示为 Jacobian 矩阵：
$$
Df(\mathbf{x}) = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$
{% endcapture %}

{% capture theorem_proof %}
根据多元微积分的定义，函数 $f$ 在点 $\mathbf{x}$ 处可微意味着存在线性映射 $L: \mathbb{R}^n \to \mathbb{R}^m$ 使得：
$$
f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + L(\mathbf{h}) + o(\|\mathbf{h}\|)
$$

这个线性映射 $L$ 就是全导数 $Df(\mathbf{x})$。
{% endcapture %}

{% include theorem.html 
   type="theorem" 
   title="Theorem" 
   name="Total derivative-Jacobian"
   content=theorem_content 
   proof=theorem_proof 
%}

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


---
{% include bibliography.html keys="rezende2015variational,silver2016mastering," %}
