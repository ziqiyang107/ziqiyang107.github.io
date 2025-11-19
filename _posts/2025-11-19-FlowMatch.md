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
p_t = [\phi_t]_{*}p_0  \tag{1}
\end{align*}
$$
</div>

where the push-forward(a.k.a change of variables) operator $*$ is defined by:
<div id="eq2">
$$
\begin{align*}
[\phi_t]_{*}p_0(x) := p_0(\phi_t^{-1}(x))\text{det}\Big[\frac{\partial \phi_t^{-1}}{\partial x}(x) \Big]  \tag{2}
\end{align*}
$$
</div>
We say a VF $v_t$ generates a probability density path $p_t$ if its flow $\phi_t$ satisfies [(1)](#eq1). So the logic is:

<div id="eq3">
$$
\begin{align*}
v_t  \xrightarrow{\text{determines $\phi_t$ via ODE}} \phi_t \xrightarrow{\phi_t(x)\sim p_t} p_t   \tag{3}
\end{align*}
$$
</div>

For any pair $\tilde{p}_t$ and $\tilde{v}_t$, the $\textbf{Continuity theorem}$ will decice if $\tilde{v}_t$ can generate $\tilde{p}_t$. In flow matching framework, we want to see if we can flow from a simple noise distribution $p_0 \rightarrow q$, where $q$ is our unknown training data distribution, and we can let $p_1(x) \approx q(x)$, and if we have a VF $u_t(x)$ that generates $p_t$, then we can learn this $u(x)$ by a flow matching loss:
<div id='eq-star'>
$$
\begin{align*}
L_{FM}(\theta)=\mathbb{E}_{t\sim U[0,1], x\sim p_t(x)}||v_t(x, \theta)-u_t(x) ||^2   \tag{*}
\end{align*}
$$
</div>
If this loss reaches zero, then we can use learned $v_t(x, \theta)$ to generated $p_t(x)$, thus get $p_1(x)$ that is approximately $q(x)$. However, we have no idea of $u_t(x)$ and $p_t$, since it related to intrinsics of training data. The trick is to regress to a conditional vector field $u_t(x|x_1)$, where $x_1$ is a data sample from the training data.


## Conditional Flow Matching
Now consider a second objective:
<div>
$$
\begin{align*}
L_{CFM}(\theta)=\mathbb{E}_{t \sim U[0,1], x_1 \sim q(x_1), x \sim p_t(x|x_1)}||v_t(x)-u_t(x|x_1)||^2
\end{align*}
$$
</div>
where $p_t(x|x_1)$ is conditional probability path such that:
<div id='eq-starp'>
$$
\begin{align*}
p_0(x|x_1) &= p(x)=N(x|0, \vecI) \quad \text{ at time }t=0 \\
p_1(x|x_1) &= N(x|x_1, \sigma_{min}^2 \vecI) \quad \text{ at time $t=11$ concentrates around }x_1 \text{ for some small }\sigma_{min}  \tag{*'}
\end{align*}
$$
</div>
for a particular sample $x_1$ from the training data. Above expected loss is easy to estimate as long as we know how to sample from $p_t(x|x_1)$ and compute $u_t(x|x_1)$, a good thing is that above [(*')](#eq-starp) has the same gradients with [(*)](#eq-star), formally　{% include cite.html key="lipman2022flow"%}:

{% capture th_content %}
Assuming that $p_t(x) > 0$ for all $x \in \mathbb{R}^d$ and $t \in [0, 1]$, then, up to a constant independent of $\theta$, $L_{CFM}$ and $L_{FM}$ are equal. Hence, $\nabla_{\theta} L_{FM}(\theta) = \nabla_{\theta} L_{CFM}(\theta)$
{% endcapture %}

{% include theorem.html 
   type="theorem" 
   title="Theorem" 
   name="FM CFM equivalence"
   content=th_content 
%}




---
{% include bibliography.html keys="chen2014stochastic,chewi2025logconcave" %}




