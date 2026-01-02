---
layout: post
title: "Rendering Basics"
date: 2025-01-02
---
## Deriving T(t)
We model ray termination as an inhomogeneous Poisson process along $t$. Let the ray be at position $r(t)$, then we have 

$$
\text{Pr[termination in }(t,t+dt)∣\text{alive at }t]=\sigma(r(t))dt+o(dt)
$$

Hence survival probability: 

$$
\text{Pr[survive }(t,t+dt)∣\text{alive at }t]=1−\sigma(r(t))dt+o(dt)
$$

Define transmittance:

$$
T(t):=\text{Pr[ray survives from }t_n​ to t]
$$

with initial condition: $T(t_n​)=1$. Consider survival from $t$ to $t+dt$: 

$$
T(t+dt)=T(t)[1−\sigma(r(t))dt]+o(dt)
$$

Rearrange to get:

$$
\frac{T(t+dt)−T(t)​}{dt} = −\sigma(r(t))T(t)+o(1)
$$

Taking $dt \rightarrow 0$, we obtain the ODE:

$$
\frac{dT}{dt}(t)=-\sigma(r(t))T(t)
$$

with $T(t_n)=1$. Solving the ODE:
<div id="eq1">
$$
\begin{align*}
\frac{dT}{T}&=-\sigma(r(t))dt \\
\log T(t)&=-\int_{t_n}^{t}\sigma(r(s))ds \\
\text{exponentiate: }T(t)&=exp\Big(-\int_{t_n}^{t}\sigma(r(s))ds \Big)
\end{align*}
$$
</div>


## Deriving expected color
Define random variable: $\tau=$termination time of the ray, we now compute its probability density. The probability that the ray: survives up to $t$, and terminates in $(t,t+dt)$ is:

$$
\text{Pr[}\tau \in (t,t+dt)]=T(t)\sigma(r(t))dt
$$

Thus the probability density function of $\tau$ is: $p_{\tau}(t)=T(t)\sigma(r(t))$. The pixel color is the expected emitted radiance at termination. Define random variable: $C:=c(r(\tau),d)$, then:

$$
\begin{align*}
E[C]&=\int_{t_n}^{t_f} ​​c(r(t),d) p_{\tau}​(t)dt \\
&=\int_{t_n}^{t_f} T(t)\sigma(r(t))c(r(t),d)dt
\end{align*}
$$

This is exactly the NeRF equation.

## Deriving discretized rendering equation
Assume within each bin, for $$t \in [t_i, t_{i+1}]$$:
<div id="eq2">
$$
\begin{align*}
\sigma(t)​\approx \sigma_i  \\
c(t) \approx c_i​
\end{align*}
$$
</div>
Within bin $i$, transmittance can be written as:

$$
T(t)=T_i ​exp\Big(−\int_{t_i}^t \sigma_i​ ds\Big)=T_i ​e^{-\sigma_i​(t−t_i​)}
$$

where:

$$
T_i​=exp \Big(\sum_{j=1}^{i-1}​\sigma_j\delta_j\Big​)
$$

Then for each bin contribution:

$$
\begin{align*}
\int_{t_i}^{​t_{i+1}} ​​T(t)\sigma_i​ c_i​ dt​=T_i ​c_i \int_0^{\delta_i} ​​\sigma_i ​e^{−\sigma_i​ s}ds \\
T_i ​c_i​(1−e^{−\sigma_i​ \delta_i}​)
\end{align*}​
$$

Summing all bins, we have

$$
\hat{C}(r) = \sum_{i=1}^N T_i (1-e^{-\sigma_i \delta_i})c_i
$$


============================================================================================
============================================================================================
============================================================================================
============================================================================================







{% capture def_content %}
Let $U \subseteq  \mathbb{R}^n$ be an open subset, then a function $f: U \rightarrow \mathbb{R}^m$ is said to be $\textbf{(totally) differentiable}$ at a point $a \in U$ if there exists a linear transformation $df_a: \mathbb{R}^n \rightarrow \mathbb{R}^m$ such that
$$
\lim_{x\rightarrow a}\frac{||f(x)-f(a)-df_a(x-a)||}{||x-a||}=0
$$
The linear map $df_a$ is called the $\textbf{(total) derivative}$ or $\textbf{(total) differential}$ of $f$ at $a$. Other notations for the total derivative include $D_af$ and $Df(a)$. A function is $\textbf{(totally) differentiable}$ if its total derivative exists at every point in its domain. 
{% endcapture %}

{% include theorem.html 
   type="definition" 
   title="Definition" 
   name="Total derivative"
   content=def_content 
%}

{% capture th_content %}
If all the partial derivatives(check partial derivatives definition) of $f$ at $a$ exist and are continuous in a neighborhood of $a$, then $f$ is differentiable at $a$, when this happens, then in addition, the total derivative of $f$ is the linear transformation corresponding to the $\textbf{Jacobian matrix}$ of partial derivatives at that point
{% endcapture %}

{% include theorem.html 
   type="theorem" 
   title="Theorem" 
   name="Total derivative-Jacobian"
   content=th_content 
%}


## Neural ODEs

If we use the neural network parameterized velocity field $\vecv_{\vecphi}(\vecz(t), t)$ to replace above $f$, and we take the integral, then we get:

$$
\log p_{\vecphi}​(\vecx, T)=\log p_{prior}​(\vecz(0))−\int_0^T \nabla_{\vecz}​\cdot \vecv_{\vecphi}​(\vecz(t),t)dt
$$



<!--
===========================================================================
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

<div id="eq1">
$$
\begin{align*}
\int q_{\vecphi}(\vecz|\vecx) \log\frac{p_{\vectheta}(\vecx, \vecz)}{q_{\vecphi}(\vecz|\vecx)} =: \text{ELBO}   \tag{1}
\end{align*}
$$
</div>

Equation [(1)](#eq1)
-->

---
{% include bibliography.html keys="rezende2015variational,chen2018neural,pontrjagin1962mathematical" %}
