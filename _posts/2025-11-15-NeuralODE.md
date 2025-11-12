---
layout: post
title: "Neural Ordinary Differential Equations Explained"
date: 2025-11-11
---
## Preliminaries
Before we get into the introduction to neural ODE, we need to review total derivative, multivariate chain rule, backpropagation.
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

Next we talk about multivariate chain rule(without proof), first give an example, and state its corollaries.
{% capture th_content %}
Consider differentiable functions $f: \mathbb{R}^m \rightarrow \mathbb{R}^k$ and $g: \mathbb{R}^n \rightarrow \mathbb{R}^m$, and a point $\veca$ in $\mathbb{R}^n$. Let $D_{\veca}g$ denote the total derivative of $g$ at $\veca$ and $D_{g(\veca)}f$ denote the total derivative of $f$ at $g(\veca)$. These two derivatives are linear transformations $\mathbb{R}^n \rightarrow \mathbb{R}^n$ and $\mathbb{R}^m \rightarrow \mathbb{R}^k$, respectively, so they can be composed. The chain rule for total derivatives is that their composite is the total derivative of $f \circ g$ at $\veca$:
$$
D_{\veca}(f \circ g) = D_{g(\veca)}f \circ D_{\veca}g
$$
Because the total derivative is a linear transformation, the functions appearing in the formula can be rewritten as matrices. The matrix corresponding to a total derivative is called a Jacobian matrix, and the composite of two derivatives corresponds to the product of their Jacobian matrices. From this perspective the chain rule therefore says:
$$
\vecJ_{f\circ g}(\veca) = \vecJ_f(g(\veca)) \vecJ_g(\veca)
$$
or
$$
d(f \circ g)_a=df_{g(a)}\cdot dg_a
$$
{% endcapture %}

{% include theorem.html 
   type="theorem" 
   title="Theorem" 
   name="Multivariate chain rule"
   content=th_content 
%}


{% capture coro_content %}
Another way of writing the chain rule is used when f and g are expressed in terms of their components as $\vecy=f(\vecu)=(f_1(\vecu),...,f_k(\vecu))$ and $\vecu = g(\vecx) = (g_1(\vecx),...,g_m(\vecx))$. In this case, the above rule for Jacobian matrices is usually written as:
$$
\frac{\partial (y_1,...,y_k)}{\partial(x_1,...,x_n)}=\frac{\partial (y_1,...,y_k)}{\partial(u_1,...,u_m)}\frac{\partial(u_1,...,u_m)}{\partial(x_1,...,x_n)}.
$$
In the special case where k = 1, so that f is a real-valued function, then this formula simplifies even further:

<div id="VJP">
$$
\begin{align*}
\frac{\partial y}{\partial(x_1,...,x_n)}=\frac{\partial y}{\partial(u_1,...,u_m)}\frac{\partial(u_1,...,u_m)}{\partial(x_1,...,x_n)} \tag{VJP}
\end{align*}
$$
</div>
Above is $\textbf{vector-Jacobian product (VJP)}$, since the right hand side is a vector times a Jacobian matrix. If we write in terms of each component:

<div id="Backprop">
$$
\begin{align*}
\frac{\partial y}{\partial x_i}=\sum_{l=1}^m \frac{\partial y}{\partial u_l}\frac{\partial u_l}{\partial x_i} \tag{Backprop}
\end{align*}
$$
</div>
If we view above scalar function $f$ as a loss function, then form [(VJP)](#VJP) and form [(Backprop)](#Backprop) become the $\textbf{Back-propagation algorithm}$
{% endcapture %}

{% include theorem.html 
   type="corollary" 
   title="Corollary" 
   name="Partial form - Backprop"
   content=coro_content 
%}

{% capture ex_content %}
Suppose that $f$ is a function of two variables, $x$ and $y$. If these two variables are independent, so that the domain of $f$ is $\mathbb{R}^2$, then the behavior of $f$ may be understood in terms of its partial derivatives in the $x$ and $y$ directions. However, in some situations, $x$ and $y$ may be dependent. For example, it might happen that $f$ is constrained to a curve $y=y(x)$. In this case, we are actually interested in the behavior of the composite function $f(x,y(x))$. The partial derivative of $f$ with respect to $x$ does not give the true rate of change of $f$ with respect to changing $x$ because changing $x$ necessarily changes $y$. However, the chain rule for the total derivative takes such dependencies into account. Write $\gamma (x)=(x,y(x))$, then the chain rule says
$$
 d(f\circ \gamma )_{x_{0}}=df_{(x_{0},y(x_{0}))}\cdot d\gamma _{x_{0}}
$$
By expressing the total derivative using Jacobian matrices, this becomes:
$$
{\frac {df(x,y(x))}{dx}}(x_{0})={\frac {\partial f}{\partial x}}(x_{0},y(x_{0}))\cdot {\frac {dx}{dx}}(x_{0})+{\frac {\partial f}{\partial y}}(x_{0},y(x_{0}))\cdot {\frac {dy}{dx}}(x_{0}).
$$
Suppressing the evaluation at $x_{0}$ for legibility, we may also write this as
$$
{\frac {df(x,y(x))}{dx}}={\frac {\partial f}{\partial x}}{\frac {dx}{dx}}+{\frac {\partial f}{\partial y}}{\frac {dy}{dx}}.
$$
This gives a straightforward formula for the derivative of $f(x,y(x))$ in terms of the partial derivatives of $f$ and the derivative of $y(x)$.
{% endcapture %}

{% include theorem.html 
   type="example" 
   title="Example" 
   name="Differentiation with direct dependencies"
   content=ex_content 
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

Equation [(1)](#eq1) During inference, we first sample from prior $\vecz_K \sim p(\vecz_K)=\mathcal{N}(\veczero, \vecI)$, then sample from the trained likelihood/generative model $\vecx \sim p_{\vectheta}(\vecx\|\vecz_K)$. An annealed version of $-$ELBO multiplies a $\beta_t=\min(1, 0.01+t/10000) \in [0,1]$ term in front of the $\log p_{\vectheta}(\vecx\|\vecz_K)$ term, and this modification is said to perform better in {% include cite.html key="rezende2015variational"%}.

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
-->

---
{% include bibliography.html keys="rezende2015variational,silver2016mastering," %}
