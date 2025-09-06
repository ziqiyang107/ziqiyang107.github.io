---
layout: post
title: "数学公式测试"
date: 2025-01-15
---


# 线性代数示例

## 换行align示例1

$$
\begin{align}
x &= 1+1 \\
&= 2
\end{align}
$$

## 换行align示例2

$$
\begin{align*}
x &= 1+1 \\
&= 2
\end{align*}
$$

## 向量运算

设向量 $\vecc{a} = (a_1, a_2, a_3)^T$ 和 $\vecc{b} = (b_1, b_2, b_3)^T$，则：

$$
\vecc{a} \cdot \vecc{b} = \sum_{i=1}^{3} a_i b_i
$$

向量的范数：$\norm{\vecc{a}} = \sqrt{\vecc{a} \cdot \vecc{a}}$

## 矩阵运算

矩阵 $\mat{A} \in \R^{n \times m}$，向量 $\vecc{x} \in \R^m$：

$$
\mat{A}\vecc{x} = \vecc{b}
$$

## 微积分

函数 $f(x, y)$ 的偏导数：

$$
\pdv{f}{x}, \quad \pdv{f}{y}
$$

微分形式：

$$
\dd{f} = \pdv{f}{x}\dd{x} + \pdv{f}{y}\dd{y}

函数 $f: \R^2 \to \R$，其梯度为：

$$
\grad f = \left(\pdv{f}{x}, \pdv{f}{y}\right)^T
$$

积分：
$$
\Int{a}{b} f(x) \dd{x}
$$

## 向量和矩阵

设向量 $\vecc{a}, \vecc{b} \in \R^3$，矩阵 $\mat{A} \in \R^{3 \times 3}$：

$$
\mat{A}\vecc{a} = \vecc{b}
$$

向量范数：$\norm{\vecc{a}} = \sqrt{\inner{\vecc{a}}{\vecc{a}}}$


## 概率论

随机变量 $X$ 的期望和方差：

$$
\E{X} = \Sum{i=1}{n} x_i \Pr{X = x_i}
$$

$$
\Var{X} = \E{X^2} - \left(\E{X}\right)^2
$$

## 量子力学符号

量子态 $\ket{\psi}$ 和观测算符 $\hat{A}$：
$$
