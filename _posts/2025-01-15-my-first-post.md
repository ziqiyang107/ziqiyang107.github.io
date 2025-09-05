---
layout: post
title: "数学公式测试"
date: 2025-01-15
---

<!-- 在文章开头定义数学命令 -->
$$
\newcommand{\vecc}[1]{\boldsymbol{#1}}
\newcommand{\mat}[1]{\mathbf{#1}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\pdv}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\dd}[1]{\,\mathrm{d}#1}
\newcommand{\norm}[1]{\left\|#1\right\|}
$$

# 线性代数示例

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
$$
