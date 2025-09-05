---
layout: post
title: "欢迎来到我的博客"
date: 2025-01-15
---

这是我的第一篇博客文章！

## 数学公式测试

行内公式：$E = mc^2$

显示公式：
$$
\begin{align}
\frac{d}{dx}\left( \int_{a}^{x} f(t) \, dt\right) &= f(x) \\
\sum_{n=1}^{\infty} \frac{1}{n^2} &= \frac{\pi^2}{6}
\end{align}
$$

## Python 代码测试

```python
def fibonacci(n):
    """计算斐波那契数列"""
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# 测试
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
