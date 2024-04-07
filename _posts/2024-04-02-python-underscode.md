---
layout: post
title: Uses of Underscore in Python
subtitle: Underscore examples in Python
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
gh-repo: arpithub/arpithub.github.io
gh-badge: [star, fork, follow]
tags: [python,python_tips]
comments: true
---

We all have seen underscore symbol in Python. One of the most common uses of the underscore is as a placeholder variable. When iterating over a sequence or generating a range of values, there are often situations where the loop variable is not needed. In such cases, Python programmers opt to use _ as a concise way to indicate that the value is irrelevant to the current context.                                      

```python
for _ in range(10):
    print("Doh!")
```

Here, _ acts as a placeholder for the loop variable, highlighting the intent to iterate a specific number of times without the need to reference the loop index.

Moreover, _ serves as a convention for denoting unused variables. In scenarios where only one value from a function call or tuple unpacking is required, Python programmers often assign the unused value to _, signaling to others that the value is intentionally disregarded.

```python
first_name, _ = get_name()
```
This will return multiple parameters but we ignore other parameters except `first_name`.

```python
>>> 1 + 2 + 3
6
>>> _ * 5
30
```
Here underscore saves the value of last expression ie. `1+2+3=5`

In short, Underscore `_` can be used as a placeholder, unused variable indicator or temporary result storage. It makes our code readable and expressive. Happy Code Writing!