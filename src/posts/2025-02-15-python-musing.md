---
title: Python Musings
subtitle: Exploring and understanding the internal workings of python
date: 2025-02-15
blurb: What happens under the hood when we overwrite a variable but that is not an in-place operation?
---

# Introduction

I have been writing Python for many years now, it is by far the most intuitive programming language I have used. However, every once in a while something seemingly obvious does not work as I'd expect.

In this post, I want to explore some of these gotchas and hopefully understand implementations in python better than I did before.

## Pass by object reference

Consider the following functions

```
from typing import List
def modify(x:int):
    x+= 1

def modify_list(x:List[int]):
    x.append(1)
```

The function `modify` takes in an integer x, modifies it _in-place_ by adding 1. When I execute the following lines,

```
x=5
modify(x)
print(x)

5
```

The value of x is unchanged although the operation we did was **in-place**.

However, when I run `modify_list`, the list is now updated.

```
x = [1,2,3]
modify_list(x)
print(x)

[1,2,3,1]
```

When a mutable object like a list or a dictionary is passed to a function, it is passed by reference. If the object is modified, the change is reflected outside the function as well. When an immutable object like an integer or tuple is passed to a function, it is equivalent to passing by value.

```
def modify_list(x:List[int]):
    x.append(5)
    x = [1,2]
x = [3,4]
modify_list(x)
print(x)

[3,4,5]
```

Although we are _re-assigning_ x, the `x` re-assigned in the function[^objectref] is in an entirely different memory location and its scope is only within the `modify_list` function.

[^objectref]: {-} To learn more about python's object reference, you can read this [post on geeksforgeeks](https://www.geeksforgeeks.org/is-python-call-by-reference-or-call-by-value/)

## Mutable default arguments

TL;DR - **Do not** use mutable objects as default arguments for a function.[^mutabledefault]

[^mutabledefault]: {-} A [post](https://florimond.dev/en/posts/2018/08/python-mutable-defaults-are-the-source-of-all-evil) with a simpler example and a story of using mutable objects as a default argument in a function.

Consider the following implementation

```
from typing import Dict
class TrieNode:
    def __init__(self, val:str,children:Dict[str,"Node"]={}) -> None:
        self.val = val
        self.children = children

node_1 = TrieNode("a")
node_2 = TrieNode("b")

node_1.children["c"] = TrieNode("c")
print(node_1.children.keys()) # Expect [c]
print(node_2.children.keys()) # Expect [] but get [c]
```

Although `node_1` and `node_2` are separate instantiations of the class `TrieNode`, they share the `children` attribute. The children attribute for both instances point to the same memory location because of the mutable dictionary object as a default argument in `__init__`.

The correct implementation would be as follows.

```
from typing import Dict, Optional
class TrieNode:
    def __init__(self, val:str,children:Optional[Dict[str,"Node"]]=None) -> None:
        self.val = val
        if children:
            self.children = children
        else:
            self.children = {}
```

In this implementation, the children attribute is initialized inside the class, so for every instantiation of this class, the `children` will be saved in a separate location.

## List implementation

In Python, a list is implemented as a dynamic array. A contiguous block of memory is initially assigned to a list. If the size of the list exceeds this size, a new block that is `k` times the original size is assigned where k > 1. Adding a new element to an array is a O(1) operation amortized. When an element added to the list causes the list size to increase, all the elements in the list need to be copied to this new block of memory making it a O(n) time complexity operation.

How are the elements of an list _actually_ stored? Note that not all elements in an array need to be of the same type and each[^leg]element could take up different amounts of memory as in the example below.

[^leg]: {-} For example, a list like `x = [1,2,3,"randomstring",5.3]` is a valid list.

```
import sys
x = [1,2,3]
print(sys.getsizeof(x))
x = [1,2,"Very long block of text"]
print(sys.getsizeof(x))

88
88

```

The result is the same in both cases, even though `Very long block of text` should take up more bytes than `3`. That is because only the references (address of the memory location + offset) to the elements are stored in the list. It is not that all the elements in the list are stored in contiguous locations (like in C, C++ arrays), but their references are stored sequentially.


## Numpy learnings

Numpy is a python library that speeds up vector and matrix operations. Under the hood, numpy runs C code to execute operations. C loops are significantly faster than Python loops and hence there is a significant speedup when numpy is used instead of python lists.

### Numpy ndarray
ndarray stands for n-dimensional array. There are two main components to a ndarray - data buffer and metadata. The data buffer is the contiguous memory block that stores the actual array data. The metadata contains information like the data type, shape of the array etc. Let's say you wanted to transpose a matrix. Numpy internally does not change the data buffer, instead it modifies the meta data to indicate how the data should be read. 

**Gotchas to be aware of**

Consider an example where you slice an array
```
import numpy as np

a = np.array([1,2,3,4])
b = a[0:2] # b -> [1,2]
c = a[[0:2]]
a[0:2] = [5,6]
print(b) # Expected: [1,2], Actual result: [5,6]
print(c) # [1,2]

```
This is because _b_ is merely a view of _a_, _b_ still points to the same data buffer as _a_. _c_ however creates an explicit copy.[^numpyref]

[^numpyref]: [Internal organization of NumPy arrays](https://numpy.org/doc/stable/dev/internals.html#numpy-internals)

### Broadcasting
Using numpy to multiply a scalar to a vector or a matrix is extremely straightforward. For example

```
import numpy as np
a = np.array([1,2,3])
b = 3
print(a*b) # np.array([3,6,9])
```

_a_ is an array of shape (1,3) and _b_ is a scalar. When the two are multiplied, _b_ is "streched" to be the same size as _a_ and numpy multiplies it point-wise. Consider another example where we want to multiply each row of a matrix by a different number

```
import numpy as np
a = np.array([[1,2],[3,4],[5,6]]) # Shape: 3x2
b = np.array([10,20,30]) # Shape: (3)
print(a*b) # Raises a dimensional mismatch error
print(a*b[:,np.newaxis]) # Now b is a (3,1) shaped array -> np.array([[10,20],[60,80],[150,180]])
```
Numpy can perform operations on two arrays when either their dimensions are equal or one of the two corresponding dimensions is 1. [^numpy-broadcasting]

[^numpy-broadcasting]: [Numpy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)