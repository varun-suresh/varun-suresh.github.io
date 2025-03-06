---
title: Transformers Memory Usage
subtitle: Understanding memory usage when running inference using Transformers
date: 2025-02-26
blurb: An in-depth breakdown with an example of how much GPU memory is required to run inference using a Transformer
---

## Introduction

Everyday we hear of new AI models that are larger and better than ever before. As of writing this post, the state-of-the-art models have hundreds of billions of parameters[^1]. How much GPU memory will I need to run this model? How long will inference take on a given GPU? In this post, I will walk through an example to understand how to calculate memory usage when running inference. 

[^1]: [DeepSeek R1 has about 671 billion parameters](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/README.md)

## Transformer walk through

Transformer[^2] is a neural network architecture that was originally introduced for language translation but is now used for vision and multimodal tasks with a lot of success. Throughout this post, I will use a decoder-only architecure GPT-2[^ https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf] as an example and walk through each layer in the model.

[^2]: [Attention is all you need](https://arxiv.org/pdf/1706.03762)

First, let's consider the memory needed to store all the parameters of the model. For this example, let's use GPT-2 XL that has 1.542B parameters. Assuming half precision, each of the parameters is 2 bytes in size.

$$
    1.542 * 10^9 * 2 \approx 3 GB
$$

I have a Nvidia 3060 GPU with 12GB memory. Loading the model onto the GPU will leave me with 
$$
12GB - 3GB \approx 9GB
$$

## Floating point operations
Given a transformer model and a GPU, can we approximately calculate the inference time without running any code?

A GPU datasheet specifies the theoretical maximum number of floating point operations it can perform in a second. Using the transformer model architecture, let's calculate the number of floating point operations (FLOP) per token

### FLOP for matrix multiplication

Let's consider an example for matrix multiplication [^3]

[^3]: [Lecture Notes on FLOPS for basic operations](https://www.stat.cmu.edu/~ryantibs/convexopt-F18/scribes/Lecture_19.pdf)
$$
A = \begin{bmatrix} 
  1 & 2 & 3 \\ 
  4 & 5 & 6 
\end{bmatrix}
\quad
B = \begin{bmatrix}
 1 & 2 & 3 & 4 \\
 5 & 6 & 7 & 8 \\
 9 & 10 & 11 & 12
 \end{bmatrix} 
$$

A is a *2x3* matrix and B is a *3x4* matrix. The product of A and B will have *2x4* elements. In the generalized case, when a matrix of dimensions *mxn* is multiplied with a matrix of dimensions *nxp*, the product is a *mxp* dimensional matrix. 

Calculating the first element of the *AxB*, 
 $$
 1.1 + 2.5 + 3.9 = 38
 $$
 Notice that there are 3 multiplication operations and 2 addition operations. In the general case, there are *n* additions and *n-1* multiplications to calculate one element in the product matrix.
 
 $$
n + (n-1) \approx 2n
 $$
 Since there are 8 elements in *AxB* and *mxp* elements in the general case, the total number of floating point operations to calculate the product of two matrices is
 $$
  \approx (2n)*m*p = 2mnp
 $$

### FLOP in a transformer block

A transformer block consists of two sub-blocks, a multi-headed attention block and a feed forward block which are shown in the picture below.


Let's calculate the number of floating point operations in each of the steps in the transformer block. 
$$
\begin{array} {l l}
t_{e}: &\text{Token embedding} \\
d_{model}: &\text{Model dimension} \\
d_{k}: &\text{Key,Query and Value matrix dimension} \\
W_{Q},W_{K},W_{V}: &\text{ Query, Key and Value matrices} \\
n_{heads} : &\text{Number of heads in the multi-headed attention block}
\end{array}
$$
Note that typically
$$
d_{k} = d_{model} / n_{heads}
$$
Note: Insert a transformer block picture

For a single token, the first step is calculating the keys, queries and values for all the attention head. 
$$
q = W_{Q}t_{e}^{T}
$$
where 
$$
\begin{array}{l l}
t_{e}: \mathbb{R}^{1xd_{model}} \\ 
W_{Q}: \mathbb{R}^{d_{model}xd_{k}}
\end{array}
$$
 
The number of floating point operations to calculate q is 
$$ 
2d_{model}d_{k} = \frac{2d_{model}^2}{n_{head}}
$$

The same number of FLOP are required to calculate k and v. Therefore the total FLOP count is
$$
3.\frac{2d_{model}^2}{n_{head}} = \frac{6d_{model}^2}{n_{head}}
$$

