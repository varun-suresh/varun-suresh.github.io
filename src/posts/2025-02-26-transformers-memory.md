---
title: Transformers FLOPS and memory usage
subtitle: Understanding memory usage and the number of floating point operations required to run inference using Transformers
date: 2025-02-26
blurb: An in-depth breakdown with an example of how much GPU memory is required to run inference using a Transformer
---
## Introduction

Everyday we hear of new AI models that are larger and better than ever before. As of writing this post, the state-of-the-art models have hundreds of billions of parameters[^1]. How much GPU memory is needed to run this model? How long will inference take on a given GPU? In this post, I will walk through an example to understand how to calculate memory usage and estimate the time taken to run inference. This post assumes familiarity with the transformer[^ https://arxiv.org/pdf/1706.03762] architecture.

[^1]: [DeepSeek R1 has about 671 billion parameters](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/README.md)

In the first part of this post, I count all the floating point operations that happen during the forward pass. Why is this important? Every GPU has a maximum number of floating point operations it can perform in a second (FLOPS) and is specified in the GPU specs. If we can calculate the number of floating point operations that our model needs to perform for a single token, we can calculate the time taken for the forward pass.[^ Assuming that we have a kv cache, I'm calculating the FLOP for a single new token. More details about this later in the post.]

 In the second part of this post, I calculate the GPU memory usage given a model. This includes the memory needed to store the model parameters and kv cache.

## Floating point operations
Given a transformer model and a GPU, can we approximately calculate the inference time without running any code? We can calculate it using this simple formula
$$
24.n_{layers}.d_{model}^2 + 2.d_{vocab}.d_{model}^2
$$

In the next few sub-sections, I go through in detail as to how I arrived at the above equation. Feel free to skip to the memory section if you aren't particularly interested in the details.

A GPU datasheet specifies the theoretical maximum number of floating point operations it can perform in a second. Using the transformer model architecture, let's calculate the number of floating point operations (FLOP) per token

### FLOP for matrix multiplication

Let's consider an example for matrix multiplication [^4]

[^4]: [Lecture Notes on FLOPS for basic operations](https://www.stat.cmu.edu/~ryantibs/convexopt-F18/scribes/Lecture_19.pdf)
$$
A = \begin{bmatrix} 
  a_{00} & a_{01} & a_{02} \\ 
  a_{10} & a_{11} & a_{12} 
\end{bmatrix}
\quad
B = \begin{bmatrix}
 b_{00} & b_{01} & b_{02} & b_{03} \\
 b_{10} & b_{11} & b_{12} & b_{13} \\
 b_{20} & b_{21} & b_{22} & b_{23}
 \end{bmatrix} 
$$

A is a $2x3$ matrix and B is a $3x4$ matrix. The product of A and B will have $2x4$ elements. In the generalized case, when a matrix of dimensions $ mxn $ is multiplied with a matrix of dimensions $nxp$, the product is a $mxp$ dimensional matrix. 

Calculating the first element of the $AxB$, 
 $$
 a_{00}.b_{00} + a_{01}.b_{10} + a_{02}.b_{20}
 $$
 Notice that there are 3 multiplication operations and 2 addition operations. In the general case, there are $n$ additions and $n-1$ multiplications to calculate one element in the product matrix.
 
 $$
n + (n-1) \approx 2n
 $$
 Since there are 8 elements in $AxB$ and $mxp$ elements in the general case, the total number of floating point operations to calculate the product of two matrices is
 $$
  \approx (2n)*m*p = 2mnp \tag{1}
 $$

### FLOP in a transformer block

A transformer block consists of two sub-blocks, a multi-headed attention block and a feed forward block which are shown in the picture below.


Let's calculate the number of floating point operations in each of the steps in the transformer block. 
$$
\begin{array} {l l}
t_{e}: &\text{Token embedding} \quad (\mathbb{R}^{1xd_{model}})\\
d_{model}: &\text{Model embedding dimension} \\
d_{k}: &\text{Key,Query and Value matrix dimension per head in the multi-headed attention block} \\
W_{Q},W_{K},W_{V}: &\text{ Query, Key and Value matrices} \quad (\mathbb{R}^{d_{model}xd_{k}})\\
n_{heads} : &\text{Number of heads in the multi-headed attention block} \\
n_{layers}: &\text{Number of transformer blocks in the model} \\
d_{vocab}: &\text{Vocabulary size for tokenization}
\end{array}
$$
Note that typically
$$
d_{k} = \frac{d_{model}}{n_{heads}}
$$
Note: Insert a transformer block picture

For a single token, the first step is calculating the keys, queries and values for all the attention head. 
$$
q = W_{Q}t_{e}^{T} 
$$

The number of floating point operations to calculate q using (1) is 
$$ 
2d_{model}d_{k} = \frac{2d_{model}^2}{n_{head}} \tag{2}
$$

The same number of FLOP are required to calculate k and v. Using (2), the total FLOP count per head is
$$
3.\frac{2d_{model}^2}{n_{head}} = \frac{6d_{model}^2}{n_{head}}
$$

With all the heads combined, the total FLOP count is
$$
6d_{model}^2 \tag{3}
$$

Scaled dot product attention is calculated using
$$
\begin{aligned}
attention = softmax\left(\frac{qk^T}{\sqrt{d_k}}\right).v \\
qk^T = 2.d_{k} = \frac{2.d_{model}}{n_{heads}} \quad \text{FLOP}   \\
softmax\left( \frac{qk^T}{\sqrt{d_k}}\right) = 2.d_{k} = \frac{2.{d_{model}}}{n_{heads}} \quad \text{FLOP}  \\
softmax\left( \frac{qk^T}{\sqrt{d_k}}\right).v = 2.d_{model} \quad \text{FLOP} \\
\text{Total FLOP in a single attention head} = \frac{4.d_{model}}{n_{heads}} + 2.d_{model} \tag{4} \\
\text{Total FLOP with all the attention heads} = 4.d_{model} + 2.d_{model}.n_{heads}
\end{aligned}
$$

Concatenation does not involve any matrix multiplications. The output of the concatenation is a $\mathbb{R}^{1xd_{model}}$ vector. The linear layer is a $\mathbb{R}^{d_{model}xd_{model}}$ matrix. Based on (1), the number of FLOP in the linear layer is [^ Assuming the linear layers have zero bias. Even if they don't, they will not contribute significantly to the FLOP count]
$$
2.d_{model}^2 \tag{5}
$$

The feed-forward layer consists of two linear layers. The first layer is a $ \mathbb{R}^{d_{model}x4d_{model}}$ matrix and the second layer is a $\mathbb{R}^{4d_{model}*d_{model}}$ matrix. Therefore the number of FLOP in these layers are
$$
2.4.d_{model}^2 + 2.4.d_{model}^2 = 16.d_{model}^2 \tag{6}
$$

Adding (3),(4),(5) and (6), the total FLOP in a transformer block is 
$$
6.d_{model}^2 + 4.d_{model} + 2.d_{model}.n_{heads} + 2.d_{model}^2 + 16.d_{model}^2  \approx 24.d_{model}^2 \tag{7}
$$

When $d_{model}$ is sufficiently large, $d_{model}^2$ term dominates, so we can ignore all the terms with the $d_{model}$ coefficient. Note that I have ignored the layernorm calculation and the residual calculation. These calculations would also be a constant factor times $d_{model}$ and will be sufficiently small.

### FLOP for other layers

The transformer model consists of token embedding, position embedding, $n_{layers}$ of transformer blocks and a final linear layer which has $\mathbb{R}^{d_{model}xd_{vocab}}$ parameters. Token and position embeddings are lookups, so there is no matrix multiplication. The number of FLOP in the final linear layer is 
$$
2.d_{model}^2.d_{vocab} \tag{8}
$$

### FLOP for the full model

The transformer model has $n_{layers}$ number of transformer blocks. Putting together (7) and (8), the total FLOP is
$$
24.d_{model}^2.n_{layers} + 2.d_{model}^2.d_{vocab} \tag{9}
$$

To get an idea of the scale of FLOP required to do the forward pass in a transformer using (9), I calculated the FLOP count for a few models.
$$
\begin{array}{c|c|c|c|c}
    \text{Model Name} & \text{Layers} & \text{Embedding dimension} & \text{Vocabulary Size} &\text{FLOP}\\ \hline
    GPT2 XL (1.55B) & 48 & 1600 & 50257 & 0.260T\\
    Llama 3.1(8B) & 32 & 4096 & 128000 & 4.307T&\\
    Llama 3.1(405B) & 126 & 16384 & 128000 & 69.53T& 
\end{array}
$$

My Nvidia 3060 GPU can theoretically do 101T[^nvidia3060] FLOPS using FP16 (half precision). If I run inference on the Llama 8B model, I can process about $\frac{101}{4.307} \approx 23$ tokens per second.[^ Note that I have not considered memory to store these parameters]

[^nvidia3060]: [Tom's Hardware page](https://www.tomshardware.com/news/nvidia-confirms-rtx-3060-specs)

## Memory requirements

In a transformer model, two major components that take up the GPU memory are the model parameters and KV cache. In the following sub-sections, I will show how to calculate the memory usage based on the model size and the size of the KV cache.

### Model parameters
Typically we store all the parameters in half precision(FP16), each parameter requires 2 bytes. If a model has $n$ parameters, the memory required to store them is simply $ 2.n $ bytes

My Nvidia 3060 GPU has 12GB memory. I cannot fit a Llama 3.1 8B model because storing the parameters alone would require 16GB of memory. But I can fit a smaller model like GPT2 XL with ~1.5B parameters which would take about 3GB of memory. 

### KV Cache
Let's first understand why KV cache is necessary. In an autoregressive model like GPT-2, the next token depends on the previous tokens. Attention is calculated using the formula

$$
Attention(Q,K,V) = softmax \left( \frac{QK^T}{\sqrt{d_k}}\right).V
$$

$Q$,$K$ and $V$ are matrices of dimensions $d_{model}xn_s$, $n_s$ is the length of the sequence. Notice that the $QK^T$ computation increases quadratically with the length of the sequence. [^kvcache] The key and value calculations are repeated for all the previous tokens which is wasteful. Caching keys and values will make the $QK^T$ computation linear with the length of the sequence $n_s$.

Let's calculate how much memory is required to cache a single token.
$$
\begin{array}{c|c}
\text{To store keys for a single attention head} & 2.d_k \\
\text{Storing keys and values} &  4.d_k \\
\text{For all heads} & 4.d_k.n_{heads} = 4.d_{model} \\
\text{For all layers in the transformer} & 4.n_{layers}.d_{model} \tag{10}
\end{array}
$$

Using (10), the memory needed to store 1000 tokens for a few models are shown below
$$
\begin{array}{c|c|c|c}
    \text{Model Name} & \text{Layers} & \text{Embedding dimension} & \text{Memory per 1000 tokens} \\ \hline
    GPT2 XL (1.55B) & 48 & 1600 &  0.307GB\\
    Llama 3.1(8B) & 32 & 4096  & 0.524GB\\
    Llama 3.1(405B) & 126 & 16384 & 8.25GB
\end{array}
$$

On my Nvidia 3060 GPU, using GPT2, I can cache about $\frac{9GB}{0.307GB} \approx 30k$ tokens. The memory used for kv cache increases linearly with both $n_{layers}$ and embedding size.

[^kvcache]: [KV cache explained](http://medium.com/@joaolages/kv-caching-explained-276520203249)
## References
1. [Attention is all you need](https://arxiv.org/pdf/1706.03762)
2. [Kipply's blog on transformer inference arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
3. [Andrej Karpathy's nanoGPT implementation](https://github.com/karpathy/nanoGPT)
4. [KV cache explained with images/GIFs](http://medium.com/@joaolages/kv-caching-explained-276520203249)