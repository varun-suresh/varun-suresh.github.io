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

I have a Nvidia 3060 GPU with 12GB memory[^ GPU memory]. Loading the model onto the GPU will leave me with 
$$
12GB - 3GB \approx 9GB
$$

## Floating point operations
Given a transformer model and a GPU, can we approximately calculate the inference time without running any code?

A GPU datasheet specifies the theoretical maximum number of floating point operations it can perform in a second. Using the transformer model architecture, let's calculate the number of floating point operations (FLOP) per token

## FLOP for matrix multiplication

Let's consider an example for matrix multiplication


\[
A = \begin{bmatrix} 
  a_{11} & a_{12} & a_{13} \\ 
  a_{21} & a_{22} & a_{23} \\ 
  a_{31} & a_{32} & a_{33} 
\end{bmatrix}
\]
