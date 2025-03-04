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

Transformer[^2] is a neural network architecture that was originally introduced for language translation but is now used for vision and multimodal tasks with a lot of success. Throughout this post, I will use an encoder-only architecure BERT[^2] as an example and walk through each layer in the model.

Let's consider a simple sentence `Hello World`


[^2]: [Attention is all you need](https://arxiv.org/pdf/1706.03762)
[^3] : [Bidirectional Encoder Transformer](https://arxiv.org/pdf/1810.04805)