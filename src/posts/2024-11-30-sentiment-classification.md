---
title: Sentiment Classification on IMDb dataset
subtitle: Fine-tune GPT-2 and BERT using LoRA
date: 2024-11-30
blurb: Use LoRA to fine-tune GPT-2 and BERT for sentiment classification.
---

# Introduction

What is sentiment classification?

Given a review, we want to classify this review as either positive or negative. For example, "The movie was incredible" should be classified as a positive review and "The performances were terrible" should be classified as negative. On a simple sentence, this is easy to do. However, reviews tend to be long and a lot of the times quite nuanced.

Sentiment classification is a well studied problem in Natural Language Processing. Large Language Models (LLMs) in the past few years have achieved excellent results in this area. In this post, I explore sentiment classification using [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) base model (124M) and [BERT](https://arxiv.org/pdf/1810.04805) base model (124M).

# Goal

Fine-tune GPT-2 and BERT with and without [LoRA](https://arxiv.org/abs/2106.09685) to do sentiment classification and benchmark their performance.

# Overview

Below is a brief overview of the data, GPT-2 and BERT. For both models, I used pre-trained weights from [HuggingFace](https://huggingface.co/) as the starting point and fine-tuned them for sentiment classification.

## Large Movie Review Dataset

This [dataset](https://ai.stanford.edu/~amaas/data/sentiment/) contains 50k movie reviews (25k training and 25k test). There are an equal number of positive and negative reviews in this dataset

## GPT-2

GPT-2 is a model from OpenAI trained for text generation i.e the model predicts the most probable next word given the previous words. For example

```
Once upon a
```

the most probable next word is `time` and that is what GPT-2 would generate. The word after that will be conditioned on `Once upon a time` and so on.The number of "words"(tokens) the model attends to before generating the next word is called the context window. The model uses causal self-attention, this means the model can only attend to words _before_ but not words that come after it.

## BERT

BERT is a model from Google and it was trained using the Masked Language Modeling(MLM) and Next Sentence Prediction (NSP) objectives. Given a sentence like

```
Once [MASK] a time
```

the model is trained to predict the word at the `[MASK]` location. Unlike GPT-2, the model attends to words _before_ and _after_ the `[MASK]` token to infer the most probable word.

# Experiments

## Zero Shot Learning with GPT-2

Given a prompt like "Review: The movie was awesome! Sentiment:", I compare the likelihood of the next token being " Positive" and " Negative" and classify the review.

I tried a few prompts and the results varied quite a lot. An additional space ("Positive" and " Positive") results in different tokens and the model is quite sensitive to these prompts. Among the few prompts I tried, the one I used eventually had the best results.

## Fine Tuning using GPT-2

In this setting, I froze the Positional Embedding weights, Token Encoding weights and the first 10 transformer blocks. Instead of the language modeling head, I used a binary classification head (a fully-connected layer with just one output followed by a sigmoid to make the output between 0 and 1 where 0 is negative and 1 is positive). I used the binary cross entropy loss function.

Parameter count when fine-tuning:

Transformer Block Layer 11:

```
Query Weights = 768 * 768 + 768 (Embedding size = 768, Weights + Bias) = 590592
Key Weights = 768 * 768 + 768
Value Weights = 768 * 768 + 768 = 590592
Layer Norm (2) = 768 * 2 (gamma and beta) * 2(2 layer norms in a transformer block) = 3072
Feed Forward Weights 1 = 768 * (4*768) + 4*768 = 2362368
Feed Forward Weights 2 = 4*768 *768 + 768 = 2360064

Total = 7087872
```

Binary Classification Head

```
Weights = 768 * 1
```

When finetuning, about 14M parameters are being modified (14M out of 124M).

## Fine Tuning using GPT-2 and LoRA

When using LoRA, the pre-trained weights are unchanged. It introduces two matrices **A** and **B** whose product is added to the weight matrix. Let's consider a concrete example:

Let W<sub>k</sub> be a weights matrix. In this case, let's consider the keys weight in a transformer block. The weights have the dimension 768 \* 768 and the bias is a 768 dimensional tensor.

Instead of modifying this large matrix, we can write

$\ W_k = W_k + \Delta W $

$\ \Delta W = AB^T $

where **A** and **B** are two low rank matrices of dimension 768 x 8 . AB<sup>T</sup> will result in a matrix of dimension 768 x 768, but the number of learned parameters are

```
768 * 8 * 2(A and B) * 2(Keys and Weights matrix) = 24576
```

significantly lower than ~7M parameters per transformer layer when fine tuning all the parameters.

## Fine Tuning using BERT

Similar to GPT-2, I froze the embeddings and the first 10 layers of the transformer. I took the mean of the embeddings of all the tokens and added a binary classification head on top.

## Fine Tuning using BERT + LoRA

In this setting, I froze the first 10 layers of the transformer and only learn the LoRA parameters for the last two layers. The binary classification head is also learnt during the process.

# Results

To reproduce these results, clone my [repository](https://github.com/varun-suresh/experiments-with-gpt2/) and run the [sentiment classification](https://github.com/varun-suresh/experiments-with-gpt2/blob/main/sentiment_classification/sentiment_classification.ipynb) notebook.

| Model/Method             | accuracy | precision |  recall |
| :----------------------- | -------: | --------: | ------: |
| GPT-2 / Zero Shot        |  0.70784 |   0.83863 | 0.51472 |
| GPT-2 / Fine-Tuned       |  0.92360 |   0.92923 | 0.91704 |
| GPT-2 / Fine-Tuned(LoRA) |  0.91068 |   0.89946 | 0.92472 |
| BERT / Fine-Tuned        |   0.9150 |    0.9122 |  0.9076 |
| BERT / Fine-Tuned(LoRA)  |   0.8855 |    0.8647 |  0.9034 |
