---
title: Volatile GPU Util
subtitle: Brief explanation of what it means and why it should always be 100%
date: 2025-04-06
blurb: What does Volatile GPU Util when running nvidia-smi mean? 
---

When running training or inference of a deep learning model on a GPU, we want the GPU to be "utilized" optimally. 

Typically, when I run training using PyTorch, in a different terminal window, I run `watch -n 1 nvidia-smi`. This gives me an idea of how much GPU memory is being used. The volatile GPU util field indicates the percentage of time operations are running on the GPU. If the data is not in GPU memory, GPU will be idle when data is being fetched from the main memory to the GPU. This results in under-utilization of the GPU and typically indicates the data loader needs to be optimized.