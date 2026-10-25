# Project Scope: Distributed LLM Training Framework

## Goal

I'm building a distributed training framework for large language models, modeled after TorchTitan. I have two goals. The first is to get good at making models run fast on NVIDIA hardware. The second is to understand how a model gets split across many GPUs. So far I've worked through the JAX scaling book and written CUDA kernels. This project builds on that by putting those ideas into a working training system, end to end.

## What I'm Building

The framework will support four kinds of parallelism, added one at a time: data parallelism (DP), then fully sharded data parallelism (FSDP), then sequence parallelism (SP), and finally expert parallelism (EP). EP comes last because it needs a mixture-of-experts model and depends on the other pieces already working. The end goal is a framework where all four axes run at the same time in a single training job, not four separate features that only work alone.

## Measuring Performance

Getting the parallelism right is only half the work. I'll profile every stage, measure throughput, and plot strong-scaling curves to see how well performance holds up as I add GPUs. I'll also track loss next to step time. A faster step isn't worth much if the model learns worse, and balancing step speed against how quickly loss goes down is a big part of real training work. Training dynamics are something I want to understand over time, but the first priority is a correct, measurable, scalable system.

## Framework and Development Setup

I'm still choosing between PyTorch and JAX. JAX has a real practical advantage: almost all of the sharding and distributed logic can be built and tested on a CPU with a small setup, and only kernel calls need a GPU. That means I can do most of the development locally and save GPU time for final checks and benchmarks. Doing the same on CPU in PyTorch is less clear and may be harder. Either way, TorchTitan is my main reference. There's no paper that explains how to build a training framework, so reading a clean, well-structured open-source codebase is the best guide available, even though TorchTitan is written for PyTorch rather than JAX.

## Model Size and Hardware

I'm not aiming for a specific model size. The model just needs to fit in memory on the hardware I have. I'll use the cheapest hardware that does the job, which will probably be A100s. Newer hardware like Blackwell only makes sense if the work turns into writing kernels specifically.


## Milestones

- [ ] Read through TorchTitan's source as the reference implementation
- [ ] Choose a framework (PyTorch vs. JAX)
- [ ] Get compute (Modal student credits, ThunderCompute, kernel competitions)
- [ ] Implement DP
- [ ] Implement FSDP
- [ ] Implement SP
- [ ] Implement EP with an MoE model
- [ ] Run all parallelism axes at the same time
- [ ] Profile throughput and strong scaling on A100s
- [ ] Track loss alongside step time
- [ ] Contribute something upstream to an open-source project
