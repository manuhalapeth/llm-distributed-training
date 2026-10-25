# Papers

These are the papers I've read for this project.

**1) TorchTitan (Liang et al., 2024)**  
https://arxiv.org/pdf/2410.06511

This paper presents TorchTitan, PyTorch's own system for training large language models across many GPUs. It shows how data, tensor and pipeline parallelism are built as modular pieces that can be stacked together on a device mesh, alongside checkpointing, Float8 training and torch.compile. It is the reference design that my framework follows.

**2) ZeRO (Rajbhandari et al., 2019)**  
https://arxiv.org/pdf/1910.02054

This paper introduces the core idea behind FSDP: instead of every GPU storing a full copy of the model, the optimizer state, gradients and weights are split across GPUs. It explains the three stages of sharding and how much memory each one saves. It is the foundation for the FSDP stage of my project.

**3) PyTorch FSDP (Zhao et al., 2023)**  
https://arxiv.org/pdf/2304.11277

This paper explains how PyTorch implements fully sharded data parallelism in practice. It covers how weights are gathered before each layer and freed after, how gradients are reduce scattered, and how communication is overlapped with compute. It is the practical guide for building FSDP myself.

**4) Ring Attention (Liu et al., 2023)**  
https://arxiv.org/pdf/2310.01889

This paper shows how to split a long sequence across GPUs and still compute exact attention. Each GPU holds a block of tokens and passes its keys and values around a ring of GPUs while computing. It is the basis for the sequence parallelism stage of my project.

**5) GShard (Lepikhin et al., 2020)**  
https://arxiv.org/pdf/2006.16668

This paper scales mixture of experts models by placing different experts on different GPUs. It introduces token routing, expert capacity limits and a load balancing loss, with tokens exchanged between GPUs through all to all communication. It is the starting point for expert parallelism.

**6) Switch Transformers (Fedus et al., 2021)**  
https://arxiv.org/pdf/2101.03961

This paper simplifies mixture of experts by sending each token to just one expert. It shows that this makes training simpler, cheaper and more stable while still scaling to trillion parameter models. It guides the routing and load balancing design in the EP stage.

**7) Megatron LM at Scale (Narayanan et al., 2021)**  
https://arxiv.org/pdf/2104.04473

This paper shows how to combine data, tensor and pipeline parallelism to train very large models on thousands of GPUs. It explains how to choose the size of each parallelism group and how each choice affects throughput. It is the main reference for running all my parallelism axes at once and measuring how well they scale.
