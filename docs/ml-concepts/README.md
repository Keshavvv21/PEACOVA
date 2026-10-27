# ML/LLM Concepts — Deep Dive Series

A from-first-principles reference covering the concepts behind the Algo Akanade agent stack in this repo (`orchestrator_framework_layer/akanade/`) — from what an LLM actually is, up through the RAG, RL/bandit, agent, and guardrail layers built on top of it. Each folder is a standalone README: definition, the underlying math/algorithm ("how"), a block diagram, worked example, and — where the concept is actually implemented in this codebase — exact file:line pointers into the real code.

## Reading order

1. [01-llm](01-llm/README.md) — What is an LLM: tokens, transformer architecture, autoregressive decoding, training stages (pretrain → SFT → RLHF/DPO)
2. [07-embeddings](07-embeddings/README.md) — Vector representations: word2vec → contextual → sentence embeddings, similarity metrics, types used in retrieval
3. [02-rag](02-rag/README.md) — Retrieval-Augmented Generation: chunking, indexing, retrieval, re-ranking, generation, failure modes
4. [09-optimizers](09-optimizers/README.md) — How gradient descent optimizers (SGD → Adam/AdamW) are built, and how they update transformer weights
5. [10-neural-network-layers](10-neural-network-layers/README.md) — The layers inside a transformer block: embedding, attention, feed-forward, normalization, residuals
6. [08-lora-qlora](08-lora-qlora/README.md) — Parameter-efficient fine-tuning: LoRA/QLoRA math, hyperparameters, and how this repo's `finetune_local_slm.py` uses it
7. [03-rl-algorithms](03-rl-algorithms/README.md) — Reinforcement learning family tree: bandits → policy gradient → PPO → RLHF/DPO for LLMs
8. [12-thompson-sampling](12-thompson-sampling/README.md) — Thompson Sampling and the contextual bandit that powers this repo's `hybrid_router.py`
9. [11-shapley-values](11-shapley-values/README.md) — Game-theoretic feature attribution (SHAP): the math, and how it explains model/feature importance
10. [06-intent-types](06-intent-types/README.md) — Intent recognition approaches, and how this repo's `intent_reasoner.py` classifies requests
11. [04-agents](04-agents/README.md) — Agent architectures (ReAct, plan-execute, tool use), and how this repo's `coding_agent.py` implements one
12. [05-guardrails](05-guardrails/README.md) — Guardrail types (input/output filtering, policy enforcement, PEFT-safety checks) and where they'd sit in this stack
13. [13-graph-rag](13-graph-rag/README.md) — GraphRAG: building a knowledge graph from documents, graph-based retrieval, community summarization, when it beats vector RAG
14. [20-advanced-rag-techniques](20-advanced-rag-techniques/README.md) — Beyond naive RAG: contextual retrieval, RAPTOR, ColBERT/late-interaction, agentic RAG, late chunking
15. [18-attention-optimizations](18-attention-optimizations/README.md) — FlashAttention, multi-query/grouped-query attention, sliding-window attention — making the attention layer fast and memory-cheap
16. [15-quantization](15-quantization/README.md) — INT8/INT4 quantization: PTQ vs QAT, GPTQ, AWQ, GGUF, bitsandbytes — how and why models shrink
17. [16-memory-optimization-training](16-memory-optimization-training/README.md) — Training-time memory: mixed precision, gradient checkpointing, ZeRO/FSDP, activation memory accounting
18. [17-inference-serving-optimization](17-inference-serving-optimization/README.md) — Serving-time optimization: KV cache, continuous batching, paged attention (vLLM), speculative decoding
19. [19-distributed-training](19-distributed-training/README.md) — Data/tensor/pipeline parallelism — how training is split across many GPUs
20. [14-cuda-gpu-optimization](14-cuda-gpu-optimization/README.md) — How CUDA/GPU optimization actually works: kernels, memory hierarchy, kernel fusion, why this all matters for LLM speed
21. [21-performance-optimization-methodology](21-performance-optimization-methodology/README.md) — How to actually find and fix performance problems: profiling tools, compute-vs-memory-vs-I/O-bound diagnosis, benchmarking methodology, and a decision framework tying every optimization technique above (quantization, batching, FlashAttention, ZeRO, etc.) to the bottleneck it fixes
22. [23-data-pipeline-slm-finetuning](23-data-pipeline-slm-finetuning/README.md) — Where the nightly fine-tune's training data actually comes from: four logged collection sources, the PEFT Analyzer scraper that solves cold start, and PII-scrubbing + near-duplicate curation into an ever-growing corpus
23. [22-peft-hyperparameter-optimization](22-peft-hyperparameter-optimization/README.md) — Picking tonight's LoRA rank/alpha/reward-threshold with the same Thompson Sampling bandit that powers `hybrid_router.py`, learned from whether last night's choice actually passed the regression gate

Each README is self-contained and can be read independently.
