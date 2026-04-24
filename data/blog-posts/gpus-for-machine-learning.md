# GPUs for Machine Learning and AI: What to Buy in 2025

Choosing a GPU for machine learning is a different decision to choosing one for gaming. The benchmarks that matter, the specs to prioritise, and the trade-offs involved are all different. This guide explains what to look for and which GPUs are worth considering for local AI and ML workloads.

## Why VRAM is the most important spec

In gaming, VRAM matters but raw performance matters more. In machine learning, VRAM is the single most important spec. It determines what model sizes you can run, what batch sizes you can train with, and whether a workload is even possible on a given card.

Running out of VRAM during training or inference doesn't just slow things down — it causes the process to fail entirely, or forces the workload to spill into system RAM, which is dramatically slower.

As a rough guide: 12GB of VRAM is enough to run inference on most 7B parameter models in 4-bit quantisation. 16GB gives you more comfortable headroom for larger models and larger batch sizes. 24GB opens up 13B parameter models and serious fine-tuning work.

For most developers running local LLMs, doing fine-tuning experiments, or building ML-powered applications, 16GB is the practical minimum worth targeting in 2025.

## Other specs that matter for ML

**Memory bandwidth** affects how quickly data can be moved in and out of VRAM during computation. Higher bandwidth means faster training and inference. GDDR7 memory, found in NVIDIA's RTX 50 series, has significantly higher bandwidth than GDDR6.

**Tensor cores and AI accelerators** are hardware units designed specifically for the matrix multiplication operations that underpin neural network training and inference. NVIDIA's Tensor Cores and AMD's AI accelerators both provide significant speedups over general-purpose shader cores for these workloads.

**Software ecosystem** is a real consideration. NVIDIA's CUDA platform has been the standard for ML frameworks including PyTorch and TensorFlow for years. AMD's ROCm platform has improved significantly but compatibility is still not as broad as CUDA. If you're following tutorials, using pre-built Docker images, or working with frameworks that have CUDA-specific optimisations, an NVIDIA card will have fewer friction points.

## The best GPUs for machine learning

### Best overall: NVIDIA GeForce RTX 5080
The RTX 5080 is the best consumer GPU for machine learning workloads. Its 16GB of GDDR7 VRAM provides enough headroom for serious inference and fine-tuning work, and its memory bandwidth is substantially higher than previous generation cards. CUDA support is full and mature. At $999 MSRP it is expensive, but for developers doing regular ML work it is a worthwhile investment. The 360W TDP requires a quality PSU and good airflow.

### Best mid-range option: NVIDIA GeForce RTX 5070
The RTX 5070 is a reasonable mid-range option for ML workloads. Its 12GB of GDDR7 VRAM is workable for inference on smaller models and light fine-tuning, though 12GB will feel limiting if your workloads grow. It is a better choice for gaming than for serious ML work, but if you want one card that does both reasonably well, it is a viable option. CUDA support is full.

### AMD option: AMD Radeon RX 9070 XT
The RX 9070 XT's 16GB of GDDR6 VRAM makes it more appealing for ML workloads than the RTX 5070 on paper. However, AMD's ROCm software stack is less mature than CUDA, and some ML frameworks and tools have limited or no ROCm support. If you are working within a well-supported framework like PyTorch with ROCm, it is a credible option. If you rely on CUDA-specific libraries or are following CUDA-based tutorials, stick with NVIDIA.

## What you can realistically run

With **12GB VRAM** (RTX 5070): inference on 7B models in 4-bit quantisation, small batch training on compact models, stable diffusion image generation.

With **16GB VRAM** (RTX 5080, RX 9070 XT): inference on 13B models in 4-bit quantisation, fine-tuning smaller models, more comfortable batch sizes for training.

These figures assume the GPU is being used exclusively for ML. Running a desktop environment and other applications simultaneously will reduce available VRAM.

## Things that won't help

Raw clock speed matters much less for ML than for gaming. A higher boost clock does not translate linearly to faster training. VRAM and memory bandwidth are much better predictors of ML performance than shader clock speeds.

Similarly, features like ray tracing cores and upscaling support (DLSS, FSR) are irrelevant for ML workloads. Don't pay a premium for these features if gaming is not part of your use case.

## Summary

For machine learning workloads, prioritise VRAM over everything else. The RTX 5080 is the best consumer option if budget allows. If you are primarily doing ML rather than gaming, the RX 9070 XT's 16GB of VRAM makes it worth considering despite AMD's less mature software ecosystem. Avoid the RTX 5070 if ML is your primary use case and you anticipate working with larger models.