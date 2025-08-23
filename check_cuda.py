import torch
import subprocess

print("==== PyTorch / CUDA 环境检查 ====")

# PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")
print(f"PyTorch 编译时 CUDA 版本: {torch.version.cuda}")
print(f"是否检测到 GPU: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA runtime 版本(通过PyTorch): {torch.version.cuda}")

# NVIDIA 驱动 & CUDA runtime
try:
    nvidia_smi = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
    print("\n=== nvidia-smi 输出 ===")
    print("\n".join(nvidia_smi.splitlines()[:10]))  # 只打印前10行
except Exception as e:
    print("nvidia-smi 无法运行:", e)


import jax, jaxlib, platform, subprocess

print("🐍 Python:", platform.python_version())
print("📦 jax:", jax.__version__)
print("📦 jaxlib:", jaxlib.__version__)

# CUDA / cuDNN 版本
print("🟢 jaxlib CUDA version:", getattr(jaxlib, "cuda_version", "N/A"))
print("🟢 jaxlib cuDNN version:", getattr(jaxlib, "cudnn_version", "N/A"))

# GPU 信息
print("💻 JAX devices:", jax.devices())

