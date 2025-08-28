import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from flax import linen as nn

# ----------------------------
# 定义一个简单的 MLP 模型
# ----------------------------
class BigMLP(nn.Module):
    hidden_dim: int = 1024
    out_dim: int = 256
    depth: int = 4

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x

# ----------------------------
# 初始化模型参数
# ----------------------------
key = random.PRNGKey(0)
model = BigMLP()
dummy_input = jnp.ones((1, 512))  # 单条样本输入
params = model.init(key, dummy_input)

# jit 编译的 forward
@jax.jit
def forward(params, x):
    return model.apply(params, x)

# ----------------------------
# 定义计时函数
# ----------------------------
def benchmark(batch_size, n_iter=50, warmup=10):
    x = jnp.ones((batch_size, 512))
    # warmup（避免编译影响）
    for _ in range(warmup):
        _ = forward(params, x).block_until_ready()

    start = time.time()
    for _ in range(n_iter):
        _ = forward(params, x).block_until_ready()
    end = time.time()

    avg = (end - start) / n_iter
    return avg

# ----------------------------
# 运行对比
# ----------------------------
for B in [1, 16, 64, 128, 256, 512]:
    t = benchmark(B)
    print(f"Batch {B:3d}: {t*1000:.3f} ms (per sample: {t*1000/B:.3f} ms)")
