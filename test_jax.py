import time
import jax
import jax.numpy as jnp
import functools

# 假设这是一个大对象（比如 3B 模型 wrapper）
class DummyBigModel:
    def __init__(self, size=10**7):
        # 模拟巨大的 Python 对象
        self.data = list(range(size))

    def apply(self, params, x):
        # 其实只依赖 params，不依赖 self.data
        return x * params


# ---------------------
# 情况 1: 把 big_model 放进 static_argnames
# ---------------------
@functools.partial(jax.jit, static_argnames=("big_model",))
def fn_static(x, params, big_model):
    return big_model.apply(params, x)


# ---------------------
# 情况 2: 把 big_model 固定在闭包，只传 params
# ---------------------
def make_fn_nostatic(big_model):
    @jax.jit
    def fn(x, params):
        return big_model.apply(params, x)
    return fn


# ---------------------
# 运行对比
# ---------------------
x = jnp.ones((1000,))
params = 2.0
big_model = DummyBigModel(size=10**6)  # 模拟一个比较大的对象

# 1. static_argnames 包含大对象
t0 = time.time()
y1 = fn_static(x, params, big_model).block_until_ready()
print("with static_argnames big_model: {:.3f}s".format(time.time() - t0))

# 2. 只传 params，不把大对象放进 jit
fn_nostatic = make_fn_nostatic(big_model)
t0 = time.time()
y2 = fn_nostatic(x, params).block_until_ready()
print("without static_argnames big_model: {:.3f}s".format(time.time() - t0))
