"""Is the synthetic task learnable from the feature the model actually reads out?"""
import importlib.util
import torch

spec = importlib.util.spec_from_file_location("t", r"Z:\asi_x\scripts\train.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

from ssha.utils import rate_encode

for (N, dim, C, seed, mu) in [(64, 32, 5, 0, 1.0), (64, 32, 2, 0, 1.0),
                              (64, 32, 2, 0, 3.0), (64, 32, 5, 0, 3.0)]:
    sample, W_rule = t.make_task(N, dim, C, seed, mu_scale=mu)
    g = torch.Generator().manual_seed(seed + 2)
    X, y = sample(256, g)
    enc = torch.sigmoid(2.0 * X)               # 编码的期望
    rate = enc.mean(dim=1)                     # rate_vec: [n, dim] 标签所依赖的量
    xseq = rate_encode(X, 8, gain=2.0)         # [T, n, N, dim]

    # 模型读出的是最后一步的节点均值；这里用输入的节点均值做线性探针
    feat = xseq[-1].mean(dim=1)                # [n, dim]
    # 最小二乘线性探针（闭式解）+ argmax 分类
    A = torch.cat([feat, torch.ones(feat.size(0), 1)], dim=1)
    Y = torch.nn.functional.one_hot(y, C).float()
    sol = torch.linalg.lstsq(A, Y).solution
    probe_acc = float(((A @ sol).argmax(dim=1) == y).float().mean())
    # 直接用 rate_vec（标签的原始定义量）做线性探针的上界
    A2 = torch.cat([rate, torch.ones(rate.size(0), 1)], dim=1)
    sol2 = torch.linalg.lstsq(A2, Y).solution
    probe2 = float(((A2 @ sol2).argmax(dim=1) == y).float().mean())

    print(f"N={N} dim={dim} C={C} mu={mu}: majority={t.majority_baseline(y):.3f} "
          f"probe_on_input_nodemean={probe_acc:.3f} probe_on_rate_vec={probe2:.3f} "
          f"classes={int(torch.bincount(y,minlength=C).gt(0).sum())}")
