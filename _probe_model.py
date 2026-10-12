"""Can a linear probe on the MODEL's own readout feature recover the label?"""
import importlib.util
import torch

spec = importlib.util.spec_from_file_location("t", r"Z:\asi_x\scripts\train.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

from ssha.model import SSHAModel
from ssha.utils import rate_encode

N, dim, C, seed = 64, 32, 5, 0
sample, W_rule = t.make_task(N, dim, C, seed, mu_scale=3.0)
g = torch.Generator().manual_seed(seed + 2)
X, y = sample(256, g)
xseq = rate_encode(X, 8, gain=2.0)

torch.manual_seed(0)
model = SSHAModel(dim, N, M=32, num_classes=C, T=8)
model.eval()

feats = []
with torch.no_grad():
    for b in range(X.size(0)):
        _, aux = model(xseq[:, b], hard=False)
        feats.append(model.classifier.in_features * [0])   # placeholder
# recompute readout feature explicitly: last hidden node-mean
feats = []
with torch.no_grad():
    for b in range(X.size(0)):
        model.reset_state()
        x = xseq[0, b]
        s_v = torch.zeros(N)
        y_e = torch.zeros(model.block.M, dim)
        for step in range(model.T):
            x, s_v, y_e, _, _ = model.block(xseq[step, b], s_v, y_e, hard=False)
        feats.append(x.mean(dim=0))
F = torch.stack(feats)                     # [n, dim] 模型读出用的特征

A = torch.cat([F, torch.ones(F.size(0), 1)], dim=1)
Y = torch.nn.functional.one_hot(y, C).float()
sol = torch.linalg.lstsq(A, Y).solution
probe = float(((A @ sol).argmax(dim=1) == y).float().mean())
maj = t.majority_baseline(y)
print(f"frozen-model readout feature: linear probe = {probe:.3f}  majority = {maj:.3f}")
print(f"feature std per dim (mean) = {float(F.std(dim=0).mean()):.4f}")
print(f"feature ||mean over batch|| = {float(F.mean(dim=0).norm()):.4f}")

# also: can the classifier head alone fit the train set? (upper bound of the frozen feature)
lin = torch.nn.Linear(dim, C)
opt = torch.optim.Adam(lin.parameters(), lr=0.05)
for it in range(500):
    loss = torch.nn.functional.cross_entropy(lin(F), y)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"trainable head on frozen features: acc = "
      f"{float((lin(F).argmax(1) == y).float().mean()):.3f}")
