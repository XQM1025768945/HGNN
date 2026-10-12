"""Verify the adversarial reviewer's findings C1, C2, C4, C5 myself."""
import torch
import torch.nn.functional as F

from ssha.block import SSHABlock
from ssha.model import SSHAModel
from ssha.weight_node import SpikingWeightNode

print("=" * 74)
print("C1  soft gate ignores the refractory mask `active`")
print("=" * 74)
torch.manual_seed(0)
wn = SpikingWeightNode(N=1, M=1, beta=0.85)
with torch.no_grad():
    wn.v_th_base.fill_(torch.log(torch.expm1(torch.tensor(0.2))))
    wn.w_pre.zero_(); wn.w_post.zero_()
    wn.b.fill_(0.6)
    wn.H_logit.zero_()
wn.reset_state()
print(f"{'t':>2} {'active':>7} {'u_pre':>8} {'s_hard':>7} {'gate':>9} {'H_soft':>9} {'H_hard':>8}")
for t in range(5):
    d = wn.s_acc  # noqa
    ref = wn.ref.clone()
    active = (torch.clamp_min(ref - 1.0, 0.0) <= 0).float()
    with torch.no_grad():
        H_soft = wn(torch.zeros(1), torch.zeros(1, 1), hard=False)
        u_pre = wn.u.clone()          # post-reset; capture via fresh state below
    wn.reset_state()
    with torch.no_grad():
        H_hard = wn(torch.zeros(1), torch.zeros(1, 1), hard=True)
    # recompute u_pre by replaying one step manually
    print(f"{t:>2} {float(active):>7.0f} {'-':>8} {'-':>7} {float(H_hard):>9.4f} "
          f"{float(H_soft):>9.4f} {float(H_hard):>8.4f}")

print("\n-- direct check: is soft H_eff > 0 while active == 0 (refractory)? --")
torch.manual_seed(0)
sw = SpikingWeightNode(N=1, M=1, beta=0.85)
with torch.no_grad():
    sw.v_th_base.fill_(torch.log(torch.expm1(torch.tensor(0.2))))
    sw.w_pre.zero_(); sw.w_post.zero_(); sw.b.fill_(0.6); sw.H_logit.zero_()
sw.reset_state()
for t in range(4):
    active_before = (sw.ref <= 0).float()
    with torch.no_grad():
        H_soft = float(sw(torch.zeros(1), torch.zeros(1, 1), hard=False))
    print(f"  t={t} ref_before={float(sw.ref):.1f} active={float(active_before):.0f} "
          f"H_soft={H_soft:.6f}  {'<-- nonzero during refractory' if active_before == 0 and H_soft > 0 else ''}")

print()
print("=" * 74)
print("C2  |e|==2 contract at PRODUCTION init vs test override")
print("=" * 74)
N, M, d = 8, 4, 6
for label, force_wpre in (("test override w_pre=1.0", True), ("production init", False)):
    viol = 0
    for seed in range(12):
        torch.manual_seed(seed)
        blk = SSHABlock(d, N, M)
        with torch.no_grad():
            H_pair = torch.zeros(N, M)
            for e, (i, j) in enumerate([(0, 1), (2, 3), (4, 5), (6, 7)]):
                H_pair[i, e] = 0.5
                H_pair[j, e] = 0.5
            blk.sw.H_logit.data = torch.log(H_pair + 1e-9)
            blk.sw.temp = 1e-3
            if force_wpre:
                blk.sw.w_pre.data.fill_(1.0)
            blk.sw.get_H_base(refresh=True)
        with torch.no_grad():
            _, _, _, H_eff, _ = blk(torch.randn(N, d), torch.ones(N), torch.zeros(M, d), hard=True)
        nnz = (H_eff > 0).sum(dim=0)
        if not bool(torch.all(nnz == 2)):
            viol += 1
    print(f"  {label:26s}: {viol}/12 seeds violate 'exactly 2 nonzeros per column'")

print()
print("=" * 74)
print("C4  hard mode struct_mass == 0 for whole trajectories, small configs")
print("=" * 74)
for (N, M, dim, T) in [(8, 4, 4, 2), (8, 4, 4, 8), (16, 8, 8, 4), (16, 4, 4, 8),
                       (32, 8, 8, 8), (64, 16, 32, 8)]:
    zero_sm = 0
    for seed in range(24):
        torch.manual_seed(seed)
        m = SSHAModel(dim, N, M, 5, T=T)
        with torch.no_grad():
            _, aux = m(torch.randn(T, N, dim), hard=True)
        if float(aux["struct_mass"]) == 0.0:
            zero_sm += 1
    print(f"  N={N:3d} M={M:3d} dim={dim:3d} T={T:2d}: {zero_sm:2d}/24 seeds struct_mass==0")

print()
print("=" * 74)
print("C5  are the synthetic task's labels near-constant?")
print("=" * 74)


def make_task_current(N, dim, classes, seed, gain=2.0):
    g = torch.Generator().manual_seed(seed)
    W_rule = torch.randn(classes, dim, generator=g) / (dim ** 0.5)

    def sample(n, generator):
        X = torch.randn(n, N, dim, generator=generator)
        rate = torch.sigmoid(gain * X).mean(dim=1)
        y = (rate @ W_rule.T).argmax(dim=1)
        return X, y
    return sample


for (N, dim, C, seed) in [(64, 32, 5, 0), (64, 32, 5, 1), (64, 32, 2, 0), (32, 16, 10, 0)]:
    s = make_task_current(N, dim, C, seed)
    g = torch.Generator().manual_seed(seed + 1)
    X, y = s(64, g)
    counts = torch.bincount(y, minlength=C).tolist()
    maj = max(counts) / 64
    rate = torch.sigmoid(2.0 * X).mean(dim=1)
    g2 = torch.Generator().manual_seed(seed)
    W_rule = torch.randn(C, dim, generator=g2) / (dim ** 0.5)
    score_std = float((rate @ W_rule.T).std(dim=0).mean())
    print(f"  N={N} dim={dim} C={C} seed={seed}: counts={counts}")
    print(f"      majority-class acc={maj:.3f}   per-sample score std={score_std:.4f}")
