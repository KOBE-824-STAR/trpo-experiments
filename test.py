import torch
from torch.distributions import Normal, TransformedDistribution, Independent
from torch.distributions.transforms import TanhTransform
from torch.distributions.kl import kl_divergence

torch.manual_seed(0)

# =========================
# 1. 构造两个高斯分布
# =========================
batch_size = 4
action_dim = 3
num_samples = 200000

mu1 = torch.tensor([[0.2, -0.5, 1.0]]).expand(batch_size, action_dim)
log_std1 = torch.tensor([[-0.1, 0.2, -0.3]]).expand(batch_size, action_dim)
std1 = log_std1.exp()

mu2 = torch.tensor([[-0.3, 0.1, 0.7]]).expand(batch_size, action_dim)
log_std2 = torch.tensor([[0.1, -0.2, 0.0]]).expand(batch_size, action_dim)
std2 = log_std2.exp()

base_dist1 = Independent(Normal(mu1, std1), 1)
base_dist2 = Independent(Normal(mu2, std2), 1)

# =========================
# 2. 计算原始高斯的解析 KL
# =========================
base_kl = kl_divergence(base_dist1, base_dist2)   # shape: [batch]
print("Analytic KL of base Gaussians:")
print(base_kl)
print()

# =========================
# 3. 构造 tanh 变换后的分布
# =========================
tanh = TanhTransform(cache_size=1)
tanh_dist1 = TransformedDistribution(base_dist1, [tanh])
tanh_dist2 = TransformedDistribution(base_dist2, [tanh])

print(kl_divergence(tanh_dist1,tanh_dist2), kl_divergence(tanh_dist1.base_dist,tanh_dist2.base_dist))