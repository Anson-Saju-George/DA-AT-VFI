import torch
from model import DiscreteVFI

device = "cuda"

m = DiscreteVFI().to(device)

I0 = torch.randn(2, 3, 256, 256).to(device)
I1 = torch.randn(2, 3, 256, 256).to(device)
t  = torch.rand(2, 1).to(device)

y = m(I0, I1, t)
print("Output:", y.shape)
