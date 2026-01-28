import torch
from torch.utils.data import DataLoader
from dataset_all import get_dataset
from model import DiscreteVFI
import torch.optim as optim

device = "cuda"

ds = get_dataset("vimeo", "vimeo_triplet", split="train")
dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

model = DiscreteVFI().to(device)
opt = optim.AdamW(model.parameters(), lr=2e-4)

for epoch in range(10):
    for batch in dl:
        I0 = batch["I0"].to(device)
        I1 = batch["I1"].to(device)
        It = batch["It"].to(device)
        t = batch["t"].to(device)

        pred = model(I0, I1, t)
        loss = (pred - It).abs().mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch} | Loss {loss.item():.4f}")
