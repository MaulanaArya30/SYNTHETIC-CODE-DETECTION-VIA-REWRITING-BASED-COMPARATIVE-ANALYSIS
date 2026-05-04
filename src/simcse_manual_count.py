import torch
import torch.nn.functional as F

def simcse_loss(z1, z2, temperature=0.1):
    batch_size = z1.shape[0]
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    sim_11 = torch.matmul(z1, z1.T) / temperature
    sim_22 = torch.matmul(z2, z2.T) / temperature
    sim_12 = torch.matmul(z1, z2.T) / temperature
    mask = torch.eye(batch_size, dtype=torch.bool)
    sim_11 = sim_11.masked_fill(mask, -9e15)
    sim_22 = sim_22.masked_fill(mask, -9e15)
    logits = torch.cat([sim_12, sim_11], dim=1)
    labels = torch.arange(batch_size)
    loss = F.cross_entropy(logits, labels)
    return loss

# Same vectors as manual calculation above
z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
z2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

print(f"Code output: {simcse_loss(z1, z2).item():.5f}")
# They should match!