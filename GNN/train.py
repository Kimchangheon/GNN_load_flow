
import torch
from torch.utils.data import DataLoader
from GNS import *
from Dataset import *

# Automatically select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to device
model = GNSSolver(d=100, K=10).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_f = nn.MSELoss()

# Load dataset and dataloader
dataset = ChanghunDataset("Changhun_multi.parquet")
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
# loaders = {N: DataLoader(dsN[N], batch_size=32, shuffle=True) for N in [4,8,16,32]}


# Training loop
for epoch in range(100):
    running = 0.0
    for batch in train_loader:
        # Move batch data to device
        bus_type = batch["bus_type"].to(device)
        Yr       = batch["Ybus_real"].to(device)
        Yi       = batch["Ybus_imag"].to(device)
        Pspec    = batch["P_spec"].to(device)
        Qspec    = batch["Q_spec"].to(device)
        Vtrue    = batch["V_true"].to(device)

        # Forward pass
        Vpred = model(bus_type, Yr, Yi, Pspec, Qspec)  # shape: (B,N,2)

        # Loss + backward + update
        loss = loss_f(Vpred, Vtrue)
        optim.zero_grad()
        loss.backward()
        optim.step()

        running += loss.item()

    print(f"Epoch {epoch:3d} | Train Loss: {running/len(train_loader):.4e}")