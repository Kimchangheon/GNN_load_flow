import torch
from torch.utils.data import DataLoader, random_split
from GNN.SimpleGNN.GNS import GNSSolver
from GNN.SimpleGNN.Dataset import ChanghunDataset
from helper import *
from collate_blockdiag import *
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------------------------------------------------------
# 0.  Hyper-parameters
# ------------------------------------------------------------------
RUNNAME = "BLOCK_20M"
BLOCK_DIAG =False
BATCH     =  16
EPOCHS    = 100
LR        = 1e-3
VAL_EVERY = 1        # epochs
PARQUET   = "212100_variations_4_8_16_32_bus_grid.parquet"

# ------------------------------------------------------------------
# 1.  Device
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------------------
# 2.  Dataset  →  Train/Valid/Test split
# ------------------------------------------------------------------
full_ds = ChanghunDataset(PARQUET)
n_total = len(full_ds)
n_train = int(0.8 * n_total)
n_val   = int(0.1 * n_total)
n_test  = n_total - n_train - n_val

train_ds, val_ds, test_ds = random_split(
        full_ds,
        lengths=[n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42))   # reproducible split


if BATCH ==1 :
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)
else :
    if BLOCK_DIAG :
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,collate_fn=collate_blockdiag)
        val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,collate_fn=collate_blockdiag)
        test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,collate_fn=collate_blockdiag)
    else :
        sizes = [full_ds[i]["N"] for i in range(len(full_ds))]  # list of ints
        # train_loader = make_size_bucketing_loader(train_ds, BATCH, shuffle=True)
        # val_loader   = make_size_bucketing_loader(val_ds,   BATCH, shuffle=False)
        # test_loader  = make_size_bucketing_loader(test_ds,  BATCH, shuffle=False)

        # build bucketed loaders
        train_sampler = MultiBucketBatchSampler(
            sizes=np.take(sizes, train_ds.indices),
            batch_size=BATCH,
            shuffle=True)

        val_sampler = MultiBucketBatchSampler(
            sizes=np.take(sizes, val_ds.indices),
            batch_size=BATCH,
            shuffle=False)

        test_sampler = MultiBucketBatchSampler(
            sizes=np.take(sizes, test_ds.indices),
            batch_size=BATCH,
            shuffle=False)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler)
        test_loader = DataLoader(test_ds, batch_sampler=test_sampler)




print(f"Dataset sizes  |  train {n_train}   valid {n_val}   test {n_test}")

# ------------------------------------------------------------------
# 3.  Model / Optim / Loss
# ------------------------------------------------------------------
model  = GNSSolver(d=100, K=10).to(device)
optim  = torch.optim.Adam(model.parameters(), lr=LR)
loss_f = torch.nn.MSELoss()

# ------------------------------------------------------------------
# 4.  Helper: one epoch pass (train=False → no grad)
# ------------------------------------------------------------------
def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total, count = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            bus_type = batch["bus_type"].to(device)
            Yr       = batch["Ybus_real"].to(device)
            Yi       = batch["Ybus_imag"].to(device)
            Pspec    = batch["P_spec"].to(device)
            Qspec    = batch["Q_spec"].to(device)
            Vtrue    = batch["V_true"].to(device)

            Vpred = model(bus_type, Yr, Yi, Pspec, Qspec)
            loss  = loss_f(Vpred, Vtrue)

            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()

            total += loss.item() * bus_type.size(0)
            count += bus_type.size(0)
    return total / count

# ------------------------------------------------------------------
# 5.  Training loop
# ------------------------------------------------------------------
train_hist, val_hist = [], []

best_val_mse = float('inf')  # Initialize best validation MSE to a very high value

for epoch in range(1, EPOCHS + 1):
    train_mse = run_epoch(train_loader, train=True)
    train_hist.append(train_mse)

    if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
        val_mse = run_epoch(val_loader, train=False)
        val_hist.append(val_mse)
        print(f"Epoch {epoch:3d} | train MSE {train_mse:.4e} | "
              f"valid MSE {val_mse:.4e}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            # Save model checkpoint
            torch.save(model.state_dict(), RUNNAME+"_"+str(EPOCHS)+"_best_model.ckpt")
            print("Checkpoint saved.")

    else:
        print(f"Epoch {epoch:3d} | train MSE {train_mse:.4e}")

import matplotlib.pyplot as plt
epochs = range(1, len(train_hist) + 1)

plt.figure()
plt.plot(epochs, train_hist, label="Train MSE")
plt.plot(epochs, val_hist,   label="Valid MSE")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Mean-Squared Error")
plt.title("Training / Validation loss")
plt.legend()
plt.tight_layout()

plt.savefig("loss_curve_block_diagonal_batching.png")   # saved next to the script
plt.show()

# ------------------------------------------------------------------
# 6.  Final test evaluation
# ------------------------------------------------------------------
test_mse = run_epoch(test_loader, train=False)
print(f"\nFinal test-set MSE: {test_mse:.4e}")