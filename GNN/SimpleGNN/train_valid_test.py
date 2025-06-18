import torch
from torch.utils.data import DataLoader, random_split
from GNN.SimpleGNN.GNS import GNSSolver
from GNN.SimpleGNN.Dataset import ChanghunDataset
from helper import *
from collate_blockdiag import *
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.makedirs("./results/ckpt", exist_ok=True)
os.makedirs("./results/plots", exist_ok=True)

# ------------------------------------------------------------------
# 0.  Hyper-parameters
# ------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser(description="Training script with configurable hyperparameters")

parser.add_argument("--PINN", action="store_true", help="Enable Physics-Informed Neural Networks")
parser.add_argument("--RUNNAME", type=str, default="PINN_BLOCK_cplx_20M", help="Name of the run and output model")
parser.add_argument("--BLOCK_DIAG", action="store_true", help="Use block diagonal mode")
parser.add_argument("--ADJ_MODE", type=str, default="cplx", help="Adjacency mode: real | cplx | other")
parser.add_argument("--BATCH", type=int, default=16, help="Batch size")
parser.add_argument("--EPOCHS", type=int, default=10, help="Number of training epochs")
parser.add_argument("--LR", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--VAL_EVERY", type=int, default=1, help="Validation frequency (in epochs)")
parser.add_argument("--PARQUET", type=str, default="./data/212100_variations_4_8_16_32_bus_grid.parquet", help="Path to Parquet data file")

args = parser.parse_args()

# Assign to variables if needed
PINN       = args.PINN
PINN       = False
RUNNAME    = args.RUNNAME
BLOCK_DIAG = args.BLOCK_DIAG
ADJ_MODE   = args.ADJ_MODE
BATCH      = args.BATCH
EPOCHS     = args.EPOCHS
LR         = args.LR
VAL_EVERY  = args.VAL_EVERY
PARQUET    = args.PARQUET

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
model  = GNSSolver(pinn_flag=PINN, adj_mode=ADJ_MODE,d=10, K=30).to(device)
optim  = torch.optim.Adam(model.parameters(), lr=LR)
loss_f = torch.nn.MSELoss()

# ------------------------------------------------------------------
# 4.  Helper: one epoch pass (train=False → no grad)
# ------------------------------------------------------------------
# ------------------------------------------------------
# helper: one full pass over a loader
# ------------------------------------------------------
def run_epoch(loader, *, train: bool, pinn: bool):
    model.train() if train else model.eval()

    sum_loss = 0.0          # physics   (always)
    sum_mse  = 0.0          # |V|,δ MSE (only meaningful when pinn=True)
    n_samples = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            B = batch["bus_type"].size(0)        # current mini-batch size
            n_samples += B

            # move to device ------------------------------------------------
            bus_type = batch["bus_type"].to(device)
            Yr       = batch["Ybus_real"].to(device)
            Yi       = batch["Ybus_imag"].to(device)
            Pstart   = batch["P_start"].to(device)
            Qstart   = batch["Q_start"].to(device)
            Vstart   = batch["V_start"].to(device)
            Vtrue    = batch["V_true"].to(device)

            # forward -------------------------------------------------------
            if pinn:
                Vpred, loss_phys = model(bus_type, Yr, Yi, Pstart, Qstart, Vstart)
                loss = loss_phys
                mse  = loss_f(Vpred, Vtrue)          # supervised metric only
            else:
                Vpred = model(bus_type, Yr, Yi, Pstart, Qstart, Vstart)
                loss  = loss_f(Vpred, Vtrue)
                mse   = loss                          # same number for convenience

            # backward / optim ---------------------------------------------
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()

            # aggregate -----------------------------------------------------
            sum_loss += loss.item() * B
            sum_mse  += mse.item()  * B

    mean_loss = sum_loss / n_samples
    mean_mse  = sum_mse  / n_samples
    return mean_loss, mean_mse        # always a tuple
# ------------------------------------------------------------------
# 5.  Training loop
# ------------------------------------------------------------------
train_loss_hist, train_mse_hist = [], []
val_loss_hist,   val_mse_hist   = [], []

best_val_loss = float('inf')                 # best physics-loss so far

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # -------- train pass ------------------------------------------------
    train_loss, train_mse = run_epoch(train_loader, train=True, pinn=PINN)
    train_loss_hist.append(train_loss)
    train_mse_hist.append(train_mse)

    # -------- validation ------------------------------------------------
    if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
        val_loss, val_mse = run_epoch(val_loader, train=False, pinn=PINN)
        val_loss_hist.append(val_loss)
        val_mse_hist.append(val_mse)

        print(f"Epoch {epoch:3d} | "
              f"train loss {train_loss:.4e}  mse {train_mse:.4e} | "
              f"valid loss {val_loss:.4e}  mse {val_mse:.4e} | "
              f"time {time.time()-t0:.2f}s")

        # -------- checkpointing on *physics* loss -----------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = f"./results/ckpt/{RUNNAME}_{EPOCHS}_best_model.ckpt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ checkpoint saved to {ckpt_path}")
    else:
        # validation skipped this epoch
        print(f"Epoch {epoch:3d} | "
              f"train loss {train_loss:.4e}  mse {train_mse:.4e} | "
              f"time {time.time()-t0:.2f}s")

# ------------------------------------------------------------------
# 6.  Curves
# ------------------------------------------------------------------
import matplotlib.pyplot as plt
epochs = range(1, len(train_loss_hist) + 1)

plt.figure(figsize=(6,4))
plt.plot(epochs, train_loss_hist, label="Train loss (physics)" if PINN else "Train MSE")
plt.plot(epochs[:len(val_loss_hist)], val_loss_hist,   label="Valid loss (physics)" if PINN else "Valid MSE")
if PINN:                 # show supervised metric as well
    plt.plot(epochs, train_mse_hist, '--', label="Train MSE")
    plt.plot(epochs[:len(val_mse_hist)], val_mse_hist,  '--', label="Valid MSE")

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training / Validation curves")
plt.legend()
plt.tight_layout()
plot_path = f"./results/plots/{RUNNAME}.png"
plt.savefig(plot_path)
plt.show()
print(f"Plot saved to {plot_path}")

# ------------------------------------------------------------------
# 7.  Final test evaluation
# ------------------------------------------------------------------
test_loss, test_mse = run_epoch(test_loader, train=False, pinn=PINN)
if PINN:
    print(f"\nTest physics-loss : {test_loss:.4e} | Test MSE : {test_mse:.4e}")
else:
    print(f"\nFinal test-set MSE: {test_loss:.4e}")