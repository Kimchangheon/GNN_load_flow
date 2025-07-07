import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from GNN.SimpleGNN.GNSNode import GNSNode
from GNN.SimpleGNN.GNSMsg import GNSMsg

# from GNN.SimpleGNN.Dataset import ChanghunDataset
from GNN.SimpleGNN.Dataset_optimized import ChanghunDataset

from helper import *
# from collate_blockdiag import *
from collate_blockdiag_optimized import *

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
parser.add_argument("--RUNNAME", type=str, default="test", help="Name of the run and output model")
parser.add_argument("--BLOCK_DIAG", action="store_true", help="Use block diagonal mode")
parser.add_argument("--NORMALIZE", action="store_true", help="Enable Physics-Informed Neural Networks")
parser.add_argument("--PER_UNIT", action="store_true", help="Enable Physics-Informed Neural Networks")

parser.add_argument("--float64", action="store_true", help="Enable Physics-Informed Neural Networks")
parser.add_argument('--mode', type=str, default="train_test", help='train_valid_test | train | valid | test')
parser.add_argument("--mag_ang_mse", action="store_true", help="normalised |V| + wrapped-angle loss ")


parser.add_argument("--ADJ_MODE", type=str, default="cplx", help="Adjacency mode: real | cplx | other")

parser.add_argument('--weight_init', type=str, default="sd0.02", help='Weight initialization method (None, He, sd0.02)')
parser.add_argument('--bias_init', type=float, default=0.0, help='Bias initialization value (0.01, 0)')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay rate')

parser.add_argument("--BATCH", type=int, default=16, help="Batch size")
parser.add_argument("--EPOCHS", type=int, default=10, help="Number of training epochs")
parser.add_argument("--LR", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--VAL_EVERY", type=int, default=1, help="Validation frequency (in epochs)")
parser.add_argument("--PARQUET", type=str, default="./data/u_start_repaired_800_variations_4_8_16_32_bus_grid_Ybus.parquet", help="Path to Parquet data file")

args = parser.parse_args()

# Assign to variables if needed
PINN       = args.PINN
RUNNAME    = args.RUNNAME
BLOCK_DIAG = args.BLOCK_DIAG
NORMALIZE = args.NORMALIZE
PER_UNIT = args.PER_UNIT

PINN       = True
BLOCK_DIAG = False
NORMALIZE = False
PER_UNIT = True

args.mag_ang_mse = True

ADJ_MODE   = args.ADJ_MODE
BATCH      = args.BATCH
# BATCH=1
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
full_ds = ChanghunDataset(PARQUET, per_unit=PER_UNIT, device=device)
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

# # ── utils ---------------------------------------------------------
def compute_norm_stats(dataset: torch.utils.data.Dataset):
    """Return (μ_bus, σ_bus, μ_edge, σ_edge) as 1-D tensors.
       bus-feat order = (|V|, θ, ΔP, ΔQ); edge-feat order = (G, B)"""
    bus_feats, edge_feats = [], []

    for s in dataset:               # <-- iterates over the *training subset* only
        v, th = s['V_start'][..., 0], s['V_start'][..., 1]
        dP    = s['P_newton'] - s['P_start']
        dQ    = s['Q_newton'] - s['Q_start']
        bus_feats.append(torch.stack([v, th, dP, dQ], -1).reshape(-1, 4))

        Yr, Yi = s['Ybus_real'], s['Ybus_imag']
        edge_feats.append(torch.stack([Yr, Yi], -1).reshape(-1, 2))

    bus_feat  = torch.cat(bus_feats,  0).float()
    edge_feat = torch.cat(edge_feats, 0).float()
    μ_bus,  σ_bus  = bus_feat.mean(0),  bus_feat.std(0).clamp_min(1e-6)
    μ_edge, σ_edge = edge_feat.mean(0), edge_feat.std(0).clamp_min(1e-6)
    return μ_bus, σ_bus, μ_edge, σ_edge
# -----------------------------------------------------------------

μ_bus, σ_bus, μ_edge, σ_edge = compute_norm_stats(train_ds)
print('μ_bus', μ_bus, '\nσ_bus', σ_bus, '\nμ_edge', μ_edge, '\nσ_edge', σ_edge)

# ------------------------------------------------------------------
# 3.  Model / Optim / Loss
# ------------------------------------------------------------------

if NORMALIZE :
    model = GNSMsg(
                   d=10, K=30, pinn=PINN,
                    μ_bus=μ_bus.to(device),
                    σ_bus=σ_bus.to(device),
                    μ_edge=μ_edge.to(device),
                    σ_edge=σ_edge.to(device),
    ).to(device)
else :
    model  = GNSMsg(d=10, K=30, pinn=PINN).to(device)

# model = model.double()

def init_weights(model, exclude_modules):
    for module in model.modules():
        if module in exclude_modules:
            continue
        if isinstance(module, nn.Linear):
            if args.weight_init == "sd0.02":
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            elif args.weight_init == "He":
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(args.bias_init)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        else:
            for name, param in module.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    if args.weight_init == "sd0.02":
                        torch.nn.init.normal_(param, mean=0, std=0.02)
                    elif args.weight_init == "He":
                        torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif 'bias' in name:
                    param.data.fill_(args.bias_init)


exclude_modules = []
init_weights(model, exclude_modules)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


total_params = count_parameters(model)
print(f"Total number of parameters: {total_params}")


# optim  = torch.optim.Adam(model.parameters(), lr=LR)
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=args.weight_decay)
loss_f = torch.nn.MSELoss()

def mag_ang_mse(Vpred, Vref, w_mag=1.0, w_ang=1/torch.pi):
    """
    Scale-balanced MSE:
      • magnitude difference divided by w_mag
      • angle error wrapped to (-π,π] and divided by w_ang
    """
    dmag = (Vpred[..., 0] - Vref[..., 0]) / w_mag
    dang = torch.atan2(
        torch.sin(Vpred[..., 1] - Vref[..., 1]),
        torch.cos(Vpred[..., 1] - Vref[..., 1])
    ) / w_ang
    return torch.mean(dmag ** 2 + dang ** 2)

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

    sum_mse_mag = 0.0
    sum_mse_ang = 0.0

    n_samples = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            B = batch["bus_type"].size(0)        # current mini-batch size
            n_samples += B

            # move to device ------------------------------------------------
            if BLOCK_DIAG :
                n_nodes_per_graph = batch["sizes"].to(device)
            else :
                n_nodes_per_graph = None
            bus_type = batch["bus_type"].to(device)
            Line    = batch["Lines_connected"].to(device)
            Yr       = batch["Ybus_real"].to(device)
            Yi       = batch["Ybus_imag"].to(device)
            Ysr       = batch["Y_Lines_real"].to(device)
            Ysi       = batch["Y_Lines_imag"].to(device)
            Yc       = batch["Y_C_Lines"].to(device)

            Pstart   = batch["P_start"].to(device)
            Qstart   = batch["Q_start"].to(device)
            Pnewton   = batch["P_newton"].to(device)
            Qnewton   = batch["Q_newton"].to(device)

            Ustart = batch["U_start"].to(device)
            Vstart   = batch["V_start"].to(device)

            Unewton = batch["U_newton"].to(device)
            Vnewton    = batch["V_newton"].to(device)

            # Ustart = Unewton.clone()
            # Vstart = Vnewton.clone()
            #
            # Pstart, Qstart = Pnewton, Qnewton
            # forward -------------------------------------------------------
            if pinn:
                Vpred, loss_phys = model(bus_type, Line, Yr, Yi, Ysr, Ysi, Yc, Pstart, Qstart, Vstart, Ustart, n_nodes_per_graph)
                assert isinstance(Vpred, torch.Tensor), "expected tensor from model"
                if args.mag_ang_mse :
                    w_mag = 1.0
                    w_ang = 1 / torch.pi
                    dmag = (Vpred[..., 0] - Vnewton[..., 0]) / w_mag
                    dang = torch.atan2(
                        torch.sin(Vpred[..., 1] - Vnewton[..., 1]),
                        torch.cos(Vpred[..., 1] - Vnewton[..., 1])
                    ) / w_ang
                    mse_mag = torch.mean(dmag ** 2)
                    mse_ang = torch.mean(dang ** 2)
                    mse = mse_mag + mse_ang
                else :
                    # Vpred = Vpred[..., 0] * torch.exp(1j * Vpred[..., 1]) # pytorch doesn't support complex
                    mse  = loss_f(Vpred, Vnewton)          # supervised metric only
                loss = loss_phys
            else:
                Vpred = model(bus_type, Line, Yr, Yi, Ysr, Ysi, Yc, Pstart, Qstart, Vstart, Ustart, n_nodes_per_graph)
                assert isinstance(Vpred, torch.Tensor), "expected tensor from model"
                if args.mag_ang_mse :
                    w_mag = 1.0
                    w_ang = 1 / torch.pi
                    dmag = (Vpred[..., 0] - Vnewton[..., 0]) / w_mag
                    dang = torch.atan2(
                        torch.sin(Vpred[..., 1] - Vnewton[..., 1]),
                        torch.cos(Vpred[..., 1] - Vnewton[..., 1])
                    ) / w_ang
                    mse_mag = torch.mean(dmag ** 2)
                    mse_ang = torch.mean(dang ** 2)
                    mse = mse_mag + mse_ang
                else :
                    # Vpred = Vpred[..., 0] * torch.exp(1j * Vpred[..., 1])
                    mse  = loss_f(Vpred, Vnewton)
                loss = mse                          # same number for convenience

            # backward / optim ---------------------------------------------
            if train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

            # aggregate -----------------------------------------------------
            sum_loss += loss.item() * B
            sum_mse  += mse.item()  * B
            sum_mse_mag += mse_mag.item()  * B
            sum_mse_ang += mse_ang.item()  * B

    mean_loss = sum_loss / n_samples
    mean_mse  = sum_mse  / n_samples
    mean_mse_mag = sum_mse_mag / n_samples
    mean_mse_ang = sum_mse_ang / n_samples
    return mean_loss, mean_mse, mean_mse_mag, mean_mse_ang        # always a tuple
# ------------------------------------------------------------------
# 5.  Training loop
# ------------------------------------------------------------------


# ---------------------------------------------------------------------
# 5.  Training / validation
# ---------------------------------------------------------------------
if "train" in args.mode:
    train_loss_hist, train_mse_hist        = [], []
    train_mse_mag_hist, train_mse_ang_hist = [], []

    val_loss_hist,   val_mse_hist          = [], []
    val_mse_mag_hist, val_mse_ang_hist     = [], []

    best_val_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # -------- train pass ------------------------------------------
        train_loss, train_mse, train_mse_mag, train_mse_ang = \
            run_epoch(train_loader, train=True, pinn=PINN)

        train_loss_hist.append(train_loss)
        train_mse_hist.append(train_mse)
        train_mse_mag_hist.append(train_mse_mag)
        train_mse_ang_hist.append(train_mse_ang)

        # -------- validation ------------------------------------------
        if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
            val_loss, val_mse, val_mse_mag, val_mse_ang = \
                run_epoch(val_loader, train=False, pinn=PINN)

            val_loss_hist.append(val_loss)
            val_mse_hist.append(val_mse)
            val_mse_mag_hist.append(val_mse_mag)
            val_mse_ang_hist.append(val_mse_ang)

            print(f"Epoch {epoch:3d} | "
                  f"train loss {train_loss:.4e}  mse {train_mse:.4e} "
                  f"(mag {train_mse_mag:.4e}, ang {train_mse_ang:.4e}) | "
                  f"valid loss {val_loss:.4e}  mse {val_mse:.4e} "
                  f"(mag {val_mse_mag:.4e}, ang {val_mse_ang:.4e}) | "
                  f"time {time.time()-t0:.2f}s")

            # ---- checkpoint on physics loss --------------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = f"./results/ckpt/{RUNNAME}_{EPOCHS}_best_model.ckpt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"  ↳ checkpoint saved to {ckpt_path}")
        else:
            # validation skipped this epoch
            print(f"Epoch {epoch:3d} | "
                  f"train loss {train_loss:.4e}  mse {train_mse:.4e} "
                  f"(mag {train_mse_mag:.4e}, ang {train_mse_ang:.4e}) | "
                  f"time {time.time()-t0:.2f}s")

    # -----------------------------------------------------------------
    # 6.  Curves
    # -----------------------------------------------------------------
    import matplotlib.pyplot as plt, os
    os.makedirs("./results/plots", exist_ok=True)

    epochs = range(1, len(train_loss_hist) + 1)

    if PINN:
        # --- Physics loss --------------------------------------------
        plt.figure(figsize=(6,4))
        plt.plot(epochs, train_loss_hist, label="Train Physics Loss")
        plt.plot(epochs[:len(val_loss_hist)], val_loss_hist,
                 label="Validation Physics Loss")
        plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("PINN: Physics Loss"); plt.legend(); plt.tight_layout()
        plt.savefig(f"./results/plots/{RUNNAME}_physics_loss.png"); plt.clf()

    # --- Supervised losses (all modes) -------------------------------
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_mse_hist, label="Train MSE (phasor/all)")
    plt.plot(epochs[:len(val_mse_hist)], val_mse_hist,
             label="Val MSE (phasor/all)")
    plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.title("Supervised MSE"); plt.legend(); plt.tight_layout()
    plt.savefig(f"./results/plots/{RUNNAME}_mse_total.png"); plt.clf()

    # --- Magnitude vs Angle ------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10,4), sharey=True)

    ax[0].plot(epochs, train_mse_mag_hist, label="Train |V|")
    ax[0].plot(epochs[:len(val_mse_mag_hist)], val_mse_mag_hist,
               label="Val |V|"); ax[0].set_title("Magnitude MSE")
    ax[0].set_yscale("log"); ax[0].set_xlabel("Epoch")

    ax[1].plot(epochs, train_mse_ang_hist, label="Train θ")
    ax[1].plot(epochs[:len(val_mse_ang_hist)], val_mse_ang_hist,
               label="Val θ"); ax[1].set_title("Angle MSE")
    ax[1].set_yscale("log"); ax[1].set_xlabel("Epoch")

    for a in ax: a.legend()
    fig.suptitle("Magnitude vs Angle MSE"); fig.tight_layout()
    fig.savefig(f"./results/plots/{RUNNAME}_mse_components.png"); plt.close(fig)

# ------------------------------------------------------------------
# 7.  Final test evaluation
# ------------------------------------------------------------------
if "test" in args.mode:
    best_ckpt_path = f"./results/ckpt/{RUNNAME}_{EPOCHS}_best_model.ckpt"
    print(f"\nLoading checkpoint from: {best_ckpt_path}")
    model.load_state_dict(torch.load(best_ckpt_path))

    # run_epoch now returns:  mean_loss, mean_mse, mean_mse_mag, mean_mse_ang
    test_loss, test_mse, test_mse_mag, test_mse_ang = \
        run_epoch(test_loader, train=False, pinn=PINN)

    if PINN:
        print(
            f"\nTest physics-loss : {test_loss:.4e}"
            f" | total MSE : {test_mse:.4e}"
            f" | |V| MSE : {test_mse_mag:.4e}"
            f" | θ MSE : {test_mse_ang:.4e}"
        )
    else:
        # in non-PINN mode test_loss == test_mse by design
        print(
            f"\nFinal test-set MSE : {test_mse:.4e}"
            f" (|V|: {test_mse_mag:.4e}, θ: {test_mse_ang:.4e})"
        )
