# training/train_sr.py
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from Core.data import SRDataset
from Core.models.espcn import ESPCN
from Core.models.srcnn import SRCNN
from Core.models.edsr_lite import EDSRLite

def build_model(name, scale):
    name = name.lower()
    if name == "espcn":
        return ESPCN(scale=scale)
    if name == "srcnn":
        return SRCNN()
    if name == "edsr_lite":
        return EDSRLite(scale=scale)
    raise ValueError(f"Unknown model {name}")

def psnr(mse):
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dossier HR")
    ap.add_argument("--model", default="espcn", choices=["espcn","srcnn","edsr_lite"])
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--out", default="checkpoints/espcn_x2.pth")
    ap.add_argument("--val_every", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model, args.scale).to(device)

    ds = SRDataset(args.data, scale=args.scale, patch_size=args.patch, augment=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)

    # L1 (Charbonnier soft-L1 optionnelle)
    criterion = nn.L1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_path = Path(args.out)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        running_mse, running_loss = 0.0, 0.0
        for lr, hr in dl:
            lr, hr = lr.to(device), hr.to(device)
            if args.model == "srcnn":
                # SRCNN attend l'upscale bicubique en entrée
                b, c, h, w = lr.shape
                lr_up = torch.nn.functional.interpolate(lr, scale_factor=args.scale, mode="bicubic", align_corners=False)
                sr = model(lr_up)
                target = hr
            else:
                sr = model(lr)
                target = hr

            loss = criterion(sr, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            mse = torch.mean((sr-target)**2).detach()
            running_loss += loss.item()
            running_mse += mse.item()

        avg_loss = running_loss / len(dl)
        avg_psnr = psnr(torch.tensor(running_mse / len(dl))).item()
        print(f"[{epoch}/{args.epochs}] loss={avg_loss:.4f} PSNR~{avg_psnr:.2f}dB")

        if epoch % args.val_every == 0:
            torch.save({"model": model.state_dict(),
                        "scale": args.scale,
                        "arch": args.model}, ckpt_path)

if __name__ == "__main__":
    main()
