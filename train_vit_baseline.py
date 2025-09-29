#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, random, time
from pathlib import Path
from typing import List, Tuple
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def human_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

def get_mean_std_from_weights(weights):
    # 안전한 기본값 (ImageNet)
    default_mean = [0.485, 0.456, 0.406]
    default_std  = [0.229, 0.224, 0.225]
    if weights is None:
        return default_mean, default_std
    # torchvision 최신: transforms()에서 추출
    try:
        tf = weights.transforms()
        if hasattr(tf, "mean") and hasattr(tf, "std"):
            return list(tf.mean), list(tf.std)
    except Exception:
        pass
    # 구버전: meta 딕셔너리에서 추출
    try:
        meta = getattr(weights, "meta", {})
        return list(meta.get("mean", default_mean)), list(meta.get("std", default_std))
    except Exception:
        return default_mean, default_std

# -----------------------------
# 데이터셋 (TS_/VS_ 접두사 제거: 기존 ResNet 스크립트와 동일 아이디어)  :contentReference[oaicite:1]{index=1}
# -----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def _find_split_dir(data_root: Path, split_key: str) -> Path:
    for p in data_root.iterdir():
        if p.is_dir() and split_key.lower() in p.name.lower():
            return p
    raise FileNotFoundError(f"'{split_key}' 폴더를 찾지 못했습니다: {data_root}")

def _find_source_dir(split_dir: Path) -> Path:
    candidates = ["01.원천데이터", "1.원천데이터", "원천데이터", "01. 원천데이터", "1. 원천데이터"]
    for name in candidates:
        p = split_dir / name
        if p.exists() and p.is_dir():
            return p
    return split_dir  # fallback

def _strip_prefix(folder_name: str) -> str:
    # TS_, VS_, TE_ 등 제거
    if len(folder_name) > 3 and folder_name[2] == "_" and folder_name[:2].isalpha():
        return folder_name[3:]
    if len(folder_name) > 3 and folder_name[2] in {"-", " "} and folder_name[:2].isalpha():
        return folder_name[3:]
    return folder_name

def _scan_split(source_dir: Path) -> Tuple[List[Tuple[Path,int]], List[str]]:
    subdirs = [d for d in source_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"이미지 클래스 폴더를 찾지 못했습니다: {source_dir}")

    class_name_to_dirs = {}
    for d in subdirs:
        cname = _strip_prefix(d.name).strip()
        class_name_to_dirs.setdefault(cname, []).append(d)

    class_names = sorted(class_name_to_dirs.keys())
    samples: List[Tuple[Path,int]] = []
    for idx, cname in enumerate(class_names):
        for d in class_name_to_dirs[cname]:
            for p in d.rglob("*"):
                if p.is_file() and _is_image(p):
                    samples.append((p, idx))

    if not samples:
        raise FileNotFoundError(f"이미지 파일을 찾지 못했습니다: {source_dir}")
    return samples, class_names

class SkinSplitDataset(Dataset):
    def __init__(self, data_root: Path, split_key: str, image_size: int, mean, std, train: bool):
        from torchvision.transforms.functional import InterpolationMode
        split_dir = _find_split_dir(data_root, split_key)
        source_dir = _find_source_dir(split_dir)
        self.samples, self.class_names = _scan_split(source_dir)

        interp = InterpolationMode.BICUBIC  # ViT 권장
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=interp),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(8, interpolation=interp),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(int(image_size * 1.14), interpolation=interp),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        return x, y

# -----------------------------
# ViT 모델 만들기
# -----------------------------
def build_vit_model(num_classes: int, arch: str = "vit_b_16", dropout: float = 0.1, pretrained: bool = True):
    # torchvision 버전별 가중치 이름 호환
    weights = None
    if arch == "vit_b_16":
        try:
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            model = models.vit_b_16(weights=weights)
        except AttributeError:
            model = models.vit_b_16(weights=None)
    elif arch == "vit_b_32":
        try:
            weights = models.ViT_B_32_Weights.DEFAULT if pretrained else None
            model = models.vit_b_32(weights=weights)
        except AttributeError:
            model = models.vit_b_32(weights=None)
    else:
        raise ValueError("arch must be vit_b_16 or vit_b_32")

    # 헤드 교체(드롭아웃 포함). torchvision 버전별로 heads 구조가 다를 수 있어 안전하게 처리
    in_dim = None
    if hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        in_dim = model.heads.head.in_features
        model.heads.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_dim, num_classes))
    elif isinstance(model.heads, nn.Linear):
        in_dim = model.heads.in_features
        model.heads = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_dim, num_classes))
    elif isinstance(model.heads, nn.Sequential):
        # 마지막 Linear를 찾아 치환
        linear = None
        for m in reversed(model.heads):
            if isinstance(m, nn.Linear):
                linear = m; break
        if linear is None:
            raise RuntimeError("Unexpected ViT heads structure.")
        in_dim = linear.in_features
        model.heads = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_dim, num_classes))
    else:
        raise RuntimeError("Unsupported ViT heads structure.")

    return model, weights

# -----------------------------
# Top-k
# -----------------------------
@torch.no_grad()
def accuracy_topk(output, target, k=1):
    topk = output.topk(k, dim=1).indices
    return topk.eq(target.view(-1,1)).any(dim=1).float().mean().item() * 100.0

# -----------------------------
# 한 에폭 학습/평가
# -----------------------------
def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    from contextlib import nullcontext
    model.train() if train else model.eval()

    # CUDA에서만 AMP 사용 (MPS/CPU는 X)
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        autocast_ctx = lambda: torch.amp.autocast(device_type="cuda")
    else:
        scaler = None
        autocast_ctx = nullcontext

    n, running_loss = 0, 0.0
    y_true_list, y_pred_list = [], []

    mode = "Train" if train else "Eval"
    pbar = tqdm(loader, desc=mode, ncols=100, leave=False)

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        bs = x.size(0)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = model(x)
                loss = criterion(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                with autocast_ctx():
                    logits = model(x)
                    loss = criterion(logits, y)

        running_loss += loss.item() * bs
        n += bs
        avg_loss = running_loss / max(1, n)
        pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}", "seen": n})

        y_true_list.append(y.detach().cpu())
        y_pred_list.append(logits.argmax(1).detach().cpu())

    epoch_loss = running_loss / max(1, n)
    y_true = torch.cat(y_true_list).numpy()
    y_pred = torch.cat(y_pred_list).numpy()
    acc_top1 = (y_true == y_pred).mean() * 100.0

    # 검증/테스트에서 top-3 재계산
    acc_top3 = None
    if not train:
        correct1 = correct3 = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                a1 = accuracy_topk(logits, y, k=1) / 100.0
                a3 = accuracy_topk(logits, y, k=3) / 100.0
                correct1 += a1 * x.size(0)
                correct3 += a3 * x.size(0)
                total += x.size(0)
        acc_top1 = (correct1 / max(1,total)) * 100.0
        acc_top3 = (correct3 / max(1,total)) * 100.0

    metrics = {"loss": float(epoch_loss), "acc_top1": float(acc_top1)}
    if acc_top3 is not None: metrics["acc_top3"] = float(acc_top3)
    if _HAS_SKLEARN:
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
    return metrics, (y_true, y_pred)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="ViT Baseline (TS_/VS_ prefix aware)")
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--out_dir", default="./runs/vit_b16_baseline", type=str)
    ap.add_argument("--arch", default="vit_b_16", choices=["vit_b_16", "vit_b_32"])
    ap.add_argument("--image_size", default=224, type=int)  # ViT 기본 224 권장
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--lr", default=1e-4, type=float)       # ViT는 ResNet보다 작은 lr 권장
    ap.add_argument("--weight_decay", default=0.05, type=float)
    ap.add_argument("--dropout", default=0.1, type=float)
    ap.add_argument("--num_workers", default=2, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--no_pretrained", action="store_true", help="사전학습 가중치 미사용(오프라인 등)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[Device] {device}")

    # 모델 & 가중치 / mean,std
    model, weights = build_vit_model(
        num_classes=15, arch=args.arch, dropout=args.dropout,
        pretrained=not args.no_pretrained
    )
    mean, std = get_mean_std_from_weights(weights)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 데이터셋 구성 (TS_/VS_ 접두사 제거)  :contentReference[oaicite:2]{index=2}
    train_ds = SkinSplitDataset(data_root, "training",   args.image_size, mean, std, train=True)
    try:
        val_ds   = SkinSplitDataset(data_root, "validation", args.image_size, mean, std, train=False)
    except FileNotFoundError:
        val_ds   = SkinSplitDataset(data_root, "valid",      args.image_size, mean, std, train=False)
    try:
        test_ds  = SkinSplitDataset(data_root, "test",       args.image_size, mean, std, train=False)
    except FileNotFoundError:
        test_ds  = val_ds

    class_names = train_ds.class_names
    num_classes = len(class_names)
    if num_classes != 15:
        # 클래스 수가 15가 아닐 경우 헤드를 재설정
        model, _ = build_vit_model(num_classes=num_classes, arch=args.arch, dropout=args.dropout, pretrained=not args.no_pretrained)

    print(f"[Classes] ({len(class_names)}): {class_names}")

    # 로더
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    # 학습 세팅
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = []
    best_val = float("inf")
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    start = time.time()
    for epoch in range(1, args.epochs+1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        tr, _ = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va, _ = run_one_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step(va["loss"])

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"], "train_acc_top1": tr["acc_top1"],
            "val_loss": va["loss"],   "val_acc_top1": va["acc_top1"],
            "val_acc_top3": va.get("acc_top3"),
            "val_f1_macro": va.get("f1_macro"), "val_f1_weighted": va.get("f1_weighted"),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

        # save last
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "args": vars(args), "classes": class_names}, last_path)
        # save best
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "args": vars(args), "classes": class_names}, best_path)
            print(f"[Checkpoint] Best updated: {best_path}")

        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    # 테스트 평가(베스트 로드)
    ck = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ck["model_state"])
    model = model.to(device)
    print("\n[Evaluate] Best checkpoint on TEST")
    te, (y_true, y_pred) = run_one_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(json.dumps({"test": te}, ensure_ascii=False, indent=2))

    # 리포트 저장
    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    with open(out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    if _HAS_SKLEARN:
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        cm = confusion_matrix(y_true, y_pred).tolist()
        with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        with open(out_dir / "confusion_matrix.json", "w", encoding="utf-8") as f:
            json.dump({"labels": class_names, "cm": cm}, f, ensure_ascii=False, indent=2)
        print("\n[Classification Report]\n" + report)

    print(f"\n[Done] Total time: {human_time(time.time()-start)}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")

if __name__ == "__main__":
    main()
