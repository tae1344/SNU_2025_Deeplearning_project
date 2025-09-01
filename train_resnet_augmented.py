#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, random, time
from pathlib import Path
from typing import List, Tuple, Optional
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# -----------------------------
# 파일/경로 & 데이터셋 유틸
# -----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def _find_source_dir(split_dir: Path) -> Path:
    """'원천데이터' 표준 폴더명이 있을 경우 우선 사용, 없으면 split_dir 바로 아래에서 클래스 폴더 탐색"""
    candidates = ["01.원천데이터", "1.원천데이터", "원천데이터", "01. 원천데이터", "1. 원천데이터"]
    for name in candidates:
        p = split_dir / name
        if p.exists() and p.is_dir():
            return p
    return split_dir  # fallback

def _strip_prefix(folder_name: str) -> str:
    """TS_, VS_, TE_ 등 접두사 제거"""
    if len(folder_name) > 3 and folder_name[2] == "_" and folder_name[:2].isalpha():
        return folder_name[3:]
    if len(folder_name) > 3 and folder_name[2] in {"-", " "} and folder_name[:2].isalpha():
        return folder_name[3:]
    return folder_name

def _scan_split(source_dir: Path) -> Tuple[List[Tuple[Path,int]], List[str]]:
    """source_dir 아래 클래스 폴더들을 스캔하여 (이미지경로, 클래스인덱스) 목록과 클래스명 목록을 반환"""
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

def _dirnames(path: Path):
    try:
        return [d.name for d in path.iterdir() if d.is_dir()]
    except Exception:
        return []

def _has_class_dirs(root: Path) -> bool:
    """root 아래가 바로 클래스 구조인지(또는 원천데이터 폴더 아래가 클래스 구조인지) 판단"""
    src_candidates = ["01.원천데이터", "1.원천데이터", "원천데이터", "01. 원천데이터", "1. 원천데이터"]
    for name in src_candidates:
        p = root / name
        if p.exists() and p.is_dir():
            subdirs = [d for d in p.iterdir() if d.is_dir()]
            if subdirs:
                return True

    for d in root.iterdir():
        if not d.is_dir():
            continue
        for pp in d.rglob("*"):
            if pp.is_file() and _is_image(pp):
                return True
    return False

def _find_split_dir(data_root: Path, split_key: str) -> Path:
    """
    'training' / 'validation' / 'test' 에 대해 다양한 동의어를 인식.
    분할 폴더가 전혀 없는 단일 구조라면 'training' 요청 시 data_root 자체를 사용.
    """
    key = split_key.lower()
    synonyms = {
        "training":   ["training", "train", "tr", "훈련", "학습"],
        "validation": ["validation", "valid", "val", "dev", "검증", "검정"],
        "test":       ["test", "testing", "te", "테스트", "시험"],
    }
    cands = synonyms.get(key, [key])

    for p in data_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if any(c in name for c in cands):
            return p

    if key == "training" and _has_class_dirs(data_root):
        return data_root

    raise FileNotFoundError(
        f"'{split_key}' 폴더를 찾지 못했습니다: {data_root}\n"
        f"존재하는 하위 폴더: { _dirnames(data_root) }"
    )


# -----------------------------
# 공통 유틸
# -----------------------------
def get_mean_std_from_weights(weights):
    """torchvision weights에서 mean/std 추출 (실패 시 ImageNet 기본값)"""
    default_mean = [0.485, 0.456, 0.406]
    default_std  = [0.229, 0.224, 0.225]
    if weights is None:
        return default_mean, default_std
    try:
        tf = weights.transforms()
        if hasattr(tf, "mean") and hasattr(tf, "std"):
            return list(tf.mean), list(tf.std)
    except Exception:
        pass
    try:
        meta = getattr(weights, "meta", {})
        mean = meta.get("mean", default_mean)
        std  = meta.get("std",  default_std)
        return list(mean), list(std)
    except Exception:
        return default_mean, default_std

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

def device_pretty(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            name = "CUDA"
        return f"cuda ({name})"
    elif device.type == "mps":
        return "mps (Apple Silicon Metal Performance Shaders)"
    else:
        return "cpu"

def human_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# -----------------------------
# 재현성: DataLoader worker seeding
# -----------------------------
def worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    try:
        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass


# -----------------------------
# Data Transforms
# -----------------------------
def build_transforms(image_size: int, mean, std, train: bool, aug_cfg: Optional[dict] = None):
    """
    Train:
      - RandomResizedCrop(scale=(min,1.0))
      - (optional) RandomHorizontalFlip(p)
      - (light) ColorJitter
      - ToTensor + Normalize
    Val/Test:
      - Resize(int(image_size*val_resize_ratio)) + CenterCrop(image_size)
      - ToTensor + Normalize
    """
    if aug_cfg is None: aug_cfg = {}
    scale_min = float(aug_cfg.get("scale_min", 0.7))
    color     = aug_cfg.get("color", (0.1, 0.1, 0.1, 0.05))
    flip_p    = float(aug_cfg.get("flip_p", 0.0))
    val_rr    = float(aug_cfg.get("val_resize_ratio", 1.15))

    if train:
        tfs = [s.RandomResizedCrop(image_size, scale=(scale_min, 1.0))]
        if flip_p and flip_p > 0:
            tfs.append(transforms.RandomHorizontalFlip(p=flip_p))
        b, c, s, h = color
        if any(v > 0 for v in (b, c, s, h)):
            tfs.append(transforms.ColorJitter(b, c, s, h))
        tfs += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(tfs)
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * val_rr)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# -----------------------------
# 데이터셋
# -----------------------------
class SkinSplitDataset(Dataset):
    def __init__(self, data_root: Path, split_key: str, image_size: int, mean, std,
                 train: bool, aug_cfg: Optional[dict] = None):
        split_dir = _find_split_dir(data_root, split_key)  # Training/Validation/Test 또는 루트
        source_dir = _find_source_dir(split_dir)
        self.samples, self.class_names = _scan_split(source_dir)
        self.tf = build_transforms(image_size, mean, std, train=train, aug_cfg=aug_cfg)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        return x, y  # 학습 루프와 맞춤 (for x, y in loader)


# -----------------------------
# 모델
# -----------------------------
def build_model(num_classes: int, dropout: float, arch: str = "resnet101"):
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)
    elif arch == "resnet101":
        weights = models.ResNet101_Weights.IMAGENET1K_V2
        backbone = models.resnet101(weights=weights)
    else:
        raise ValueError("arch must be resnet50 or resnet101")
    in_feat = backbone.fc.in_features
    backbone.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_feat, num_classes))
    return backbone, weights


# -----------------------------
# 학습/평가 루프 (tqdm progress bar 포함)
# -----------------------------
@torch.no_grad()
def accuracy_topk(output, target, k=1):
    k = min(k, output.size(1))  # 안전장치
    topk = output.topk(k, dim=1).indices
    return topk.eq(target.view(-1,1)).any(dim=1).float().mean().item() * 100.0

def run_one_epoch(model, loader, criterion, optimizer, device, use_amp=False, train=True, desc: Optional[str]=None):
    model.train() if train else model.eval()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total = 0
    running_loss = 0.0
    correct1 = 0
    correct3 = 0

    y_true_list, y_pred_list = [], []

    iterator = loader
    if _HAS_TQDM:
        iterator = tqdm(loader, total=len(loader), desc=desc or ("Train" if train else "Eval"), leave=False, ncols=100)

    for x, y in iterator:
        x, y = x.to(device), y.to(device)
        bs = x.size(0)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                logits = model(x)
                loss = criterion(logits, y)

        running_loss += loss.item() * bs
        total += bs

        with torch.no_grad():
            pred1 = logits.argmax(1)
            correct1 += (pred1 == y).sum().item()
            k = min(3, logits.size(1))
            if k > 1:
                topk = logits.topk(k, dim=1).indices
                correct3 += topk.eq(y.view(-1,1)).any(dim=1).float().sum().item()
            else:
                correct3 += (pred1 == y).sum().item()
            # F1 계산을 위해 저장
            y_true_list.append(y.detach().cpu())
            y_pred_list.append(pred1.detach().cpu())

        if _HAS_TQDM:
            postfix = {
                "loss": f"{running_loss/max(1,total):.4f}",
                "acc1": f"{(correct1/max(1,total))*100:.2f}%",
            }
            if not train:
                postfix["acc3"] = f"{(correct3/max(1,total))*100:.2f}%"
            iterator.set_postfix(postfix)

    epoch_loss = running_loss / max(1, total)
    acc_top1 = (correct1 / max(1, total)) * 100.0
    acc_top3 = (correct3 / max(1, total)) * 100.0 if not train else None

    y_true = torch.cat(y_true_list).numpy() if y_true_list else np.array([])
    y_pred = torch.cat(y_pred_list).numpy() if y_pred_list else np.array([])

    metrics = {"loss": epoch_loss, "acc_top1": float(acc_top1)}
    if acc_top3 is not None:
        metrics["acc_top3"] = float(acc_top3)
    if _HAS_SKLEARN and y_true.size and y_pred.size:
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
    return metrics, (y_true, y_pred)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="ResNet Baseline (TS_/VS_ prefix aware)")
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--out_dir", default="./runs/resnet_baseline", type=str)
    ap.add_argument("--image_size", default=256, type=int)
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--dropout", default=0.5, type=float)
    ap.add_argument("--num_workers", default=2, type=int)
    ap.add_argument("--arch", default="resnet101", choices=["resnet50","resnet101"])

    # ---- Augmentation options ----
    ap.add_argument("--aug_scale_min", default=0.7, type=float,
                    help="최소 확대 비율 (RandomResizedCrop scale=(min,1.0))")
    ap.add_argument("--aug_color", nargs=4, type=float, default=[0.1, 0.1, 0.1, 0.05],
                    metavar=("BRIGHTNESS","CONTRAST","SATURATION","HUE"),
                    help="ColorJitter 강도 (색 왜곡은 과하지 않게 기본값 권장)")
    ap.add_argument("--aug_flip_p", default=0.0, type=float,
                    help="수평 뒤집기 확률 (기본 0.0 = 미사용)")
    ap.add_argument("--val_resize_ratio", default=1.15, type=float,
                    help="검증 시 Resize 배율 (CenterCrop 전)")

    ap.add_argument("--seed", default=42, type=int)

    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    print(f"[Device] {device_pretty(device)}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 정규화 통계를 위해 weights만 먼저 선택 (모델은 나중에 빌드)
    if args.arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    else:
        weights = models.ResNet101_Weights.IMAGENET1K_V2
    mean, std = get_mean_std_from_weights(weights)

    data_root = Path(args.data_root).expanduser()
    if not data_root.exists():
        raise SystemExit(f"[!] data_root가 존재하지 않습니다: {data_root}")

    # 2) 증강 설정
    aug_cfg = {
        "scale_min": args.aug_scale_min,
        "color": tuple(args.aug_color),
        "flip_p": args.aug_flip_p,
        "val_resize_ratio": args.val_resize_ratio,
    }
    print(
        f"[Augment] scale_min={aug_cfg['scale_min']} color={aug_cfg['color']} "
        f"flip_p={aug_cfg['flip_p']} val_resize_ratio={aug_cfg['val_resize_ratio']}"
    )

    # 3) 데이터셋 구성
    train_ds = SkinSplitDataset(data_root, "training",   args.image_size, mean, std, train=True,  aug_cfg=aug_cfg)
    try:
        val_ds   = SkinSplitDataset(data_root, "validation", args.image_size, mean, std, train=False, aug_cfg=aug_cfg)
    except FileNotFoundError:
        try:
            val_ds = SkinSplitDataset(data_root, "valid", args.image_size, mean, std, train=False, aug_cfg=aug_cfg)
        except FileNotFoundError:
            print("[Warn] validation split not found. Using TRAIN as VALID (임시 대체).")
            val_ds = train_ds  # ⚠️ 연구/임시용

    try:
        test_ds  = SkinSplitDataset(data_root, "test",       args.image_size, mean, std, train=False, aug_cfg=aug_cfg)
    except FileNotFoundError:
        test_ds  = val_ds

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"[Classes] ({num_classes}): {class_names}")

    # 4) 로더
    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin,
        generator=gen, worker_init_fn=worker_init_fn
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        generator=gen, worker_init_fn=worker_init_fn
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        generator=gen, worker_init_fn=worker_init_fn
    )

    # 5) 모델 빌드(클래스 수 확정 후)
    model, _ = build_model(num_classes=num_classes, dropout=args.dropout, arch=args.arch)
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    use_amp = (device.type == "cuda")

    history = []
    best_val = float("inf")
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    start = time.time()
    for epoch in range(1, args.epochs+1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        tr, _ = run_one_epoch(model, train_loader, criterion, optimizer, device, use_amp, train=True,  desc=f"Train {epoch}/{args.epochs}")
        va, _ = run_one_epoch(model, val_loader,   criterion, optimizer, device, use_amp=False, train=False, desc=f"Val   {epoch}/{args.epochs}")
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
    te, (y_true, y_pred) = run_one_epoch(model, test_loader, criterion, optimizer, device, use_amp=False, train=False, desc="Test")
    print(json.dumps({"test": te}, ensure_ascii=False, indent=2))

    # 리포트 저장
    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    with open(out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    if _HAS_SKLEARN and y_true.size and y_pred.size:
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        cm = confusion_matrix(y_true, y_pred).tolist()
        with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        with open(out_dir / "confusion_matrix.json", "w", encoding="utf-8") as f:
            json.dump({"labels": class_names, "cm": cm}, f, ensure_ascii=False, indent=2)
        print("\n[Classification Report]\n" + report)

    # -----------------------------
    # 결과 요약 출력 & 저장
    # -----------------------------
    total_time = human_time(time.time()-start)
    summary_txt = []
    summary_txt.append("모델 학습 완료 ✅")
    summary_txt.append(f"- 모델 아키텍처      : {args.arch}")
    summary_txt.append(f"- 클래스 수          : {num_classes}")
    summary_txt.append(f"- 총 파라미터 수     : {total_params:,} (학습 가능한: {trainable_params:,})")
    summary_txt.append(f"- 학습 샘플 수       : {len(train_ds):,}")
    summary_txt.append(f"- 검증 샘플 수       : {len(val_ds):,}")
    summary_txt.append(f"- 테스트 샘플 수     : {len(test_ds):,}")
    summary_txt.append(f"- 사용 디바이스      : {device_pretty(device)}")
    summary_txt.append(f"- 총 학습 시간       : {total_time}")

    print("\n[Summary]\n" + "\n".join(summary_txt))

    summary_json = {
        "status": "모델 학습 완료",
        "arch": args.arch,
        "num_classes": num_classes,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
        "num_test_samples": len(test_ds),
        "device": device_pretty(device),
        "total_time": total_time,
        "best_checkpoint": str(out_dir / "best.pt"),
        "last_checkpoint": str(out_dir / "last.pt"),
        "final_metrics": {"val": history[-1] if history else None, "test": te},
    }
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_txt) + "\n")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] Total time: {total_time}")
    print(f"Best checkpoint: {out_dir / 'best.pt'}")
    print(f"Last checkpoint: {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
