import random
import torch
import torch.nn as nn
import numpy as np
import json, os, re
from sklearn.metrics import f1_score, roc_auc_score


# -----------------------------
# 공통 유틸
# -----------------------------
def get_mean_std_from_weights(weights):
    """torchvision weights에서 mean/std 추출 (실패 시 ImageNet 기본값)"""
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
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
        std = meta.get("std", default_std)
        return list(mean), list(std)
    except Exception:
        return default_mean, default_std


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("set_seed", seed)


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
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def worker_init_fn(worker_id: int):
    """
    각 워커 프로세스에서 동일한 시드 설정
    """
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def save_checkpoint(state, checkpoint_dir, filename):
    """
    모델 체크포인트 저장
    Args:
        state (dict): 모델 상태 딕셔너리 (state_dict, optimizer 등 포함).
        checkpoint_dir (str): 체크포인트 저장 디렉토리.
        filename (str): 저장할 파일 이름.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved at {filepath}")


def calculate_f1_score(outputs, labels):
    """
    F1 Score 계산 (macro-averaged)
    Args:
        outputs (torch.Tensor): 모델 출력값 (logits).
        labels (torch.Tensor): 실제 레이블.
    Returns:
        float: Macro-averaged F1 score.
    """
    _, preds = torch.max(outputs, 1)
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    return f1_score(labels_np, preds_np, average="macro")


def calculate_auroc(outputs, labels, num_classes=15):
    """
    멀티클래스 AUROC 계산 (macro-averaged)
    """
    outputs_np = outputs.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # One-vs-rest AUROC 계산
    auroc_scores = []
    for i in range(num_classes):
        try:
            # i번째 클래스에 대한 이진 분류로 변환
            binary_labels = (labels_np == i).astype(int)
            binary_outputs = outputs_np[:, i]

            if len(np.unique(binary_labels)) > 1:  # 클래스가 하나만 있으면 계산 불가
                auroc = roc_auc_score(binary_labels, binary_outputs)
                auroc_scores.append(auroc)
        except:
            continue

    return np.mean(auroc_scores) if auroc_scores else 0.0


def save_auroc_data(outputs, labels, epoch, phase, save_dir):
    """
    AUROC 계산에 필요한 값을 JSON 파일로 저장
    Args:
        outputs (list): 모델 출력값 (2차원 리스트: [batch_size, num_classes]).
        labels (list): 실제 레이블 리스트 (1차원: [batch_size]).
        epoch (int): 현재 에포크.
        phase (str): "train" 또는 "val".
        save_dir (str): JSON 파일 저장 경로.
    """
    os.makedirs(save_dir, exist_ok=True)

    # outputs는 [batch_size, num_classes] 형태의 2차원 리스트
    # 각 샘플의 outputs를 4자리 소수점으로 반올림
    processed_outputs = []
    for sample_outputs in outputs:
        processed_outputs.append([round(float(o), 4) for o in sample_outputs])

    data = {
        "epoch": epoch,
        "phase": phase,
        "outputs": processed_outputs,  # 2차원 리스트 그대로 저장
        "labels": labels,  # 1차원 리스트
    }
    file_path = os.path.join(save_dir, f"{phase}_auroc_data_epoch_{epoch}.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"AUROC data saved for {phase} at epoch {epoch} to {file_path}")


def calculate_top_k_accuracy(outputs, labels, k=3):
    """
    Top-3 정확도
    """
    _, top_k_preds = outputs.topk(k, dim=1)
    correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds))
    return correct.sum().item()


# -----------------------------
# 파일경로 정규화 유틸
# -----------------------------
def _normalize_split_segment(segment: str) -> str:
    """Normalize dataset split folder names to canonical ones.

    Examples:
    - "train", "Train", "training", "Training" -> "training"
    - "val", "valid", "validation"            -> "validation"
    - "test", "Test", "testing"                -> "test"
    """
    lower = segment.lower()
    if lower in {"train", "training"}:
        return "training"
    if lower in {"val", "valid", "validation"}:
        return "validation"
    if lower in {"test", "testing"}:
        return "test"
    return segment


_KOR_CANONICALS = {
    "원천데이터": re.compile(r"^\s*(?:\d+\.)?\s*원천데이터\s*$"),
    # Allow future extension if needed, e.g. 라벨링데이터 prefixing like "02. 라벨링데이터"
    "라벨링데이터": re.compile(r"^\s*(?:\d+\.)?\s*라벨링데이터\s*$"),
}


def _normalize_korean_segment(segment: str) -> str:
    """Normalize Korean dataset folder names with numeric prefixes like '01. 원천데이터'."""
    for canonical, pattern in _KOR_CANONICALS.items():
        if pattern.match(segment):
            return canonical
    return segment


def normalize_dataset_path(path: str) -> str:
    """Normalize a dataset directory path by fixing common variations.

    - Unify split folder names (training/validation/test)
    - Strip numeric prefixes like '01.' and spaces before Korean folder names
      such as '원천데이터', '라벨링데이터'
    """
    if not path:
        return path

    parts = []
    for seg in path.split(os.sep):
        seg = _normalize_split_segment(seg)
        seg = _normalize_korean_segment(seg)
        parts.append(seg)
    return os.sep.join(parts)
