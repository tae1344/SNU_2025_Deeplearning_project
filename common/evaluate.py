from common.utils import (
    calculate_top_k_accuracy,
    calculate_f1_score,
    calculate_auroc,
)
from tqdm.auto import tqdm
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from common.config import CLASS_NAMES
import os
from common.config import LOG_DIR


def evaluate_test_set(model, test_loader, criterion, device):
    """
    Test set으로 최종 모델 성능 평가
    모든 모델에서 공통으로 사용
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_top3_correct = 0
    test_total = 0

    test_outputs = []
    test_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()
            test_top3_correct += calculate_top_k_accuracy(outputs, labels, k=3)

            test_outputs.extend(outputs.cpu().tolist())
            test_labels.extend(labels.cpu().tolist())
            all_predictions.extend(preds.cpu().tolist())

    # 메트릭 계산
    test_loss /= len(test_loader)
    test_accuracy = test_correct / test_total
    test_top3_accuracy = test_top3_correct / test_total
    test_f1_score = calculate_f1_score(
        torch.tensor(test_outputs), torch.tensor(test_labels)
    )
    test_auroc = calculate_auroc(torch.tensor(test_outputs), torch.tensor(test_labels))

    # 결과 딕셔너리 반환
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_top3_accuracy": test_top3_accuracy,
        "test_f1_score": test_f1_score,
        "test_auroc": test_auroc,
        "test_outputs": test_outputs,
        "test_labels": test_labels,
        "test_predictions": all_predictions,
    }

    return results


def print_test_results(results, model_name="Model"):
    """Test 결과를 보기 좋게 출력"""
    print(f"\n{'='*60}")
    print(f"{model_name} - Final Test Results")
    print(f"{'='*60}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Accuracy (Top-1): {results['test_accuracy']:.4f}")
    print(f"Test Accuracy (Top-3): {results['test_top3_accuracy']:.4f}")
    print(f"Test F1-Score: {results['test_f1_score']:.4f}")
    print(f"Test AUROC: {results['test_auroc']:.4f}")
    print(f"{'='*60}")

    # 혼동 행렬 시각화
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(results["test_labels"], results["test_predictions"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(LOG_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()  # 이미지를 화면에 표시
    plt.close()
