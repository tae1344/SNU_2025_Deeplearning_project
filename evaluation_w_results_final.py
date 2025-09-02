import os
import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
from config import BATCH_SIZE, TEST_LABEL_DIR, TEST_IMAGE_DIR, CLASS_NAMES, TRANSFORM, DEVICE, LOG_DIR, MODEL_DIR, NUM_CLASSES, DROPOUT_RATE, set_korean_font
from logger import Logger
from model import CustomResNet101

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import time
import numpy as np
import pandas as pd

def calculate_top_k_accuracy(outputs, labels, k=3):

    _, top_k_preds = outputs.topk(k, dim=1)
    correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds))
    return correct.sum().item()

def evaluate_all_checkpoints():

    set_korean_font()

    # Font added
    font_path = "/root/.fonts/NanumGothic.ttf"
    fontprop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = fontprop.get_name()

    # Logger 설정
    logger = Logger(LOG_DIR, "evaluate_log.json")

    # 평가 시작 시간
    start_time = time.time()
    logger.log({"message": f"Evaluation started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"})

    # 데이터 로드
    test_dataset = CustomDataset(
        label_folder=TEST_LABEL_DIR,
        image_folder=TEST_IMAGE_DIR,
        transform=TRANSFORM
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 체크포인트 파일 리스트 (epoch 1~10)
    checkpoint_files = [f"checkpoint_epoch_{i}.pth" for i in range(1, 11)]

    for ckpt_file in checkpoint_files:
        print(f"\nEvaluating checkpoint: {ckpt_file}")
        checkpoint_path = os.path.join(LOG_DIR, ckpt_file)

        # 모델 로드 (CPU 환경용)
        model = CustomResNet101(NUM_CLASSES).to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        # 평가 단계 초기화
        test_labels, test_preds = [], []
        test_probs = []
        top3_correct = 0
        total_samples = 0
        all_results = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                test_labels.extend(labels.cpu().numpy())
                test_preds.extend(preds.cpu().numpy())
                test_probs.extend(probs.cpu().numpy())

                top3_correct += calculate_top_k_accuracy(outputs, labels, k=3)
                total_samples += labels.size(0)

                for i in range(len(labels)):
                    result = {
                        'image_id': test_dataset.identifiers[i],
                        'actual': CLASS_NAMES[labels[i].item()],
                        'predicted': CLASS_NAMES[preds[i].item()]
                    }
                    all_results.append(result)

        # 평가 결과 계산
        cm = confusion_matrix(test_labels, test_preds)
        report = classification_report(test_labels, test_preds, target_names=CLASS_NAMES, output_dict=True)
        top1_accuracy = np.trace(cm) / np.sum(cm)
        top3_accuracy = top3_correct / total_samples

        test_labels_bin = label_binarize(test_labels, classes=range(NUM_CLASSES))
        test_probs = np.array(test_probs)
        roc_auc_ovr = roc_auc_score(test_labels_bin, test_probs, average='macro', multi_class='ovr')

        # Confusion Matrix 분석 (TP, FP, TN, FN)
        tp_per_class = np.diag(cm)
        fp_per_class = cm.sum(axis=0) - tp_per_class
        fn_per_class = cm.sum(axis=1) - tp_per_class
        tn_per_class = cm.sum() - (tp_per_class + fp_per_class + fn_per_class)

        # 클래스별 결과 출력
        for idx, class_name in enumerate(CLASS_NAMES):
            logger.log({
                "checkpoint": ckpt_file,
                "class": class_name,
                "TP": int(tp_per_class[idx]),
                "FP": int(fp_per_class[idx]),
                "TN": int(tn_per_class[idx]),
                "FN": int(fn_per_class[idx])
            })
            print(f"Class: {class_name}, TP: {tp_per_class[idx]}, FP: {fp_per_class[idx]}, "
                  f"TN: {tn_per_class[idx]}, FN: {fn_per_class[idx]}")

        # 로그 기록
        logger.log({
            "checkpoint": ckpt_file,
            "top1_accuracy": top1_accuracy,
            "top3_accuracy": top3_accuracy,
            "roc_auc_ovr": roc_auc_ovr,
            "classification_report": report
        })

        # Confusion Matrix 이미지로 저장
        cm_image_path = os.path.join(LOG_DIR, f'confusion_matrix_rotated_{ckpt_file}.png')

        # Confusion Matrix 표시 및 저장
        fig, ax = plt.subplots(figsize=(12, 12))  # 그래프 크기 조정
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')

        #plt.xticks(rotation=45, ha='right', fontsize=10)  # 글꼴 크기 및 정렬 설정
        #plt.yticks(fontsize=10)

        for label in ax.get_xticklabels():
            label.set_fontproperties(fontprop)
            label.set_rotation(45)
            label.set_ha('right')
            label.set_fontsize(10)

        for label in ax.get_yticklabels():
            label.set_fontproperties(fontprop)
            label.set_fontsize(10)
        plt.title(f'Confusion Matrix - {ckpt_file}', fontproperties=fontprop, fontsize=16)

        # 저장 및 종료
        plt.tight_layout()
        plt.savefig(cm_image_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion Matrix saved to {cm_image_path}")

        # CSV 파일로 예측값과 실제값 저장
        results_df = pd.DataFrame(all_results)
        results_csv_path = os.path.join(LOG_DIR, f'prediction_results_{ckpt_file}.csv')
        results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')

        # 상세한 성능 지표 출력
        print(f"Checkpoint {ckpt_file} evaluated.")
        print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
        print(f"Top-3 Accuracy: {top3_accuracy:.4f}")
        print(f"ROC-AUC (One-vs-Rest): {roc_auc_ovr:.4f}")
        print(f"Macro F-1: {report['macro avg']['f1-score']:.4f}")
        print(f"Macro Recall: {report['macro avg']['recall']:.4f}")
        print(f"Prediction results saved to: {results_csv_path}")

    # 평가 종료 시간
    end_time = time.time()
    duration = end_time - start_time
    logger.log({"message": f"Evaluation completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"})
    logger.log({"message": f"Total evaluation time: {duration:.2f} seconds"})
    logger.save()

    print(f"\nAll checkpoints evaluation completed. Total time: {duration:.2f} seconds")


if __name__ == "__main__":
    print("Starting evaluation...")
    evaluate_all_checkpoints()
    print("Evaluation complete.")
