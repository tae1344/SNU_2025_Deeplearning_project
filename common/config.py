import os
from torchvision import transforms
import torch
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 경로 설정
BASE_DIR = os.getcwd()  # 프로젝트 루트 디렉토리 경로

# 학습, 검증, 테스트 데이터 경로
TRAIN_LABEL_DIR = os.path.join(
    BASE_DIR, "data/train", "라벨링데이터"
)  # 학습 라벨 데이터 경로
TRAIN_IMAGE_DIR = os.path.join(
    BASE_DIR, "data/train", "원천데이터"
)  # 학습 이미지 데이터 경로

VAL_LABEL_DIR = os.path.join(
    BASE_DIR, "data/validation", "라벨링데이터"
)  # 검증 라벨 데이터 경로
VAL_IMAGE_DIR = os.path.join(
    BASE_DIR, "data/validation", "원천데이터"
)  # 검증 이미지 데이터 경로

TEST_LABEL_DIR = os.path.join(
    BASE_DIR, "data/test", "라벨링데이터"
)  # 테스트 라벨 데이터 경로
TEST_IMAGE_DIR = os.path.join(
    BASE_DIR, "data/test", "원천데이터"
)  # 테스트 이미지 데이터 경로

# 로그 및 모델 저장 경로
LOG_DIR = os.path.join(BASE_DIR, "logs")
TRAIN_LOG_DIR = os.path.join(BASE_DIR, "logs/train_logs")
TEST_LOG_DIR = os.path.join(BASE_DIR, "logs/test_logs")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# 모델 파라미터
NUM_CLASSES = 15  # 클래스 수
IMG_SIZE = 224  # 입력 이미지의 해상도(256x256 픽셀), 마지막 10~20ep에 320으로 리핏 권장
DROPOUT_RATE = 0.5  # 드롭아웃 비율
BATCH_SIZE = 32  # 배치 크기
EPOCHS = 10  # 기본 에포크 수
BASE_LEARNING_RATE = 5e-4  # 기본 learing rate(0.0005)
WD = 0.05  # 가중치 감쇠(weight decay)(0.05) - 과적화 방지를 위한 정규화 기법
WARMUP_EPOCHS = 1  # 워밍업 에폭 수 - 학습률을 점진적으로 증가시크는 에폭 수, 학습 초기에 안정적인 학습을 위한 warmup 기법
SEED = 42

WORKER_NUM = 2 if torch.cuda.is_available() else 0
# DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"  # GPU 사용 여부
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 여부

# 클래스 이름 정의 (고정된 순서)
CLASS_NAMES = [
    "광선각화증",
    "기저세포암",
    "멜라닌세포모반",
    "보웬병",
    "비립종",
    "사마귀",
    "악성흑색종",
    "지루각화증",
    "편평세포암",
    "표피낭종",
    "피부섬유종",
    "피지샘증식증",
    "혈관종",
    "화농 육아종",
    "흑색점",
]

mean = [0.485, 0.456, 0.406]  # ImageNet
std = [0.229, 0.224, 0.225]

# Transform 설정
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 이미지 크기 조정
        transforms.ToTensor(),  # 텐서 변환
        transforms.Normalize(mean, std),  # 정규화
    ]
)

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


# 한글 폰트 설정 함수
def setup_korean_font():
    """한글 폰트 설정"""
    try:
        # 시스템 한글 폰트 시도
        korean_fonts = [
            "Malgun Gothic",
            "NanumGothic",
            "AppleGothic",
            "Noto Sans CJK KR",
        ]
        for font in korean_fonts:
            if font in [f.name for f in fm.fontManager.ttflist]:
                plt.rcParams["font.family"] = font
                print(f"한글 폰트 설정 완료 : {font}")
                return True
    except:
        pass

    # 기본 폰트 사용
    plt.rcParams["font.family"] = "DejaVu Sans"
    print("한글 폰트 설정 실패, 기본 폰트 사용 : DejaVu Sans")
    return False


# 한글 폰트 설정 실행
setup_korean_font()
