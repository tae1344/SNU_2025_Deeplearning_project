import os
import json
from PIL import Image
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, label_folder, image_folder, transform=None):
        self.label_folder = label_folder
        self.image_folder = image_folder
        self.transform = transform
        self.identifiers = []  # 각 데이터의 identifier
        self.images = []  # 이미지 데이터
        self.labels = []  # 라벨
        self.label_counts = Counter()  # 클래스별 데이터 개수
        # 진단명 -> 라벨 매핑
        self.diagnosis_to_label = {
            "광선각화증": 0,
            "기저세포암": 1,
            "멜라닌세포모반": 2,
            "보웬병": 3,
            "비립종": 4,
            "사마귀": 5,
            "악성흑색종": 6,
            "지루각화증": 7,
            "편평세포암": 8,
            "표피낭종": 9,
            "피부섬유종": 10,
            "피지샘증식증": 11,
            "혈관종": 12,
            "화농 육아종": 13,
            "흑색점": 14
        }

        self._load_data()

    def _load_data(self):
        """
        JSON 파일에서 데이터를 로드하고 이미지 및 라벨을 저장
        """

        all_json_files = []

        # os.walk(디렉토리 읽기, top-down or bottom-up) return (dirpath, dirnames, filenames)
        for root, _, files in os.walk(self.label_folder):
            for json_file in files:
                if json_file.endswith(".json"):
                    all_json_files.append((root, json_file))

        # JSON 파일 순회
        for root, json_file in tqdm(all_json_files, desc="Loading data", unit="file"):
            json_file_path = os.path.join(root, json_file)
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for annotation in data['annotations']:
                    diagnosis_name = annotation['diagnosis_info']['diagnosis_name']
                    identifier = annotation['identifier']  # identifier 가져오기

                    # 고정된 매핑에서 해당 진단명을 탐색
                    if diagnosis_name not in self.diagnosis_to_label:
                        print(f"Unknown diagnosis found: {diagnosis_name}. Skipping this entry.")
                        continue
                    
                    label = self.diagnosis_to_label[diagnosis_name]

                    img_path = os.path.join(self.image_folder, annotation['bbox']['file_path'])

                    try:
                        image = Image.open(img_path).convert("RGB")

                        if self.transform:
                            image = self.transform(image)

                        self.images.append(image)
                        self.identifiers.append(identifier)  # identifier 저장
                        self.labels.append(label)
                        self.label_counts[diagnosis_name] += 1  # 클래스별 개수 추가
                    except (FileNotFoundError, OSError) as e:
                        print(f"이미지를 읽을 수 없습니다: {img_path}. 오류: {e}")
                        continue

    # override
    def __len__(self):
        return len(self.images)

    # override
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def get_label_mappings(self):
        """
        클래스 번호와 진단명을 매핑하여 반환
        """
        return self.diagnosis_to_label

    def get_label_counts(self):
        """
        각 클래스별 데이터 개수를 반환
        """
        return self.label_counts
    