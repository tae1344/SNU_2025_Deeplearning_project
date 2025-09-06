import os
import json
from PIL import Image
from common.config import CLASS_NAMES, TEST_IMAGE_DIR, TEST_LABEL_DIR

# 각 카테고리별 진단명과 설명 매핑
DIAGNOSIS_INFO = {
    "광선각화증": "지속적인 자외선 노출에 의해 발생한 인설을 동반한 붉은 반점이나 구진 형태의 병변으로 편평세포암으로 발전할 수 있음",
    "기저세포암": "두경부에 호발하는 가장 흔한 피부암으로 진단을 위해 조직검사가 필요함",
    "멜라닌세포모반": "멜라닌세포로 이루어진 모반으로 주위 정상 피부와 명확한 경계를 보임",
    "보웬병": "피부에 발생하는 상피내암의 일종으로 진단을 위해 조직검사가 필요함",
    "비립종": "피부 표면 가까이에 위치한 1mm 내외의 흰색 구진으로 각질에 차 있는 낭종임",
    "사마귀": "사람 유두종 바이러스의 감염에 의해 발생한 표면이 오톨도톨한 구진으로 접촉에 의해 전염됨",
    "악성흑색종": "멜라닌세포의 악성 종양으로 진단을 위해 조직검사가 필요함",
    "지루각화증": "갈색이나 검정색의 구진 혹은 판으로 검버섯이라고도 불림",
    "편평세포암": "얼굴과 손등, 팔과 같은 자외선 노출부위에 호발하는 피부암의 일종으로 진단을 위해 조직검사가 필요함",
    "표피낭종": "피부색을 띄며 천천히 자라는 증상이 없는 중양. 때때로 염증이 생기며 붉고 압통이 발생하기도 함",
    "피부섬유종": "성인의 다리에 호발하는 갈색이나 홍갈색의 단단한 구진 양상의 양성 종양",
    "피지샘증식증": "중년 이후 얼굴에 호발하는 피지샘의 증식에 의해 발생한 융기된 병변",
    "혈관종": "모세혈관 증식에 의해 발생한 선홍색 융기된 병변으로 병변이 깊은 경우 푸르스름한 혹의 형태로 나타나기도 함",
    "화농 육아종": "모세혈관의 증식에 의해 발생한 0.5-2cm 가량의 붉은 돌기로 부딪히거나 긁혔을 때 쉽게 출혈됨",
    "흑색점": "얼굴과 손등, 팔과 같은 자외선 노출 부위에 나타나는 갈색 반점과 반",
}


def get_image_dimensions(image_path):
    """이미지의 크기를 가져옵니다."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"이미지 크기 확인 실패 {image_path}: {e}")
        return (512, 512)  # 기본값


def create_json_label(image_filename, category_name):
    """이미지 파일명을 기반으로 JSON 라벨을 생성합니다."""
    # 파일명에서 확장자 제거
    identifier = os.path.splitext(image_filename)[0]

    # 이미지 경로
    image_path = os.path.join(category_name, image_filename)

    # 이미지 크기 확인
    full_image_path = os.path.join(TEST_IMAGE_DIR, category_name, image_filename)
    width, height = get_image_dimensions(full_image_path)

    # JSON 구조 생성
    json_data = {
        "annotations": [
            {
                "identifier": identifier,
                "diagnosis_info": {
                    "diagnosis_name": category_name,
                    "onset": "N/A",
                    "distribution": "N/A",
                    "bodypart": "N/A",
                    "symptom": "N/A",
                    "desc": DIAGNOSIS_INFO.get(category_name, "진단 정보 없음"),
                },
                "generated_parameters": {
                    "gender": "N/A",
                    "age_range": "N/A",
                    "bodypart": "N/A",
                    "race": "Asian",
                },
                "photograph": {
                    "file_path": image_path,
                    "width": width,
                    "height": height,
                },
                "bbox": {
                    "xpos": 0,
                    "ypos": 0,
                    "file_path": image_path,
                    "width": width,
                    "height": height,
                },
            }
        ]
    }

    return json_data


def generate_missing_labels():
    """누락된 라벨링 파일들을 생성합니다."""
    print("테스트 데이터의 누락된 라벨링 파일을 생성합니다...")

    total_created = 0

    # 각 카테고리별로 처리
    for category in CLASS_NAMES:
        print(f"\n처리 중인 카테고리: {category}")

        # 원천데이터 디렉토리 경로
        source_dir = os.path.join(TEST_IMAGE_DIR, category)
        # 라벨링데이터 디렉토리 경로
        label_dir = os.path.join(TEST_LABEL_DIR, category)

        # 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
            print(f"  라벨링 디렉토리 생성: {label_dir}")

        # 원천데이터 디렉토리가 존재하는지 확인
        if not os.path.exists(source_dir):
            print(f"  원천데이터 디렉토리가 존재하지 않습니다: {source_dir}")
            continue

        # 이미지 파일 목록 가져오기
        image_files = [
            f
            for f in os.listdir(source_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_files:
            print(f"  이미지 파일이 없습니다: {source_dir}")
            continue

        print(f"  이미지 파일 수: {len(image_files)}")

        # 각 이미지 파일에 대해 JSON 라벨 생성
        created_count = 0
        for image_file in image_files:
            # JSON 파일명 생성 (이미지 파일명에서 확장자를 .json으로 변경)
            json_filename = os.path.splitext(image_file)[0] + ".json"
            json_path = os.path.join(label_dir, json_filename)

            # 이미 JSON 파일이 존재하는지 확인
            if os.path.exists(json_path):
                continue

            # JSON 라벨 생성
            json_data = create_json_label(image_file, category)

            # JSON 파일 저장
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                created_count += 1
                total_created += 1
            except Exception as e:
                print(f"    JSON 파일 저장 실패 {json_path}: {e}")

        print(f"  생성된 라벨링 파일 수: {created_count}")

    print(f"\n총 생성된 라벨링 파일 수: {total_created}")
    print("라벨링 파일 생성이 완료되었습니다!")


def check_existing_labels():
    """기존 라벨링 파일과 이미지 파일의 매칭 상태를 확인합니다."""
    print("기존 라벨링 파일과 이미지 파일의 매칭 상태를 확인합니다...")

    for category in CLASS_NAMES:
        source_dir = os.path.join(TEST_IMAGE_DIR, category)
        label_dir = os.path.join(TEST_LABEL_DIR, category)

        if not os.path.exists(source_dir):
            continue

        # 이미지 파일 목록
        image_files = [
            f
            for f in os.listdir(source_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        image_identifiers = {os.path.splitext(f)[0] for f in image_files}

        # 라벨링 파일 목록
        if os.path.exists(label_dir):
            label_files = [f for f in os.listdir(label_dir) if f.endswith(".json")]
            label_identifiers = {os.path.splitext(f)[0] for f in label_files}
        else:
            label_identifiers = set()

        # 매칭되지 않는 이미지 파일들
        missing_labels = image_identifiers - label_identifiers

        print(f"\n{category}:")
        print(f"  이미지 파일 수: {len(image_files)}")
        print(f"  라벨링 파일 수: {len(label_identifiers)}")
        print(f"  누락된 라벨링 파일 수: {len(missing_labels)}")

        if missing_labels:
            print(
                f"  누락된 파일들: {list(missing_labels)[:5]}{'...' if len(missing_labels) > 5 else ''}"
            )


if __name__ == "__main__":
    # 기존 상태 확인
    check_existing_labels()

    # 사용자 확인
    response = input("\n누락된 라벨링 파일을 생성하시겠습니까? (y/n): ")
    if response.lower() in ["y", "yes", "예"]:
        generate_missing_labels()
    else:
        print("라벨링 파일 생성을 취소했습니다.")
