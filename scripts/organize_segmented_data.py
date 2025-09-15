"""
세그멘테이션된 이미지를 클래스별 폴더로 정리하는 스크립트
"""

import os
import shutil
from pathlib import Path
import json
from tqdm import tqdm


def organize_segmented_data(
    segmented_images_dir,
    segmented_masks_dir,
    original_label_dir,
    output_images_dir,
    output_masks_dir,
):
    """
    세그멘테이션된 이미지를 클래스별로 정리

    Args:
        segmented_images_dir: 세그멘테이션된 이미지 디렉토리
        segmented_masks_dir: 세그멘테이션된 마스크 디렉토리
        original_label_dir: 원본 라벨 디렉토리 (클래스 정보 추출용)
        output_images_dir: 정리된 이미지 출력 디렉토리
        output_masks_dir: 정리된 마스크 출력 디렉토리
    """

    # 출력 디렉토리 생성
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    # 클래스별 폴더 생성
    class_names = [
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

    for class_name in class_names:
        os.makedirs(os.path.join(output_images_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(output_masks_dir, class_name), exist_ok=True)

    # 원본 라벨에서 파일명-클래스 매핑 생성
    print("라벨 파일에서 클래스 정보 추출 중...")
    file_to_class = {}

    for class_name in class_names:
        class_label_dir = os.path.join(original_label_dir, class_name)
        if os.path.exists(class_label_dir):
            for json_file in os.listdir(class_label_dir):
                if json_file.endswith(".json"):
                    # JSON 파일에서 이미지 파일명 추출
                    json_path = os.path.join(class_label_dir, json_file)
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if "image" in data and "filename" in data["image"]:
                                image_filename = data["image"]["filename"]
                                # 확장자 제거하여 매핑
                                base_name = os.path.splitext(image_filename)[0]
                                file_to_class[base_name] = class_name
                    except Exception as e:
                        print(f"JSON 파일 읽기 실패: {json_path}, 오류: {e}")

    print(f"총 {len(file_to_class)}개의 파일-클래스 매핑 생성")

    # 세그멘테이션된 이미지들을 클래스별로 정리
    segmented_images = os.listdir(segmented_images_dir)
    segmented_masks = os.listdir(segmented_masks_dir)

    print(f"세그멘테이션된 이미지: {len(segmented_images)}개")
    print(f"세그멘테이션된 마스크: {len(segmented_masks)}개")

    organized_count = 0
    missing_class_count = 0

    for img_file in tqdm(segmented_images, desc="이미지 정리 중"):
        if not img_file.endswith((".jpg", ".jpeg", ".png")):
            continue

        # 파일명에서 원본 이미지명 추출 (예: ISIC_0034524_processed.jpg -> ISIC_0034524)
        # 먼저 확장자 제거, 그 다음 _processed 제거
        base_name = img_file
        if base_name.endswith((".jpg", ".jpeg", ".png")):
            base_name = os.path.splitext(base_name)[0]  # 확장자 제거
        if base_name.endswith("_processed"):
            base_name = base_name.replace("_processed", "")  # _processed 제거

        # 클래스 찾기
        if base_name in file_to_class:
            class_name = file_to_class[base_name]

            # 이미지 복사
            src_img = os.path.join(segmented_images_dir, img_file)
            dst_img = os.path.join(output_images_dir, class_name, img_file)
            shutil.copy2(src_img, dst_img)

            # 마스크 복사 (해당하는 마스크 파일 찾기)
            mask_file = base_name + "_mask.png"
            if mask_file in segmented_masks:
                src_mask = os.path.join(segmented_masks_dir, mask_file)
                dst_mask = os.path.join(output_masks_dir, class_name, mask_file)
                shutil.copy2(src_mask, dst_mask)

            organized_count += 1
        else:
            missing_class_count += 1
            if missing_class_count <= 5:  # 처음 5개만 출력
                print(f"클래스 정보 없음: {base_name}")

    print(f"\n정리 완료!")
    print(f"정리된 이미지: {organized_count}개")
    print(f"클래스 정보 없는 이미지: {missing_class_count}개")

    # 각 클래스별 이미지 수 확인
    print("\n클래스별 이미지 수:")
    for class_name in class_names:
        class_dir = os.path.join(output_images_dir, class_name)
        if os.path.exists(class_dir):
            count = len(
                [
                    f
                    for f in os.listdir(class_dir)
                    if f.endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            print(f"  {class_name}: {count}개")


def organize_new_test_data(
    segmented_images_dir,
    segmented_masks_dir,
    original_images_dir,
    output_images_dir,
    output_masks_dir,
):
    """new_test 데이터를 위한 특별한 정리 함수 (라벨링 데이터 없음)"""

    # 출력 디렉토리 생성
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    # 클래스별 폴더 생성
    class_names = [
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

    for class_name in class_names:
        os.makedirs(os.path.join(output_images_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(output_masks_dir, class_name), exist_ok=True)

    # 원본 이미지에서 파일명-클래스 매핑 생성
    print("원본 이미지에서 클래스 정보 추출 중...")
    file_to_class = {}

    for class_name in class_names:
        class_image_dir = os.path.join(original_images_dir, class_name)
        if os.path.exists(class_image_dir):
            for img_file in os.listdir(class_image_dir):
                if img_file.endswith((".jpg", ".jpeg", ".png")):
                    base_name = os.path.splitext(img_file)[0]
                    file_to_class[base_name] = class_name

    print(f"총 {len(file_to_class)}개의 파일-클래스 매핑 생성")

    # 세그멘테이션된 이미지들을 클래스별로 정리
    segmented_images = os.listdir(segmented_images_dir)
    segmented_masks = os.listdir(segmented_masks_dir)

    print(f"세그멘테이션된 이미지: {len(segmented_images)}개")
    print(f"세그멘테이션된 마스크: {len(segmented_masks)}개")

    organized_count = 0
    missing_class_count = 0

    for img_file in tqdm(segmented_images, desc="이미지 정리 중"):
        if not img_file.endswith((".jpg", ".jpeg", ".png")):
            continue

        # 파일명에서 원본 이미지명 추출
        base_name = img_file
        if base_name.endswith((".jpg", ".jpeg", ".png")):
            base_name = os.path.splitext(base_name)[0]
        if base_name.endswith("_processed"):
            base_name = base_name.replace("_processed", "")

        # 클래스 찾기
        if base_name in file_to_class:
            class_name = file_to_class[base_name]

            # 이미지 복사
            src_img = os.path.join(segmented_images_dir, img_file)
            dst_img = os.path.join(output_images_dir, class_name, img_file)
            shutil.copy2(src_img, dst_img)

            # 마스크 복사
            mask_file = base_name + "_mask.png"
            if mask_file in segmented_masks:
                src_mask = os.path.join(segmented_masks_dir, mask_file)
                dst_mask = os.path.join(output_masks_dir, class_name, mask_file)
                shutil.copy2(src_mask, dst_mask)

            organized_count += 1
        else:
            missing_class_count += 1
            if missing_class_count <= 5:
                print(f"클래스 정보 없음: {base_name}")

    print(f"\n정리 완료!")
    print(f"정리된 이미지: {organized_count}개")
    print(f"클래스 정보 없는 이미지: {missing_class_count}개")

    # 각 클래스별 이미지 수 확인
    print("\n클래스별 이미지 수:")
    for class_name in class_names:
        class_dir = os.path.join(output_images_dir, class_name)
        if os.path.exists(class_dir):
            count = len(
                [
                    f
                    for f in os.listdir(class_dir)
                    if f.endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            print(f"  {class_name}: {count}개")


def main():
    # 데이터 경로 설정
    base_dir = os.getcwd()

    # 각 split에 대해 정리
    splits = ["train", "validation", "test", "new_test"]

    for split in splits:
        print(f"\n=== {split.upper()} 데이터 정리 ===")

        segmented_images_dir = f"{base_dir}/data/processed_{split}_segmented/images"
        segmented_masks_dir = f"{base_dir}/data/processed_{split}_segmented/masks"
        original_label_dir = f"{base_dir}/data/{split}/라벨링데이터"
        original_images_dir = f"{base_dir}/data/{split}/원천데이터"
        output_images_dir = (
            f"{base_dir}/data/processed_{split}_segmented_organized/images"
        )
        output_masks_dir = (
            f"{base_dir}/data/processed_{split}_segmented_organized/masks"
        )

        # 디렉토리 존재 확인
        if not os.path.exists(segmented_images_dir):
            print(
                f"❌ 세그멘테이션된 이미지 디렉토리가 없습니다: {segmented_images_dir}"
            )
            continue

        if split in ("test", "new_test"):
            if not os.path.exists(original_images_dir):
                print(f"❌ 원본 이미지 디렉토리가 없습니다: {original_images_dir}")
                continue

            organize_new_test_data(
                segmented_images_dir,
                segmented_masks_dir,
                original_images_dir,
                output_images_dir,
                output_masks_dir,
            )
        else:
            if not os.path.exists(original_label_dir):
                print(f"❌ 원본 라벨 디렉토리가 없습니다: {original_label_dir}")
                continue

            organize_segmented_data(
                segmented_images_dir,
                segmented_masks_dir,
                original_label_dir,
                output_images_dir,
                output_masks_dir,
            )


if __name__ == "__main__":
    main()
