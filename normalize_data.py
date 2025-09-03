import os
import re
import sys
from typing import Dict, List


# Canonical split names and their synonyms
SPLIT_SYNONYMS: Dict[str, List[str]] = {
    "train": ["train", "training", "Train", "Training"],
    "validation": ["val", "valid", "validation", "Val", "Validation", "VALID"],
    "test": ["test", "testing", "Test", "Testing"],
}


# Canonical Korean folder names with patterns to match variants like "01. 원천데이터"
KOR_CANONICAL_PATTERNS: Dict[str, re.Pattern] = {
    "원천데이터": re.compile(r"^\s*(?:\d+\.)?\s*원천데이터\s*$"),
    "라벨링데이터": re.compile(r"^\s*(?:\d+\.)?\s*라벨링\s*데이터\s*$"),
}


PREFIX_PATTERN = re.compile(r"^(?:TS_|VS_|TE_)+", re.IGNORECASE)


def find_first_matching_name(existing_names: List[str], synonyms: List[str]) -> str:
    lower_to_actual = {name.lower(): name for name in existing_names}
    for s in synonyms:
        if s.lower() in lower_to_actual:
            return lower_to_actual[s.lower()]
    return ""


def safe_rename(src_path: str, dst_path: str) -> str:
    """Rename src->dst avoiding collisions by appending numeric suffixes."""
    if src_path == dst_path:
        return dst_path
    base, ext = os.path.splitext(dst_path)
    candidate = dst_path
    counter = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{counter}{ext}"
        counter += 1
    os.rename(src_path, candidate)
    return candidate


def normalize_split_folders(data_root: str) -> Dict[str, str]:
    """Rename split folders to canonical names 'train', 'validation', 'test'.

    Returns a mapping from canonical split name to absolute path.
    """
    result: Dict[str, str] = {}
    if not os.path.isdir(data_root):
        print(f"[WARN] Data root not found: {data_root}")
        return result

    entries = [
        d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
    ]

    for canonical, synonyms in SPLIT_SYNONYMS.items():
        current = find_first_matching_name(entries, synonyms)
        if not current:
            print(
                f"[INFO] Split folder not found for {canonical} (synonyms: {synonyms})"
            )
            continue

        src = os.path.join(data_root, current)
        dst = os.path.join(data_root, canonical)

        if os.path.abspath(src) != os.path.abspath(dst):
            print(f"[RENAME] {src} -> {dst}")
            # If destination exists but is different folder, merge by moving contents
            if os.path.exists(dst):
                # Move children from src into dst, then remove src
                for name in os.listdir(src):
                    s = os.path.join(src, name)
                    d = os.path.join(dst, name)
                    if os.path.isdir(s):
                        # If directory exists, keep as is; otherwise move
                        if not os.path.exists(d):
                            os.rename(s, d)
                        else:
                            print(f"[SKIP] Directory already exists: {d}")
                    else:
                        d = safe_rename(s, d)
                # Remove src if empty
                try:
                    os.rmdir(src)
                except OSError:
                    pass
            else:
                os.rename(src, dst)

        result[canonical] = dst

    return result


def normalize_korean_subfolders(split_path: str) -> Dict[str, str]:
    """Within a split ('train'/'validation'/'test'), rename Korean folders to canonical.

    Returns mapping {canonical_name: path}
    """
    mapping: Dict[str, str] = {}
    if not os.path.isdir(split_path):
        return mapping

    entries = [
        d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))
    ]
    for canonical, pattern in KOR_CANONICAL_PATTERNS.items():
        current = ""
        for name in entries:
            if pattern.match(name):
                current = name
                break
        if not current:
            print(f"[INFO] Korean folder '{canonical}' not found under {split_path}")
            continue
        src = os.path.join(split_path, current)
        dst = os.path.join(split_path, canonical)
        if os.path.abspath(src) != os.path.abspath(dst):
            print(f"[RENAME] {src} -> {dst}")
            if os.path.exists(dst):
                # Merge contents, then remove src
                for name in os.listdir(src):
                    s = os.path.join(src, name)
                    d = os.path.join(dst, name)
                    if os.path.isdir(s):
                        if not os.path.exists(d):
                            os.rename(s, d)
                        else:
                            print(f"[SKIP] Directory already exists: {d}")
                    else:
                        d = safe_rename(s, d)
                try:
                    os.rmdir(src)
                except OSError:
                    pass
            else:
                os.rename(src, dst)
        mapping[canonical] = dst
    return mapping


def strip_prefixes_in_class_dirs(parent_path: str) -> None:
    """Remove TS_/VS_/TE_ prefixes from immediate subdirectory names (class folders).

    This does NOT rename files; only class directory names directly under parent_path.
    """
    if not os.path.isdir(parent_path):
        return
    for name in os.listdir(parent_path):
        src = os.path.join(parent_path, name)
        if not os.path.isdir(src):
            continue
        new_name = PREFIX_PATTERN.sub("", name)
        if new_name and new_name != name:
            dst = os.path.join(parent_path, new_name)
            dst = safe_rename(src, dst)
            print(f"[RENAME] {src} -> {dst}")


def main():
    # Project root assumed as current directory
    project_root = os.path.abspath(os.path.dirname(__file__))
    data_root = os.path.join(project_root, "data")

    print(f"[START] Normalizing dataset structure under: {data_root}")

    # 1) Normalize split folders
    split_paths = normalize_split_folders(data_root)

    # 2) For each split, normalize Korean subfolders and strip prefixes
    for split in ("train", "validation", "test"):
        split_path = split_paths.get(split, os.path.join(data_root, split))
        if not os.path.isdir(split_path):
            continue
        kor_paths = normalize_korean_subfolders(split_path)

        # Process both if exist
        for kor_name in ("라벨링데이터", "원천데이터"):
            folder = kor_paths.get(kor_name, os.path.join(split_path, kor_name))
            if os.path.isdir(folder):
                print(f"[CLEAN] Stripping class-folder prefixes in: {folder}")
                strip_prefixes_in_class_dirs(folder)

    print("[DONE] Normalization complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
