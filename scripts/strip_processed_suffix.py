import os
from pathlib import Path


def strip_processed(root: Path, dry_run: bool = False):
    count = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if "_processed" in p.stem:
            new_name = p.stem.replace("_processed", "") + p.suffix
            new_path = p.with_name(new_name)
            if new_path.exists():
                print(f"skip (exists): {new_path}")
                continue
            print(f"{'DRY' if dry_run else 'RENAME'}: {p.name} -> {new_path.name}")
            if not dry_run:
                p.rename(new_path)
            count += 1
    print(f"done. renamed {count} files (dry_run={dry_run})")


if __name__ == "__main__":
    base = Path(f"{os.getcwd()}/data/segmented_test")

    # 1) 드라이런으로 미리 확인
    strip_processed(base, dry_run=True)
    # 2) 실제 변경
    strip_processed(base, dry_run=False)
