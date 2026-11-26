"""실험에 사용한 clean 미러에서 test 파일 6개만 검증·추출한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


DRIVE_ID = "1BUP46fXOlLVeGKa7PT1CtBEi3s0HGHUi"
EXPECTED_SHA256 = {
    "ply_data_test0.h5": "fb1631826ff4102aac03ee8a59ed309f8ce839a210b6da1a40a2e4d45bba9d56",
    "ply_data_test1.h5": "dc002c8e9e168cedd22afbdfa8f1fae19d0f2fafb6efa993e4c88fbcaee8f353",
    "ply_data_test_0_id2file.json": "a821815a14bd8ba38b8036fdd7bc5798a3d16d098a2bd5be52afcd038013079a",
    "ply_data_test_1_id2file.json": "74fb14133606193f452d60a72d6fc053e9604995c263457184260dfcfbd3ce26",
    "test_files.txt": "c19374061df2c9afff7e11acffd69468b8212a47d2782a16c0a921ee4583a104",
    "shape_names.txt": "7cf9a7b0d1f39eeb861dc43406947d29c1cd429c2f4ff2530ec153d4b8d83609",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(archive: Path, output: Path, download: bool = False):
    output.mkdir(parents=True, exist_ok=True)
    needed = [name for name, digest in EXPECTED_SHA256.items()
              if not (output / name).is_file() or sha256(output / name) != digest]
    if needed and not archive.exists():
        if not download:
            raise FileNotFoundError(f"미러 ZIP이 없습니다: {archive}. --download로 내려받을 수 있습니다.")
        archive.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "gdown", DRIVE_ID, "-O", str(archive)], check=True)
    if needed:
        with zipfile.ZipFile(archive) as source:
            for name in needed:
                # 지정한 멤버만 읽고 고정된 파일명으로 쓴다. extractall은 사용하지 않는다.
                raw = source.read(f"modelnet40_ply_hdf5_2048/{name}")
                if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256[name]:
                    raise ValueError(f"실험에 사용한 clean 파일과 SHA-256이 다릅니다: {name}")
                temporary = output / f"{name}.part"
                temporary.write_bytes(raw)
                temporary.replace(output / name)
    report = {
        "source": f"https://drive.google.com/file/d/{DRIVE_ID}/view",
        "verification": "선택된 ZIP 멤버 CRC 및 실험에 사용한 6개 파일의 SHA-256 일치",
        "files": [{"file": name, "sha256": digest, "bytes": (output / name).stat().st_size,
                   "reused": name not in needed} for name, digest in EXPECTED_SHA256.items()],
        "test_files_order": ["ply_data_test0.h5", "ply_data_test1.h5"],
        "excluded": "ModelNet40 train 및 ShapeNetPart 파일은 추출하지 않음",
    }
    (output / "preparation_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/modelnet40_ply_hdf5_2048_mirror.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/modelnet40_ply_hdf5_2048"))
    parser.add_argument("--download", action="store_true", help="필요한 ZIP이 없으면 지정된 미러에서 다운로드")
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output, args.download), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
