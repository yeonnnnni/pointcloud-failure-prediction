"""코드 없이 PPT를 만들 수 있도록 검증된 문서·그림·소형 표를 묶는다."""
import argparse
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/PPT_HANDOFF_20261126.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((root / "results/audit.json").read_text())
    if not audit.get("passed") or audit.get("pilot"):
        raise RuntimeError("전체 실험 독립 감사 통과가 필요합니다.")
    files = [root / "PRESENTATION_HANDOFF.md", root / "README.md"]
    files += sorted((root / "doc").glob("*.md"))
    files += sorted((root / "results").glob("*.json"))
    files += sorted((root / "results/pilot").glob("*"))
    files += sorted((root / "results/figures").glob("*"))
    files += sorted((root / "results/tables").glob("*"))
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if not path.is_file() or path.stat().st_size > 5 * 1024 * 1024:
                raise ValueError(f"전달용 소형 파일이 아닙니다: {path.name}")
            archive.write(path, path.relative_to(root))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("전달 ZIP 무결성 검증 실패")
        print(f"PPT 인수인계 {len(archive.namelist())}개 파일, {output.stat().st_size:,}bytes: {output.name}")


if __name__ == "__main__":
    main()
