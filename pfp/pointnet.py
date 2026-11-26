"""공식 ModelNet40-C PointNet을 그대로 사용하는 추론 전용 래퍼."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

OFFICIAL_COMMIT = "fad786c430c007cfbb7b4fe90ed1c9873d960921"
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "vendor" / "ModelNet40-C"
PREPROCESSING_VERSION = "official-pointnet-c-all-xyz-float32-no-normalize-v1"
CLASS_NAMES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_core(source_dir: str | Path = DEFAULT_SOURCE):
    """불필요한 다른 모델의 CUDA 확장 없이 공식 PointNet 소스만 읽는다."""
    path = Path(source_dir) / "pointnet_pyt" / "pointnet" / "model.py"
    spec = importlib.util.spec_from_file_location("pfp_official_pointnet_core", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PointNet(nn.Module):
    def __init__(self, source_dir: str | Path = DEFAULT_SOURCE):
        super().__init__()
        self.model = official_core(source_dir).PointNetCls(k=40, feature_transform=True)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # 공식 models/pointnet.py와 동일한 연산. 출력은 log_softmax 값이다.
        points = points.to(next(self.parameters()).device).transpose(2, 1).float()
        return self.model(points)[0]


def load_model(checkpoint: str | Path, device: str = "cuda:0", source_dir: str | Path = DEFAULT_SOURCE):
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))
    model = PointNet(source_dir)
    # 공식 배포 checkpoint에는 cfg와 optimizer도 들어 있어 weights_only=False 필요.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved["model_state"]
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    model.checkpoint_sha256 = sha256(checkpoint)
    model.official_commit = OFFICIAL_COMMIT
    return model


@torch.inference_mode()
def predict(model: nn.Module, points: np.ndarray, device: str = "cuda:0", batch_size: int = 32) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 1:
        raise ValueError(f"예상 shape (B,N,3), 실제 {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("추론 입력에 비유한 좌표가 있습니다.")
    if batch_size < 1:
        raise ValueError("batch_size는 양수여야 합니다.")
    model.eval()
    outputs = []
    for start in range(0, len(points), batch_size):
        tensor = torch.from_numpy(np.array(points[start:start + batch_size], dtype=np.float32, copy=True)).to(device)
        output = model(tensor)
        if output.shape != (len(tensor), 40) or not torch.isfinite(output).all():
            raise ValueError(f"유효하지 않은 PointNet 출력: {tuple(output.shape)}")
        outputs.append(output.cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 40), dtype=np.float32)


def verify_official_parity(model, points, device="cuda:0", source_dir=DEFAULT_SOURCE):
    """공식 wrapper class의 실제 AST를 별도로 실행하여 작은 배치를 대조한다.

    전체 models package import는 미사용 CUDA 확장을 요구한다. class 본문은
    변경하지 않고 그 class가 참조하는 nn/PointNetCls/클래스 수만 주입한다.
    """
    wrapper_path = Path(source_dir) / "models" / "pointnet.py"
    tree = ast.parse(wrapper_path.read_text())
    class_nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = {"nn": nn, "PointNetCls": official_core(source_dir).PointNetCls,
                 "DATASET_NUM_CLASS": {"modelnet40_c": 40}}
    exec(compile(ast.Module(body=class_nodes, type_ignores=[]), str(wrapper_path), "exec"), namespace)
    reference = namespace["PointNet"](dataset="modelnet40_c", task="cls_trans").to(device).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    tensor = torch.from_numpy(np.array(points, dtype=np.float32, copy=True)).to(device)
    with torch.inference_mode():
        expected = reference(pc=tensor)["logit"].cpu().numpy()
    actual = predict(model, points, device, batch_size=len(points))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    return {"n": len(points), "max_abs_difference": float(np.abs(actual - expected).max()),
            "predictions_equal": bool(np.array_equal(actual.argmax(1), expected.argmax(1))),
            "official_wrapper_sha256": sha256(wrapper_path),
            "official_core_sha256": sha256(Path(source_dir) / "pointnet_pyt/pointnet/model.py")}


def load_clean(clean_dir: str | Path):
    """공식 DGCNN protocol의 test_files 순서와 첫 1024점을 보존한다."""
    import h5py
    root = Path(clean_dir)
    listed = [line.strip() for line in (root / "test_files.txt").read_text().splitlines() if line.strip()]
    data, labels, files = [], [], []
    for item in listed:
        path = root / Path(item).name
        with h5py.File(path, "r") as handle:
            data.append(handle["data"][:, :1024].astype(np.float32))
            labels.append(handle["label"][:].reshape(-1).astype(np.int64))
        files.append({"file": path.name, "sha256": sha256(path)})
    return np.concatenate(data), np.concatenate(labels), files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/pointnet_model_best_test.pth")
    parser.add_argument("--clean-dir", default="data/modelnet40_ply_hdf5_2048")
    parser.add_argument("--output", default="results/clean_metrics.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = load_model(args.checkpoint, args.device)
    data, labels, files = load_clean(args.clean_dir)
    parity = verify_official_parity(model, data[:8], args.device)
    start = time.perf_counter()
    logits = predict(model, data, args.device, args.batch_size)
    elapsed = time.perf_counter() - start
    predicted = logits.argmax(axis=1)
    report = {"scope": "official clean test: first 1024 points, independent from corruption object IDs",
              "n": len(labels), "correct": int((predicted == labels).sum()),
              "accuracy": float((predicted == labels).mean()), "elapsed_seconds": elapsed,
              "official_reference_accuracy": 0.907, "official_commit": OFFICIAL_COMMIT,
              "checkpoint_sha256": model.checkpoint_sha256, "parity": parity, "files": files}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    Path("local").mkdir(exist_ok=True)
    np.savez_compressed("local/clean_predictions.npz", logits=logits, true_label=labels, pred_label=predicted)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
