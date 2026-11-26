"""파일럿 → 조건별 재개 가능한 전체 추론·geometry·평가."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.special import softmax

from .data import (CONFIDENCE, CORRUPTIONS, KEYS, discover_conditions, load_or_create_splits,
                   pilot_ids, sha256, validate_table, write_json)
from .geometry import GEOMETRY, extract_geometry


def log(message):
    print(f"{datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')} {message}", flush=True)


def confidence_table(logits, ids, labels, corruption, severity):
    if logits.shape != (len(ids), 40) or not np.isfinite(logits).all():
        raise ValueError(f"40차원 유한 logit 검증 실패: {logits.shape}")
    probability = softmax(logits.astype(np.float64), axis=1)
    ordered = np.sort(probability, axis=1)
    prediction = logits.argmax(axis=1)
    return pd.DataFrame({"object_id": ids, "corruption": corruption, "severity": severity,
                         "true_label": labels[ids], "pred_label": prediction,
                         "error": (prediction != labels[ids]).astype(int),
                         "msp": ordered[:, -1], "margin": ordered[:, -1] - ordered[:, -2],
                         "entropy": -(probability * np.log(np.maximum(probability, 1e-300))).sum(axis=1)})


def run(args):
    import torch
    from .pointnet import DEFAULT_SOURCE, load_model, predict
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    all_conditions, level_map = discover_conditions(args.data_dir, require_complete=args.mode == "full" and not args.wait_data)
    if args.mode == "full" and args.wait_data:
        all_conditions = [(c, s, Path(args.data_dir) / f"data_{c}_{s}.npy") for c in sorted(CORRUPTIONS) for s in range(1, 6)]
        level_map = {str(s): s for s in range(1, 6)}
    label_path = Path(args.data_dir) / "label.npy"
    labels = np.load(label_path).reshape(-1).astype(int)
    if len(labels) != 2468 or set(labels) != set(range(40)):
        raise ValueError(f"공식 라벨 수/클래스 확인 필요: {len(labels)}개")
    splits = load_or_create_splits(labels, args.splits, args.seed)
    ids = pilot_ids(splits, args.pilot_objects, args.seed) if args.mode == "pilot" else np.arange(len(labels))
    conditions = [c for c in all_conditions if c[0] in {"gaussian", "cutout"} and c[1] in {1, 5}] if args.mode == "pilot" else all_conditions
    if args.mode == "pilot" and {(c, s) for c, s, _ in conditions} != {(c, s) for c in ["gaussian", "cutout"] for s in [1, 5]}:
        raise ValueError("파일럿 4개 조건 파일이 모두 필요함")
    if args.mode == "full":
        pilot_report = json.loads(Path(args.pilot_report).read_text())
        if not pilot_report.get("passed") or not pilot_report.get("evaluation_passed"):
            raise ValueError("파일럿 무결성·탐지 평가를 먼저 통과해야 함")
    code_hashes = {name: sha256(Path(__file__).parent / name)
                   for name in ["pipeline.py", "data.py", "geometry.py", "pointnet.py"]}
    code_hashes.update({f"official/{name}": sha256(DEFAULT_SOURCE / name)
                       for name in ["pointnet_pyt/pointnet/model.py", "models/pointnet.py"]})
    configuration = {"checkpoint_sha256": sha256(args.checkpoint), "label_sha256": sha256(label_path),
                     "split_sha256": sha256(args.splits), "ids": ids.tolist(), "seed": args.seed,
                     "features": CONFIDENCE + GEOMETRY, "geometry_knn": 8,
                     "geometry_input": "original_corrupted_xyz", "model_input": "official_all_corrupted_points_float32",
                     "code_sha256": code_hashes, "torch": torch.__version__, "numpy": np.__version__,
                     "batch_size": args.batch_size, "tf32": False}
    if args.mode == "full":
        pilot_config = json.loads(Path(args.pilot_report).with_name("configuration.json").read_text())
        for field in ["checkpoint_sha256", "label_sha256", "split_sha256", "features", "geometry_knn", "model_input", "code_sha256"]:
            if pilot_config[field] != configuration[field]:
                raise ValueError(f"파일럿과 전체 실행 설정 불일치: {field}; 파일럿 재검증 필요")
    write_json(root / "configuration.json", configuration)
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "device": args.device,
                   "gpu": torch.cuda.get_device_name(torch.device(args.device)) if "cuda" in args.device else None,
                   "pid": os.getpid(), "batch_size": args.batch_size, "geometry_workers": args.workers,
                   "checkpoint_path": str(Path(args.checkpoint).resolve()), "data_path": str(Path(args.data_dir).resolve()),
                   "level_mapping": level_map, "mode": args.mode}
    write_json(root / "environment.local.json", environment)
    log(f"mode={args.mode}, objects={len(ids)}, conditions={len(conditions)}, rows={len(ids)*len(conditions)}, pid={os.getpid()}")
    model = load_model(args.checkpoint, args.device)
    tables, inventory = [], []
    start = time.perf_counter()
    cache_root = root / "conditions"
    cache_root.mkdir(exist_ok=True)
    pending = list(conditions)
    waiting_seconds = 0
    for number in range(1, len(conditions) + 1):
        ready = [condition for condition in pending if condition[2].exists()]
        while not ready:
            if waiting_seconds >= args.wait_data:
                raise FileNotFoundError(f"미완료 데이터 {len(pending)}조건: {pending[0][2]}")
            log(f"다운로드 대기: {number-1}/{len(conditions)}조건 완료, {len(pending)}조건 미수신")
            write_json(root / "status.json", {"mode": args.mode, "completed_conditions": number-1,
                       "total_conditions": len(conditions), "state": "waiting_for_data", "pid": os.getpid(),
                       "waiting_seconds": waiting_seconds,
                       "updated_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat()})
            time.sleep(15)
            waiting_seconds += 15
            ready = [condition for condition in pending if condition[2].exists()]
        corruption, severity, data_path = ready[0]
        pending.remove(ready[0])
        name = f"{corruption}_{severity}"
        cache = cache_root / name
        cache.mkdir(exist_ok=True)
        points = np.load(data_path, mmap_mode="r")
        if points.ndim != 3 or points.shape[0] != len(labels) or points.shape[-1] != 3 or points.shape[1] < 2:
            raise ValueError(f"{name}: 공식 배열 shape 확인 필요 {points.shape}")
        signature = {**configuration, "data_sha256": sha256(data_path), "condition": name, "shape": list(points.shape)}
        fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        meta_path, table_path, logits_path = cache / "metadata.json", cache / "table.parquet", cache / "logits.npz"
        try:
            metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except (json.JSONDecodeError, OSError):
            metadata = {}
        cached = metadata.get("fingerprint") == fingerprint and table_path.exists() and logits_path.exists()
        if cached:
            try:
                if metadata.get("table_sha256") != sha256(table_path) or metadata.get("logits_sha256") != sha256(logits_path):
                    raise ValueError("캐시 파일 hash 불일치")
                table = pd.read_parquet(table_path)
                validate_table(table, splits, ids, [(corruption, severity)])
                with np.load(logits_path) as saved:
                    if not np.array_equal(saved["object_id"], ids):
                        raise ValueError("logit ID 불일치")
                    predicted = confidence_table(saved["logits"], ids, labels, corruption, severity)
                    if not np.array_equal(predicted.pred_label, table.pred_label):
                        raise ValueError("logit과 표 예측 불일치")
                    np.testing.assert_allclose(predicted[CONFIDENCE], table[CONFIDENCE], rtol=1e-12, atol=1e-12)
            except Exception as exc:
                log(f"{name}: 캐시 검증 실패, 재실행: {exc}")
                cached = False
        if not cached:
            log(f"[{number}/{len(conditions)}] {name} shape={points.shape} 시작")
            def geometry_job():
                begin = time.perf_counter()
                features = extract_geometry(points, ids, args.workers)
                return features, time.perf_counter() - begin
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(geometry_job)
                inference_start = time.perf_counter()
                logits = predict(model, points[ids], args.device, args.batch_size)
                if "cuda" in args.device:
                    torch.cuda.synchronize()
                inference_seconds = time.perf_counter() - inference_start
                features, geometry_seconds = future.result()
            confidence = confidence_table(logits, ids, labels, corruption, severity)
            geometry = pd.DataFrame(features, columns=GEOMETRY)
            geometry["object_id"], geometry["corruption"], geometry["severity"] = ids, corruption, severity
            table = confidence.merge(geometry, on=KEYS, how="outer", validate="one_to_one", indicator=True)
            if not table._merge.eq("both").all():
                raise ValueError(f"{name}: confidence/geometry 조인 누락")
            table = table.drop(columns="_merge").merge(splits[["object_id", "split"]], on="object_id", validate="many_to_one")
            report = validate_table(table, splits, ids, [(corruption, severity)])
            temporary = table_path.with_suffix(".tmp.parquet")
            table.to_parquet(temporary, index=False)
            temporary.replace(table_path)
            np.savez_compressed(logits_path, logits=logits, object_id=ids)
            metadata = {"fingerprint": fingerprint, "shape": list(points.shape), "data_sha256": signature["data_sha256"],
                        "table_sha256": sha256(table_path), "logits_sha256": sha256(logits_path),
                        "inference_seconds": inference_seconds, "geometry_seconds": geometry_seconds, **report}
            write_json(meta_path, metadata)
        tables.append(table)
        inventory.append({"corruption": corruption, "severity": severity, "filename": data_path.name,
                          "rows": int(points.shape[0]), "points": int(points.shape[1]), "bytes": data_path.stat().st_size,
                          "data_sha256": metadata["data_sha256"], "inference_seconds": metadata["inference_seconds"],
                          "geometry_seconds": metadata["geometry_seconds"], "reused": cached})
        elapsed = time.perf_counter() - start
        status = {"mode": args.mode, "completed_conditions": number, "total_conditions": len(conditions),
                  "rows": sum(len(t) for t in tables), "elapsed_seconds": elapsed,
                  "eta_seconds": elapsed / number * (len(conditions) - number), "last_condition": name,
                  "pid": os.getpid(), "updated_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat()}
        write_json(root / "status.json", status)
        log(f"[{number}/{len(conditions)}] {name} {'캐시 검증' if cached else '완료'} infer={metadata['inference_seconds']:.2f}s geometry={metadata['geometry_seconds']:.2f}s ETA={status['eta_seconds']/60:.1f}min")
        del points
    table = pd.concat(tables, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    report = validate_table(table, splits, ids, [(c, s) for c, s, _ in conditions])
    table.to_parquet(root / "table.parquet", index=False)
    pd.DataFrame(inventory).to_csv(root / "data_inventory.csv", index=False)
    report.update({"mode": args.mode, "elapsed_seconds": time.perf_counter()-start,
                   "table_sha256": sha256(root / "table.parquet"), "evaluation_passed": False})
    write_json(root / "validation.json", report)
    if not args.skip_evaluation:
        from .evaluation import evaluate
        evaluate(table, root / "evaluation", pilot=args.mode == "pilot")
        report["evaluation_passed"] = True
        write_json(root / "validation.json", report)
    log(f"전체 파이프라인 완료: {root}; 검증={report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pilot", "full"], required=True)
    parser.add_argument("--data-dir", default="data/modelnet40_c")
    parser.add_argument("--checkpoint", default="weights/pointnet_model_best_test.pth")
    parser.add_argument("--splits", default="configs/object_splits.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-report", default="artifacts/pilot/validation.json")
    parser.add_argument("--pilot-objects", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--wait-data", type=int, default=0, help="전체 실행 중 미수신 조건을 기다릴 최대 누적 초")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
