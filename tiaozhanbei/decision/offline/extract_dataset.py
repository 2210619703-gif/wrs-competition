import json
import os

# 组长数据集根目录（含 manifest.json 与 0001/0002/...）
# 统一使用 tiaozhanbei/dataset_learn（不再在本目录复制一份）
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 决策模块根目录
DATASET_ROOT = os.path.abspath(os.path.join(_PKG, os.pardir, "dataset_learn"))
OUTPUT_FILE = os.path.join(_PKG, "simple_dataset.json")
DEFAULT_STORAGE_XYZ = [0.30, -0.20, 0.0]


def filter_valid_objects(objects):
    """与工作流「代码执行_预处理」保持一致：过滤无效零件。"""
    valid = []
    for obj in objects:
        pixels = obj.get("visible_pixels", 0)
        bbox = obj.get("bbox")
        if pixels <= 0 or bbox is None:
            continue
        valid.append(obj)
    return valid


def resolve_target_container(ann: dict) -> dict:
    """
    新数据集把收纳盒写在 scene_layout.storage_box.pos_m；
    旧数据可能有 target_container.xyz。
    """
    storage = ann.get("scene_layout", {}).get("storage_box", {})
    if isinstance(storage, dict) and storage.get("pos_m"):
        xyz = [float(v) for v in storage["pos_m"]]
        while len(xyz) < 3:
            xyz.append(0.0)
        return {"xyz": xyz[:3], "name": "收纳盒"}

    tc = ann.get("target_container")
    if isinstance(tc, dict) and tc.get("xyz") is not None:
        xyz = [float(v) for v in tc["xyz"]]
        while len(xyz) < 3:
            xyz.append(0.0)
        return {"xyz": xyz[:3], "name": tc.get("name", "收纳盒")}

    return {"xyz": list(DEFAULT_STORAGE_XYZ), "name": "收纳盒"}


def extract_all_samples():
    manifest_path = os.path.join(DATASET_ROOT, "manifest.json")
    if not os.path.exists(manifest_path):
        print("错误：找不到 manifest.json，请检查 DATASET_ROOT 路径。")
        print("当前路径：", manifest_path)
        return []

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sample_list = []
    # manifest.samples 可能只含本轮增量；完整集以数字子目录为准（约 500 组）
    sample_ids = sorted(
        d
        for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d)) and d.isdigit()
    )
    if not sample_ids:
        samples = manifest.get("samples", {})
        sample_ids = sorted(samples.keys()) if isinstance(samples, dict) else list(samples)

    for sample_id in sample_ids:
        ann_file = os.path.join(DATASET_ROOT, sample_id, "annotations.json")
        if not os.path.exists(ann_file):
            print(f"跳过缺失标注：{sample_id}")
            continue

        with open(ann_file, "r", encoding="utf-8") as f:
            ann = json.load(f)

        valid_objects = filter_valid_objects(ann.get("objects", []))
        target = resolve_target_container(ann)
        item = {
            "sample_id": sample_id,
            "user_cmd": ann.get("instruction", "抓取指定工业零件"),
            "vision_objects": {
                "objects": valid_objects,
                "target_container": target,
                "scene_layout": ann.get("scene_layout", {}),
            },
        }
        sample_list.append(item)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sample_list, f, ensure_ascii=False, indent=2)

    print(f"提取完成，共 {len(sample_list)} 组样本 -> {OUTPUT_FILE}")
    if sample_list:
        t0 = sample_list[0]["vision_objects"]["target_container"]
        print(f"示例 {sample_list[0]['sample_id']} 收纳盒坐标: {t0.get('xyz')}")
    return sample_list


if __name__ == "__main__":
    extract_all_samples()
