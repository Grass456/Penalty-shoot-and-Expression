"""存盘人脸 crop → 图像张量（M2 部分微调用，plan.md §9）。

与 `scripts/cache_fer_features.py` 的数值路径严格一致（同一 ImageNet mean/std、
同尺寸 224、同 PIL RGB 读取），差别只在：这里返回张量而非 backbone embedding，
因为微调时 backbone 权重逐折变化，不能复用特征缓存（会泄漏折间信息）。

用法：
    from src.data.face_images import load_face_tensor
    tensor, ids = load_face_tensor(valid_df, project_root)   # (N, 3, 224, 224) float32

行顺序 = 传入 DataFrame 的顺序（调用方保证与 folds 索引对齐，即按 event_id 排序的 valid）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision import transforms


def preprocessing(img_size: int = 224) -> transforms.Compose:
    """ImageNet 归一化 + resize（与 cache_fer_features.make_preprocess 同值）。"""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((int(img_size), int(img_size))),
        transforms.ToTensor(),
        transforms.Normalize(list(mean), list(std)),
    ])


def load_face_tensor(valid, project_root: Path | str, img_size: int = 224,
                     verbose: bool = True) -> tuple[torch.Tensor, list[str]]:
    """读 valid（需含 event_id / face_path 列）的全部人脸图，返回 (N,3,S,S) 张量与 event_id 列表。

    缺失文件直接报错并列出前几个 —— 静默跳过会让张量与 folds 索引错位。
    """
    from PIL import Image

    root = Path(project_root)
    prep = preprocessing(img_size)
    paths = [root / str(p) for p in valid["face_path"]]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} 张人脸图缺失，如 {missing[0]}（张量会与 folds 错位，必须修复后重跑）")

    arr = np.zeros((len(paths), 3, int(img_size), int(img_size)), dtype=np.float32)
    for i, p in enumerate(paths):
        img = np.asarray(Image.open(p).convert("RGB"))
        arr[i] = prep(img).numpy()
    if verbose:
        print(f"人脸张量: {arr.shape}  {arr.nbytes / 1e6:.0f} MB  ({len(paths)} 张)")
    assert np.isfinite(arr).all(), "人脸张量含 NaN/Inf"
    return torch.from_numpy(arr), valid["event_id"].astype(str).tolist()


def assert_embedding_parity(tensor: torch.Tensor, event_ids: list[str], cache_npz: Path | str,
                            model, device: str = "cuda", n_check: int = 8,
                            tol: float = 1e-4) -> float:
    """图像管线一致性检查：本张量经 backbone 前向的 embedding 应与缓存 npz 一致。

    全项目其他臂都读 features/fer/<cache>/embeddings.npz，M2 直接读图。两处若预处理
    不等价（RGB 顺序、resize 插值、归一化值），M2 与 M1 的对比就不成立。这里抽查
    前 n_check 个事件，返回最大绝对偏差，超过 tol 报错。

    **两项调用细节都是 2026-11-05 实测定的：**

    1. 逐张前向（batch=1），与缓存构建脚本一致。batch≥4 时 cuDNN 换核，偏差约 1.4e-3。
    2. 检查期间临时打开 tf32，与缓存构建时的环境一致；结束后恢复原设置。
       本脚本的训练环境故意关 tf32（要求可复现），但那条路径会让同一张量、同一预处理的
       输出偏差 1.3e-3 —— 那是数值环境差异，不是预处理差异，混在一起会误报。

    真实预处理错误的量级（实测，供设 tol 参考）：BGR/RGB 互换 6.9e-1、
    归一化缺失 9.6e-1、水平翻转 2.4e-1、尺寸改 192 为 5.9e-1。
    正确环境下偏差为 0，tol=1e-4 与两类量级各有三个数量级余量。

    只在 backbone 为原始（未微调）状态时有效 —— 调用方须在训练开始前调用。
    """
    c = np.load(cache_npz, allow_pickle=True)
    cids = c["event_ids"].astype(str)
    pos = {e: i for i, e in enumerate(cids)}
    idx = [i for i, e in enumerate(event_ids) if e in pos][:n_check]
    if not idx:
        raise ValueError("一致性检查: 缓存 npz 与事件列表无交集")
    rows = [pos[event_ids[i]] for i in idx]
    ref = np.asarray(c["embedding"], dtype=np.float64)[rows]

    model = model.to(device).eval()
    prev_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = True   # 与缓存构建环境对齐（见 docstring 第 2 点）
    try:
        got = []
        with torch.no_grad():
            for i in idx:
                got.append(model(tensor[i:i + 1].to(device)).float().cpu().numpy())
        got = np.concatenate(got).astype(np.float64)
    finally:
        torch.backends.cudnn.allow_tf32 = prev_tf32

    dev = float(np.abs(got - ref).max())
    if dev > tol:
        raise AssertionError(
            f"图像管线与缓存 embedding 不一致: max|Δ| = {dev:.3e} > {tol:.1e}。"
            "正确预处理下应为 0（真实错误量级 0.24–0.96）。"
            "M2 直接读图，必须与其他臂的预处理数值路径一致，否则对比不成立。")
    return dev
