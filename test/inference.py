#!/usr/bin/env python3
"""
LLaVA 1.5 13B（HuggingFace 格式）在 dir.IMAGE_FOLDERS 上逐图推理，结果写入 results/。

- 默认从 ckpt/checkpoint-170500 加载；也可用 --model_dir 指向其它本地目录或 Hub 名（如 llava-hf/llava-1.5-13b-hf）。
- 每张图写一行 jsonl（字段与 AIGI-Holmes/inference.py 一致：image, folder, section, label, prompt, response, error, model_dir）。
- 每个数据集目录一个 .jsonl，每写完一条即 flush。
- 每个 IMAGE_FOLDERS 目录默认最多推理 128 张（超出部分跳过，可用 --max_per_folder 调整；0 表示不限制）。

GPU：在 import torch 之前根据 --gpu-id（或环境变量 REVEAL_GPU_ID）设置 CUDA_VISIBLE_DEVICES，
    即「第几张」物理卡（0 起算）；进程内仅见 cuda:0。

用法（REVEAL 根目录）:
  python3 inference.py --gpu-id 3
  python3 inference.py --gpu-id 2 --model_dir /path/to/llava-1.5-13b-hf
  python3 inference.py --max_new_tokens 1024 --dtype bf16 --out_dir ./results
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

# 与 jsonl 中 prompt 字段一致；VLM chat 会去掉开头的 "<image>\\n" 交给模板（模板会再插图像）
USER_PROMPT = (
    "<image>\n"
    "Please help me determine whether this image is real or synthetic (i.e., modified, composited, or generated). "
    "Return 1 for generated images and 0 for real images. During your reasoning, pay attention to the image's "
    "lighting, texture, shadows, edges, and object details, especially any signs that may reveal whether the image "
    "has been artificially modified or generated, or characteristics that suggest its authenticity.\n\n"
    "Reasoning: <think> [Detailed reasoning process, step-by-step analysis of the image's features, "
    "providing the reasoning conclusion] </think>\n"
    "Answer: <answer> [1 or 0] </answer>"
)

DEFAULT_RESULTS_DIR = Path(
    "/mnt/shangcephfs/mm-base-vision-ascend/sanmucao/code/ms-swift/test"
)
_DEFAULT_MODEL_DIR = (
    "/mnt/shangcephfs/mm-base-vision-ascend/sanmucao/datasets/benchmark/REVEAL/model/REVEAL"
)

# 每个 IMAGE_FOLDERS 目录最多取多少张参与推理（sorted 后取前 N 张）
DEFAULT_MAX_PER_FOLDER = 128


def _preparse_gpu_id() -> str:
    """在 import torch 之前解析「第几张 GPU」。"""
    for i, a in enumerate(sys.argv):
        if a == "--gpu-id" and i + 1 < len(sys.argv):
            return sys.argv[i + 1].strip() or "0"
        if a.startswith("--gpu-id="):
            return a.split("=", 1)[1].strip() or "0"
    return (os.environ.get("REVEAL_GPU_ID", "0") or "0").strip()


os.environ["CUDA_VISIBLE_DEVICES"] = _preparse_gpu_id()

import torch
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent

from dir import IMAGE_FOLDERS  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def _user_text_for_vlm_chat() -> str:
    if USER_PROMPT.startswith("<image>\n"):
        return USER_PROMPT[len("<image>\n") :]
    if USER_PROMPT.startswith("<image>"):
        return USER_PROMPT[len("<image>") :].lstrip("\n")
    return USER_PROMPT


def _list_images(root: str) -> List[Path]:
    p = Path(root)
    if not p.is_dir():
        return []
    out: List[Path] = []
    for a in p.rglob("*"):
        if a.is_file() and a.suffix.lower() in EXTS:
            out.append(a)
    return sorted(out)


def _label_from_folder(folder: str) -> int:
    s = str(folder).rstrip("/").lower()
    parent = Path(folder).name.lower()
    if parent in ("0_real", "1_fake"):
        return 0 if parent == "0_real" else 1
    if s.endswith("fake") or "/fake" in s:
        return 1
    if s.endswith("real") or "/real" in s:
        return 0
    if parent == "fake":
        return 1
    if parent == "real":
        return 0
    raise ValueError(f"无法从路径推断标签: {folder!r}")


def _section_from_path(path: Path) -> str:
    sp = str(path).replace("\\", "/")
    for seg in sp.split("/"):
        if len(seg) == 3 and seg[0].isdigit() and seg[1] == "." and seg[2].isdigit():
            return seg
    m = re.search(r"/(\d\.\d)/", sp)
    return m.group(1) if m else "unknown"


def _gather_samples(max_per_folder: int = DEFAULT_MAX_PER_FOLDER) -> Tuple[List[Path], List[int], List[str], List[str]]:
    files: List[Path] = []
    labels: List[int] = []
    sections: List[str] = []
    image_folders: List[str] = []
    for folder in IMAGE_FOLDERS:
        ps = _list_images(folder)
        if not ps:
            print(f"[skip] 无图片: {folder}", flush=True)
            continue
        n_all = len(ps)
        if max_per_folder > 0 and n_all > max_per_folder:
            ps = ps[:max_per_folder]
            print(f"[cap] {folder}: 仅测前 {max_per_folder}/{n_all} 张（sorted）", flush=True)
        y = _label_from_folder(folder)
        for p in ps:
            files.append(p)
            labels.append(y)
            sections.append(_section_from_path(p))
            image_folders.append(str(Path(folder).resolve()))
    return files, labels, sections, image_folders


def _safe_run_subdir(name: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", name).strip("_")
    return s or "run"


def _dataset_dir_to_jsonl_name(folder: str) -> str:
    """与 AIGI-Holmes 一致：.../1.1/test/fake -> 1.1_test_fake；.../gen/pixel/1_fake -> gen_pixel_1_fake。"""
    p = Path(folder).resolve()
    parts = p.parts
    if len(parts) >= 3 and parts[-2] == "test":
        stem = f"{parts[-3]}_test_{parts[-1]}"
    elif len(parts) >= 2 and parts[-1] in ("0_real", "1_fake"):
        stem = "_".join(parts[-3:]) if len(parts) >= 3 else "_".join(parts[-2:])
    elif len(parts) >= 1:
        stem = "_".join(parts[-4:]) if len(parts) >= 4 else p.name
    else:
        stem = "data"
    stem = _safe_run_subdir(stem)
    return f"{stem}.jsonl"


class JsonlByDatasetDirWriter:
    def __init__(self, out_root: Path, run_name: str):
        self.subdir = (out_root / _safe_run_subdir(run_name)).resolve()
        self.subdir.mkdir(parents=True, exist_ok=True)
        self._handles: Dict[str, TextIO] = {}

    def append_row(self, folder: str, row: Dict[str, Any]) -> None:
        k = str(Path(folder).resolve())
        if k not in self._handles:
            path = self.subdir / _dataset_dir_to_jsonl_name(k)
            self._handles[k] = open(path, "a", encoding="utf-8")
        f = self._handles[k]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()

    def close(self) -> None:
        for f in self._handles.values():
            f.close()
        self._handles.clear()


def _resolve_dtype(name: str) -> torch.dtype:
    n = (name or "auto").lower()
    if n == "bf16" or (n == "auto" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    if n == "fp32":
        return torch.float32
    return torch.float16


def _model_run_name(model_dir: Path) -> str:
    s = str(model_dir)
    if "/" in s or "\\" in s:
        return _safe_run_subdir(s.replace("\\", "/").replace("/", "__"))
    return model_dir.name or "model"


def _auto_model_for_vlm():
    """兼容 transformers 版本：旧版有 ``AutoModelForVision2Seq``，5.x 起多为 ``AutoModelForImageTextToText``。"""
    try:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq
    except ImportError:
        pass
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError:
        pass
    from transformers.models.auto.modeling_auto import AutoModelForImageTextToText

    return AutoModelForImageTextToText


def _load_hf_vlm(model_id: str, dtype: torch.dtype, *, ignore_mismatched_sizes: bool = False):
    """按 checkpoint 的 config.json 选模型类（Auto），勿对训练导出权重强行用 LlavaForConditionalGeneration。

    训练导出的 LLaVA 常为 ``LlavaLlamaForCausalLM`` / ``model_type: llava_llama``，与 Hub 上
    ``llava-hf/llava-1.5-13b-hf`` 的 ``LlavaForConditionalGeneration`` 权重布局不同，混用会触发
    state dict 尺寸不匹配。若 Auto 加载仍有个别层不匹配，可试 ``--ignore_mismatched_sizes``。
    """
    from transformers import AutoProcessor,LlavaForConditionalGeneration

    AutoVLM = _auto_model_for_vlm()
    load_kw: Dict[str, Any] = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if ignore_mismatched_sizes:
        load_kw["ignore_mismatched_sizes"] = True
    # model = LlavaForConditionalGeneration.from_pretrained(model_id, **load_kw)\
    device = f"cuda:{0}"
    model = LlavaForConditionalGeneration.from_pretrained("/mnt/shenzhen2cephfs/mm-base-vision/sanmucao/code/eccv_rebuttal/REVEAL/ckpt/v0-20251106-103918/checkpoint-71500", torch_dtype="auto", device_map=device)
#     model = LlavaForConditionalGeneration.from_pretrained(
#     "/mnt/shenzhen2cephfs/mm-base-vision/sanmucao/code/eccv_rebuttal/REVEAL/ckpt/checkpoint-170500",
#     torch_dtype="auto",
#     device_map=device,
#     ignore_mismatched_sizes=True,   # ⭐关键
#     trust_remote_code=True
# )
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
    if torch.cuda.is_available():
        model = model.to(0)
    else:
        model = model.to("cpu")
    model.eval()
    return model, processor


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.inference_mode()
def _vlm_generate_one(
    model,
    processor,
    image_path: Path,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> Tuple[str, Optional[str]]:
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception as e:  # noqa: BLE001
        return "", f"read_image: {e}"
    user_text = _user_text_for_vlm_chat()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    try:
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:  # noqa: BLE001
        return "", f"apply_chat_template: {e}"
    try:
        inputs = processor(text=prompt, images=im, return_tensors="pt")
    except Exception as e:  # noqa: BLE001
        return "", f"processor: {e}"

    dev = next(model.parameters()).device
    inputs = _to_device(dict(inputs), dev)
    gen_kw: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kw["temperature"] = temperature
    try:
        gen_ids = model.generate(**inputs, **gen_kw)
    except Exception as e:  # noqa: BLE001
        return "", f"generate: {e}"

    in_len = inputs["input_ids"].shape[1]
    new_tokens = gen_ids[:, in_len:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return text.strip(), None


def run_vlm(
    files: List[Path],
    labels: List[int],
    sections: List[str],
    image_folders: List[str],
    model_ref: str,
    out_root: Path,
    max_new_tokens: int,
    dtype_name: str,
    do_sample: bool,
    temperature: float,
    ignore_mismatched_sizes: bool = False,
) -> int:
    dtype = _resolve_dtype(dtype_name)
    print(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r} | "
        f"加载: {model_ref!r} | dtype={dtype}",
        flush=True,
    )
    model, processor = _load_hf_vlm(
        model_ref, dtype, ignore_mismatched_sizes=ignore_mismatched_sizes
    )
    if os.path.isdir(model_ref):
        run_name = _model_run_name(Path(model_ref))
    else:
        run_name = _safe_run_subdir(model_ref.replace("/", "__"))

    writer = JsonlByDatasetDirWriter(out_root, run_name)
    n_ok = 0
    try:
        for i in tqdm(range(len(files)), desc="llava1.5-13b"):
            p = files[i]
            resp, err = _vlm_generate_one(
                model,
                processor,
                p,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )

            row = {
                "image": str(p.resolve()),
                "folder": image_folders[i],
                "section": sections[i],
                "label": int(labels[i]),
                "prompt": USER_PROMPT,
                "response": resp,
                "error": err,
                "model_dir": str(Path(model_ref).resolve()) if os.path.isdir(model_ref) else str(model_ref),
            }
            writer.append_row(image_folders[i], row)
            if not err:
                n_ok += 1
    finally:
        writer.close()
    print(f"完成: 成功 {n_ok}/{len(files)} | 输出: {writer.subdir}", flush=True)
    return 0 if n_ok == len(files) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="REVEAL：LLaVA 1.5 13B 推理（与 AIGI-Holmes jsonl 格式对齐）")
    ap.add_argument(
        "--gpu-id",
        type=str,
        default=_preparse_gpu_id(),
        help="物理 GPU 编号（0 起）；在 import torch 前已通过 argv 解析，此处仅用于帮助信息一致",
    )
    ap.add_argument(
        "--model_dir",
        type=str,
        default=str(_DEFAULT_MODEL_DIR),
        help="本地 HF 目录或 Hub id（如 llava-hf/llava-1.5-13b-hf），需含 config 与权重",
    )
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument(
        "--max_per_folder",
        type=int,
        default=DEFAULT_MAX_PER_FOLDER,
        help=f"每个 IMAGE_FOLDERS 目录最多推理多少张（默认 {DEFAULT_MAX_PER_FOLDER}，sorted 后截取）；0 表示不限制",
    )
    ap.add_argument("--max_new_tokens", type=int, default=10240)
    ap.add_argument("--dtype", type=str, default="auto", choices=("auto", "fp16", "bf16", "fp32"))
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="from_pretrained 时忽略权重与模型形状不一致的层（慎用，仅作兜底）",
    )
    args = ap.parse_args()

    model_ref = args.model_dir.strip()
    if os.path.isdir(model_ref) and not (Path(model_ref) / "config.json").is_file():
        print(f"本地目录缺少 config.json: {model_ref}\n可改用 Hub：--model_dir llava-hf/llava-1.5-13b-hf")
        return 1

    files, labels, sections, image_folders = _gather_samples(max_per_folder=args.max_per_folder)
    if not files:
        print("无可用样本，请检查 dir.IMAGE_FOLDERS。")
        return 1

    out_root = Path(args.out_dir).expanduser().resolve()
    return run_vlm(
        files,
        labels,
        sections,
        image_folders,
        model_ref,
        out_root,
        max_new_tokens=args.max_new_tokens,
        dtype_name=args.dtype,
        do_sample=args.do_sample,
        temperature=args.temperature,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
