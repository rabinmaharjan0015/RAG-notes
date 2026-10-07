"""Model loading. Each model is downloaded ONCE into backend/models/, then loaded offline."""

from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

import threading

from .config import MODELS_DIR, EMBEDDING_MODEL, QA_MODEL, VLM_MODEL, ENABLE_VLM, logger

_embedder = None
_generator = None  # (tokenizer, model)


def ensure_local_model(repo_id: str) -> str:
    """Return a local folder for the model, downloading it only the first time."""
    local = MODELS_DIR / repo_id.replace("/", "--")
    marker = local / ".download_complete"
    if marker.exists():
        return str(local)  # already downloaded -> no network access at all

    from huggingface_hub import snapshot_download, list_repo_files

    # Skip duplicate weight formats (ONNX / OpenVINO / TF / Flax / .bin twins) to keep
    # the one-time download small.
    ignore = ["onnx/*", "openvino/*", "*.onnx", "*.h5", "*.msgpack", "*.ot", "*.tflite", "coreml/*"]
    try:
        if any(f.endswith(".safetensors") for f in list_repo_files(repo_id)):
            ignore.append("pytorch_model*.bin")
    except Exception:
        pass

    logger.info(f"[models] First-time download of {repo_id} -> {local} (one time only)")
    snapshot_download(repo_id=repo_id, local_dir=str(local), ignore_patterns=ignore)
    marker.write_text(datetime.now(timezone.utc).isoformat())
    logger.info(f"[models] {repo_id} saved. It will not be downloaded again.")
    return str(local)


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(ensure_local_model(EMBEDDING_MODEL), device="cpu")
        _embedder.max_seq_length = 256
        logger.info("[models] Embedding model ready")
    return _embedder


def get_generator() -> Optional[tuple]:
    """Optional LLM. Returns None when HF_QA_MODEL is empty."""
    global _generator
    if _generator is None and QA_MODEL:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        path = ensure_local_model(QA_MODEL)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto")
        model.eval()
        _generator = (tok, model)
        logger.info("[models] Generator ready")
    return _generator


def _is_e5() -> bool:
    return "e5" in EMBEDDING_MODEL.lower()


def embed_passages(texts: List[str]) -> np.ndarray:
    if _is_e5():
        texts = [f"passage: {t}" for t in texts]
    return get_embedder().encode(
        texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
    )


def embed_query(text: str) -> np.ndarray:
    if _is_e5():
        text = f"query: {text}"
    return get_embedder().encode([text], show_progress_bar=False, normalize_embeddings=True)[0]


# ─── Vision-language model (figures, charts, photos) ────────────
_vlm = None
_vlm_failed = False
_vlm_lock = threading.Lock()


def get_vlm():
    """Lazy-load the Hugging Face VLM. Returns (processor, model) or None if unavailable."""
    global _vlm, _vlm_failed
    if _vlm is not None or _vlm_failed or not ENABLE_VLM or not VLM_MODEL:
        return _vlm
    with _vlm_lock:
        if _vlm is not None or _vlm_failed:
            return _vlm
        try:
            from transformers import AutoProcessor

            try:
                from transformers import AutoModelForImageTextToText as AutoVLM
            except ImportError:  # older transformers
                from transformers import AutoModelForVision2Seq as AutoVLM

            path = ensure_local_model(VLM_MODEL)
            processor = AutoProcessor.from_pretrained(path)
            model = AutoVLM.from_pretrained(path, torch_dtype="auto")
            model.eval()
            _vlm = (processor, model)
            logger.info("[models] Vision model ready")
        except Exception as e:  # never break ingestion because the VLM is missing
            _vlm_failed = True
            logger.warning(f"[models] Vision model unavailable, images will use captions/context only: {e}")
    return _vlm


def vlm_available() -> bool:
    return bool(ENABLE_VLM and VLM_MODEL and not _vlm_failed)


def vlm_generate(image_path, prompt: str, max_new_tokens: int = 120) -> Optional[str]:
    """Ask the VLM about one image. Returns None if the model is unavailable or fails."""
    vlm = get_vlm()
    if vlm is None:
        return None
    try:
        import torch
        from PIL import Image

        processor, model = vlm
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((768, 768))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[img], return_tensors="pt")
        with _vlm_lock, torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return gen.strip() or None
    except Exception as e:
        logger.warning(f"VLM generation failed: {e}")
        return None
