#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import time
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd

# Dépendances modèle local
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# 1. Dataclasses de config
# =========================

@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = False
    repetition_penalty: float = 1.0


@dataclass
class ModelConfig:
    # ici: chemin local vers le modèle (HF format) ou nom HF si tu acceptes le téléchargement
    model_name: str = "models/qwen3-32b"   # exemple: dossier local
    device_map: str = "auto"
    dtype: str = "float16"                # "float16" / "bfloat16" / "float32"
    local_files_only: bool = True         # True => pas de téléchargement
    trust_remote_code: bool = False       # True si modèle en a besoin (Qwen, etc.)


@dataclass
class IOConfig:
    data_path: str = "data/spans_dataset.tsv"
    sep: str = "\t"
    encoding: str = "utf-8"
    sentence_col: str = "Sentence_en"
    out_dir: str = "results"


@dataclass
class RuntimeConfig:
    batch_size: int = 100
    log_file: str = "run_local.log"
    log_level: str = "INFO"
    resume: bool = True


@dataclass
class PromptConfig:
    template_paths: List[str] = None
    template_path: Optional[str] = None
    sentence_var: str = "sentence"


@dataclass
class GlobalConfig:
    model: ModelConfig
    generation: GenerationConfig
    io: IOConfig
    runtime: RuntimeConfig
    prompt: PromptConfig


# =========================
# 2. Utilitaires de config
# =========================

def _dict_to_dataclass(dc_cls, data: Dict[str, Any]):
    field_names = {f.name for f in dc_cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return dc_cls(**filtered)

def load_config_from_json(path: str) -> GlobalConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier de config introuvable : {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    paths = raw.get("paths", {})

    def expand(value):
        if isinstance(value, str):
            return value.format(**paths)
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value

    model_raw = expand(raw.get("model", {}))
    gen_raw = expand(raw.get("generation", {}))
    io_raw = expand(raw.get("io", {}))
    runtime_raw = expand(raw.get("runtime", {}))
    prompt_raw = expand(raw.get("prompt", {}))

    # compat : si template_paths vide mais template_path rempli
    prompt_cfg = _dict_to_dataclass(PromptConfig, prompt_raw)
    if (not prompt_cfg.template_paths or len(prompt_cfg.template_paths) == 0) and prompt_cfg.template_path:
        prompt_cfg.template_paths = [prompt_cfg.template_path]
    if not prompt_cfg.template_paths:
        raise ValueError("Aucun prompt n'est défini (prompt.template_paths est vide).")

    return GlobalConfig(
        model=_dict_to_dataclass(ModelConfig, model_raw),
        generation=_dict_to_dataclass(GenerationConfig, gen_raw),
        io=_dict_to_dataclass(IOConfig, io_raw),
        runtime=_dict_to_dataclass(RuntimeConfig, runtime_raw),
        prompt=prompt_cfg,
    )


# =========================
# 3. Logging + checkpoint
# =========================

def setup_logging(log_path: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("span_batch_local")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    fh = RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(logger.level)

    ch = logging.StreamHandler()
    ch.setLevel(logger.level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def checkpoint_path(out_dir: str, model_suffix: str) -> str:
    return os.path.join(out_dir, f"checkpoint_{model_suffix}.json")

def load_checkpoint(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("next_row", 0))

def save_checkpoint(path: str, next_row: int) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"next_row": int(next_row)}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# =========================
# 4. Chargement des données
# =========================

def load_dataset(path: str, sep: str = "\t", encoding: str = "utf-8") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return pd.read_csv(path, sep=sep, encoding=encoding)


# =========================
# 5. Prompt externalisé
# =========================

def load_prompt_template(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt template introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def render_prompt(template: str, sentence: str, var_name: str = "sentence") -> str:
    try:
        return template.format(**{var_name: sentence})
    except KeyError as e:
        raise KeyError(
            f"Variable manquante dans le template. Attendu: {{{var_name}}}. Erreur: {e}"
        ) from e


# =========================
# 6. Inférence locale (Transformers)
# =========================

def _resolve_dtype(dtype_str: str):
    s = (dtype_str or "").lower()
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float32", "fp32"):
        return torch.float32
    # fallback prudent
    return torch.float16

@torch.inference_mode()
def generate_with_local_model(
    sentence: str,
    prompt_template: str,
    prompt_var: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    gen_cfg: GenerationConfig,
) -> Tuple[str, float]:
    prompt = render_prompt(prompt_template, sentence, var_name=prompt_var)

    inputs = tokenizer(prompt, return_tensors="pt")
    # envoyer sur le bon device (si device_map=auto, le modèle est sharded => on laisse HF gérer;
    # mais les inputs doivent au moins être sur un device "valide". On met sur le premier param device.)
    try:
        first_param = next(model.parameters())
        device = first_param.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except StopIteration:
        pass

    t0 = time.perf_counter()

    out_ids = model.generate(
        **inputs,
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        do_sample=gen_cfg.do_sample,
        top_p=gen_cfg.top_p,
        top_k=gen_cfg.top_k,
        repetition_penalty=gen_cfg.repetition_penalty,
        # important: évite warning si pas de pad_token
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None,
    )

    elapsed_s = time.perf_counter() - t0

    # On ne veut que la complétion (sans le prompt) si possible
    gen_part = out_ids[0]
    if "input_ids" in inputs:
        prompt_len = inputs["input_ids"].shape[-1]
        gen_part = gen_part[prompt_len:]

    text = tokenizer.decode(gen_part, skip_special_tokens=True).strip()
    return text, elapsed_s


# =========================
# 6bis. Parsing des spans
# =========================

def parse_spans_from_prediction(pred_text: str) -> List[str]:
    pred_text = (pred_text or "").strip()
    if not pred_text:
        return []
    try:
        data = json.loads(pred_text)
    except json.JSONDecodeError:
        return [pred_text]

    spans_raw = data.get("spans", [])
    spans: List[str] = []
    for item in spans_raw:
        if isinstance(item, str):
            spans.append(item)
        elif isinstance(item, dict) and isinstance(item.get("span"), str):
            spans.append(item["span"])
    return spans


# =========================
# 7. Ecriture batch append
# =========================

def append_batch_tsv(out_path: str, df_batch: pd.DataFrame, sep: str, write_header_if_new: bool) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = write_header_if_new and (not os.path.exists(out_path) or os.path.getsize(out_path) == 0)
    df_batch.to_csv(out_path, sep=sep, index=False, mode="a", header=header)


# =========================
# 8. Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Chemin du fichier JSON de config.")
    parser.add_argument("--model", type=str, default=None, help="Chemin/nom du modèle (écrase model.model_name).")
    args = parser.parse_args()

    cfg = load_config_from_json(args.config)

    if args.model is not None:
        cfg.model.model_name = args.model

    io_cfg = cfg.io
    rt_cfg = cfg.runtime
    gen_cfg = cfg.generation
    model_cfg = cfg.model

    os.makedirs(io_cfg.out_dir, exist_ok=True)

    model_suffix = os.path.basename(model_cfg.model_name.rstrip("/")).replace(":", "_").replace("-", "_").lower()
    log_path = os.path.join(io_cfg.out_dir, rt_cfg.log_file)
    logger = setup_logging(log_path, rt_cfg.log_level)

    logger.info("Config model: %s", asdict(model_cfg))
    logger.info("Config generation: %s", asdict(gen_cfg))
    logger.info("Config IO: %s", asdict(io_cfg))
    logger.info("Config runtime: %s", asdict(rt_cfg))
    logger.info("Config prompt: %s", asdict(cfg.prompt))

    # --- charger prompts ---
    prompt_var = cfg.prompt.sentence_var
    prompt_templates: List[str] = []
    prompt_names: List[str] = []

    for p in cfg.prompt.template_paths:
        prompt_templates.append(load_prompt_template(p))
        prompt_names.append(os.path.splitext(os.path.basename(p))[0])

    # --- charger dataset ---
    df = load_dataset(io_cfg.data_path, sep=io_cfg.sep, encoding=io_cfg.encoding)
    if io_cfg.sentence_col not in df.columns:
        raise ValueError(f"Colonne '{io_cfg.sentence_col}' introuvable. Dispo: {list(df.columns)}")

    # --- charger modèle local ---
    dtype = _resolve_dtype(model_cfg.dtype)

    logger.info("Loading tokenizer/model (local_files_only=%s)...", model_cfg.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name,
        local_files_only=model_cfg.local_files_only,
        trust_remote_code=model_cfg.trust_remote_code,
        use_fast=True,
    )

    # pad_token (utile pour generate)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name,
        local_files_only=model_cfg.local_files_only,
        trust_remote_code=model_cfg.trust_remote_code,
        torch_dtype=dtype,
        device_map=model_cfg.device_map,   # "auto" => accélère + shard multi-gpu si possible
    )
    model.eval()

    # --- reprise ---
    out_path = os.path.join(io_cfg.out_dir, f"spans_long_local_{model_suffix}.tsv")
    ckpt = checkpoint_path(io_cfg.out_dir, f"local_{model_suffix}")

    total = len(df)
    start_row = load_checkpoint(ckpt) if rt_cfg.resume else 0
    start_row = max(0, min(start_row, total))

    logger.info("Start_row=%d / total=%d (resume=%s)", start_row, total, rt_cfg.resume)

    batch_size = max(1, int(rt_cfg.batch_size))
    processed = start_row
    latencies_all: List[float] = []
    global_start = time.perf_counter()
    write_header_if_new = (start_row == 0)

    while processed < total:
        batch_start_idx = processed
        batch_end_idx = min(processed + batch_size, total)

        t_batch0 = time.perf_counter()
        df_batch = df.iloc[batch_start_idx:batch_end_idx].copy()
        sentences = df_batch[io_cfg.sentence_col].tolist()

        batch_preds: List[List[str]] = [[] for _ in prompt_templates]
        batch_times: List[List[float]] = [[] for _ in prompt_templates]

        for row_idx, sent in enumerate(sentences, start=batch_start_idx):
            s = str(sent)

            for p_idx, tmpl in enumerate(prompt_templates):
                pred_text, elapsed_s = generate_with_local_model(
                    sentence=s,
                    prompt_template=tmpl,
                    prompt_var=prompt_var,
                    tokenizer=tokenizer,
                    model=model,
                    gen_cfg=gen_cfg,
                )
                batch_preds[p_idx].append(pred_text)
                batch_times[p_idx].append(float(elapsed_s))
                latencies_all.append(float(elapsed_s))

            done = row_idx + 1
            remaining = total - done
            mean_s = (sum(latencies_all) / len(latencies_all)) if latencies_all else 0.0
            eta_s = remaining * mean_s

            if done % 50 == 0 or done == batch_end_idx:
                logger.info(
                    "Progress: %d/%d (remaining=%d) | avg=%.3fs/call | ETA=%.1fs",
                    done, total, remaining, mean_s, eta_s
                )

        # --- format LONG ---
        long_rows: List[Dict[str, Any]] = []
        for local_idx, sent in enumerate(sentences):
            base_row = df_batch.iloc[local_idx].to_dict()

            for p_idx, prompt_name in enumerate(prompt_names):
                pred_text = batch_preds[p_idx][local_idx]
                latency = batch_times[p_idx][local_idx]
                spans = parse_spans_from_prediction(pred_text)
                spans_count = len(spans)

                if spans_count == 0:
                    long_rows.append({
                        **base_row,
                        "model": model_cfg.model_name,
                        "prompt_name": prompt_name,
                        "prompt_index": p_idx,
                        "span_index": -1,
                        "span_text": "",
                        "spans_count": 0,
                        "raw_output": pred_text,
                        "latency_s": latency,
                    })
                else:
                    for s_idx, span_text in enumerate(spans):
                        long_rows.append({
                            **base_row,
                            "model": model_cfg.model_name,
                            "prompt_name": prompt_name,
                            "prompt_index": p_idx,
                            "span_index": s_idx,
                            "span_text": span_text,
                            "spans_count": spans_count,
                            "raw_output": pred_text,
                            "latency_s": latency,
                        })

        df_long_batch = pd.DataFrame(long_rows)
        append_batch_tsv(out_path, df_long_batch, io_cfg.sep, write_header_if_new=write_header_if_new)
        write_header_if_new = False

        processed = batch_end_idx
        save_checkpoint(ckpt, processed)

        t_batch = time.perf_counter() - t_batch0
        flat_times = [t for sub in batch_times for t in sub]
        batch_mean = (sum(flat_times) / len(flat_times)) if flat_times else 0.0

        logger.info(
            "Batch saved: rows %d..%d -> %s | batch_time=%.2fs | model_latency_mean=%.3fs",
            batch_start_idx, batch_end_idx - 1, out_path, t_batch, batch_mean
        )

    total_time = time.perf_counter() - global_start
    overall_mean = (sum(latencies_all) / len(latencies_all)) if latencies_all else 0.0

    logger.info("DONE. total_rows=%d | total_time=%.2fs | avg_latency=%.3fs", total, total_time, overall_mean)
    logger.info("Checkpoint final: %s (next_row=%d)", ckpt, total)


if __name__ == "__main__":
    main()
