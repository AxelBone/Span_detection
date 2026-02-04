#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd
import time
import requests
import logging
from logging.handlers import RotatingFileHandler


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
    # Gardé pour compat, peu utilisé avec Ollama
    model_name: str = "qwen3-32b"
    device_map: str = "auto"
    dtype: str = "float16"
    local_files_only: bool = False
    trust_remote_code: bool = False


@dataclass
class IOConfig:
    data_path: str = "data/spans_dataset.tsv"
    sep: str = "\t"
    encoding: str = "utf-8"
    sentence_col: str = "Sentence_en"
    out_dir: str = "results"
    pred_col_prefix: str = "span_pred"
    gold_span_col: str = "gold_span_text"   # <- ajouté


@dataclass
class OllamaConfig:
    base_url: str = "https://compute-01.odh.local/ollama"
    model: str = "deepseek-r1:8b-llama-distill-q4_K_M"
    timeout_s: float = 120.0
    verify_ssl: bool = False


@dataclass
class RuntimeConfig:
    batch_size: int = 100
    log_file: str = "run.log"
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
    ollama: OllamaConfig
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
    ollama_raw = expand(raw.get("ollama", raw.get("vllm", {})))
    runtime_raw = expand(raw.get("runtime", {}))
    prompt_raw = expand(raw.get("prompt", {}))

    # --- résolution chemins IO ---
    filename = io_raw.get("filename")
    data_root = paths.get("data_root")
    if filename is not None:
        if data_root and not os.path.isabs(filename):
            io_raw["data_path"] = os.path.join(data_root, filename)
        else:
            io_raw["data_path"] = filename

    out_dir = io_raw.get("out_dir", "results")
    project_root = paths.get("project_root")
    if out_dir is not None and project_root and not os.path.isabs(out_dir):
        io_raw["out_dir"] = os.path.join(project_root, out_dir)

    model_name = model_raw.get("model_name")
    models_root = paths.get("models_root")
    if model_name is not None and models_root and not os.path.isabs(model_name):
        model_raw["model_name"] = os.path.join(models_root, model_name)

    model_cfg = _dict_to_dataclass(ModelConfig, model_raw)
    gen_cfg = _dict_to_dataclass(GenerationConfig, gen_raw)
    io_cfg = _dict_to_dataclass(IOConfig, io_raw)
    ollama_cfg = _dict_to_dataclass(OllamaConfig, ollama_raw)
    runtime_cfg = _dict_to_dataclass(RuntimeConfig, runtime_raw)
    prompt_cfg = _dict_to_dataclass(PromptConfig, prompt_raw)

    if (not prompt_cfg.template_paths or len(prompt_cfg.template_paths) == 0) and prompt_cfg.template_path:
        prompt_cfg.template_paths = [prompt_cfg.template_path]

    if not prompt_cfg.template_paths:
        raise ValueError("Aucun prompt n'est défini (prompt.template_paths est vide).")

    return GlobalConfig(
        model=model_cfg,
        generation=gen_cfg,
        io=io_cfg,
        ollama=ollama_cfg,
        runtime=runtime_cfg,
        prompt=prompt_cfg,
    )


# =========================
# 3. Logging + checkpoint
# =========================

def setup_logging(log_path: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("span_batch")
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


def render_prompt_with_vars(template: str, vars_dict: Dict[str, str]) -> str:
    try:
        return template.format(**vars_dict)
    except KeyError as e:
        raise KeyError(
            f"Variable manquante dans le template: {e}. Vars dispo: {list(vars_dict.keys())}"
        ) from e


# =========================
# 6. Appel Ollama
# =========================

def generate_with_ollama(
    prompt: str,
    ollama_cfg: OllamaConfig,
    gen_cfg: GenerationConfig,
    session: Optional[requests.Session] = None,
) -> Tuple[str, float]:
    """
    Appelle l'API Ollama /api/generate pour un prompt donné.
    """
    url = f"{ollama_cfg.base_url}/api/generate"
    headers = {"Content-Type": "application/json"}

    payload: Dict[str, Any] = {
        "model": ollama_cfg.model,
        "prompt": prompt,
        "stream": False,
        "temperature": gen_cfg.temperature,
        "max_tokens": gen_cfg.max_new_tokens,
    }

    sess = session or requests
    t0 = time.perf_counter()
    resp = sess.post(
        url,
        headers=headers,
        json=payload,
        timeout=ollama_cfg.timeout_s,
        verify=ollama_cfg.verify_ssl,
    )
    elapsed_s = time.perf_counter() - t0

    resp.raise_for_status()
    data = resp.json()

    text = data.get("response", "") or ""
    text = text.strip()
    return text, elapsed_s


# =========================
# 6bis. Parsing sorties
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


def parse_negation_from_prediction(pred_text: str) -> Optional[bool]:
    """
    Attendu: {"negation": true|false}
    Retourne None si parsing impossible.
    """
    pred_text = (pred_text or "").strip()
    if not pred_text:
        return None
    try:
        data = json.loads(pred_text)
    except json.JSONDecodeError:
        return None
    val = data.get("negation", None)
    return val if isinstance(val, bool) else None


# =========================
# 7. Ecriture batch append
# =========================

def append_batch_tsv(out_path: str, df_batch: pd.DataFrame, sep: str, write_header_if_new: bool) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = write_header_if_new and (not os.path.exists(out_path) or os.path.getsize(out_path) == 0)
    df_batch.to_csv(out_path, sep=sep, index=False, mode="a", header=header)


# =========================
# 8. Main (format long)
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Chemin du fichier JSON de config.")
    parser.add_argument("--model", type=str, default=None, help="Nom du modèle Ollama (écrase ollama.model).")
    args = parser.parse_args()

    cfg = load_config_from_json(args.config)
    gen_cfg = cfg.generation
    io_cfg = cfg.io
    ollama_cfg = cfg.ollama
    rt_cfg = cfg.runtime

    if args.model is not None:
        ollama_cfg.model = args.model

    # prompts negation-only (basés sur basename)
    NEGATION_ONLY_PROMPTS = {"negation_detection", "negation_detection_with_examples"}

    model_suffix = ollama_cfg.model.replace(":", "_").replace("-", "_").replace(" ", "_").lower()
    os.makedirs(io_cfg.out_dir, exist_ok=True)

    log_path = os.path.join(io_cfg.out_dir, rt_cfg.log_file)
    logger = setup_logging(log_path, rt_cfg.log_level)

    logger.info("Config generation: %s", asdict(gen_cfg))
    logger.info("Config IO: %s", asdict(io_cfg))
    logger.info("Config Ollama: %s", asdict(ollama_cfg))
    logger.info("Config runtime: %s", asdict(rt_cfg))
    logger.info("Config prompt: %s", asdict(cfg.prompt))

    # --- prompts ---
    sentence_var = cfg.prompt.sentence_var
    prompt_templates: List[str] = []
    prompt_names: List[str] = []

    for p in cfg.prompt.template_paths:
        prompt_templates.append(load_prompt_template(p))
        prompt_names.append(os.path.splitext(os.path.basename(p))[0])

    logger.info("Loaded %d prompt templates: %s", len(prompt_templates), cfg.prompt.template_paths)

    df = load_dataset(io_cfg.data_path, sep=io_cfg.sep, encoding=io_cfg.encoding)

    if io_cfg.sentence_col not in df.columns:
        raise ValueError(f"Colonne '{io_cfg.sentence_col}' introuvable. Colonnes dispo : {list(df.columns)}")

    # colonne gold span (pour prompts negation-only)
    gold_span_col = io_cfg.gold_span_col

    if any(name in NEGATION_ONLY_PROMPTS for name in prompt_names):
        if gold_span_col not in df.columns:
            raise ValueError(
                f"Prompts negation-only détectés {NEGATION_ONLY_PROMPTS} "
                f"mais aucune colonne gold span trouvée ('{io_cfg.gold_span_col}')."
            )

    out_path = os.path.join(io_cfg.out_dir, f"spans_long_{model_suffix}.tsv")
    ckpt = checkpoint_path(io_cfg.out_dir, model_suffix)

    total = len(df)
    start_row = load_checkpoint(ckpt) if rt_cfg.resume else 0
    start_row = max(0, min(start_row, total))

    logger.info("Start_row=%d / total=%d (resume=%s)", start_row, total, rt_cfg.resume)

    batch_size = max(1, int(rt_cfg.batch_size))
    processed = start_row
    global_start = time.perf_counter()
    latencies_all: List[float] = []
    write_header_if_new = (start_row == 0)

    with requests.Session() as session:
        while processed < total:
            batch_start_idx = processed
            batch_end_idx = min(processed + batch_size, total)

            t_batch0 = time.perf_counter()
            df_batch = df.iloc[batch_start_idx:batch_end_idx].copy()

            sentences = df_batch[io_cfg.sentence_col].tolist()
            gold_spans_batch = df_batch[gold_span_col].tolist() if gold_span_col in df_batch.columns else None

            batch_outputs: List[List[str]] = [[] for _ in prompt_templates]
            batch_times: List[List[float]] = [[] for _ in prompt_templates]

            for local_i, sent in enumerate(sentences):
                sentence_value = str(sent)
                span_value = ""
                if gold_spans_batch is not None:
                    span_value = "" if pd.isna(gold_spans_batch[local_i]) else str(gold_spans_batch[local_i])

                for p_idx, tmpl in enumerate(prompt_templates):
                    prompt_name = prompt_names[p_idx]

                    if prompt_name in NEGATION_ONLY_PROMPTS:
                        vars_dict = {"span": span_value}
                    else:
                        vars_dict = {sentence_var: sentence_value}

                    prompt = render_prompt_with_vars(tmpl, vars_dict)

                    pred_text, elapsed_s = generate_with_ollama(
                        prompt=prompt,
                        ollama_cfg=ollama_cfg,
                        gen_cfg=gen_cfg,
                        session=session,
                    )

                    batch_outputs[p_idx].append(pred_text)
                    batch_times[p_idx].append(float(elapsed_s))
                    latencies_all.append(float(elapsed_s))

                done = batch_start_idx + local_i + 1
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

            for local_idx in range(len(df_batch)):
                base_row = df_batch.iloc[local_idx].to_dict()

                for p_idx, prompt_name in enumerate(prompt_names):
                    pred_text = batch_outputs[p_idx][local_idx]
                    latency = batch_times[p_idx][local_idx]

                    if prompt_name in NEGATION_ONLY_PROMPTS:
                        neg_val = parse_negation_from_prediction(pred_text)
                        long_rows.append({
                            **base_row,
                            "model": ollama_cfg.model,
                            "prompt_name": prompt_name,
                            "prompt_index": p_idx,
                            "span_index": -1,
                            "span_text": "",
                            "spans_count": 0,
                            "negation_pred": neg_val,
                            "raw_output": pred_text,
                            "latency_s": latency,
                        })
                    else:
                        spans = parse_spans_from_prediction(pred_text)
                        spans_count = len(spans)

                        if spans_count == 0:
                            long_rows.append({
                                **base_row,
                                "model": ollama_cfg.model,
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
                                    "model": ollama_cfg.model,
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
