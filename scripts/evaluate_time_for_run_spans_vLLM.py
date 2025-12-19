#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd
import re
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
    model_name: str = "models/meditron-7b"
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


@dataclass
class VLLMConfig:
    base_url: str = "http://lx181:9502/v1"
    model: str = "qwen3-32b"
    timeout_s: float = 120.0
    api_key: str = ""


@dataclass
class RuntimeConfig:
    # batch = fréquence de sauvegarde (checkpoint + append TSV)
    batch_size: int = 100
    # log
    log_file: str = "run.log"
    log_level: str = "INFO"
    # reprise
    resume: bool = True


@dataclass
class PromptConfig:
    # chemin du template prompt (fichier texte)
    template_path: str = "prompts/span_prompt.txt"
    # nom de la variable utilisée dans le template, ex: {sentence}
    sentence_var: str = "sentence"


@dataclass
class GlobalConfig:
    model: ModelConfig
    generation: GenerationConfig
    io: IOConfig
    vllm: VLLMConfig
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
        return value

    model_raw = expand(raw.get("model", {}))
    gen_raw = expand(raw.get("generation", {}))
    io_raw = expand(raw.get("io", {}))
    vllm_raw = expand(raw.get("vllm", {}))
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
    vllm_cfg = _dict_to_dataclass(VLLMConfig, vllm_raw)
    runtime_cfg = _dict_to_dataclass(RuntimeConfig, runtime_raw)
    prompt_cfg = _dict_to_dataclass(PromptConfig, prompt_raw)

    return GlobalConfig(
        model=model_cfg,
        generation=gen_cfg,
        io=io_cfg,
        vllm=vllm_cfg,
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


def render_prompt(template: str, sentence: str, var_name: str = "sentence") -> str:
    try:
        return template.format(**{var_name: sentence})
    except KeyError as e:
        raise KeyError(
            f"Variable manquante dans le template. Attendu: {{{var_name}}}. Erreur: {e}"
        ) from e


# =========================
# 6. Appel vLLM
# =========================

def _build_headers(vllm_cfg: VLLMConfig) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if vllm_cfg.api_key:
        headers["Authorization"] = f"Bearer {vllm_cfg.api_key}"
    return headers


def generate_span_for_sentence_vllm(
    sentence: str,
    prompt_template: str,
    prompt_var: str,
    vllm_cfg: VLLMConfig,
    gen_cfg: GenerationConfig,
    session: Optional[requests.Session] = None,
) -> Tuple[str, float]:
    prompt = render_prompt(prompt_template, sentence, var_name=prompt_var)

    url = f"{vllm_cfg.base_url}/completions"
    headers = _build_headers(vllm_cfg)

    payload: Dict[str, Any] = {
        "model": vllm_cfg.model,
        "prompt": prompt,
        "max_tokens": gen_cfg.max_new_tokens,
        "temperature": gen_cfg.temperature,
        "top_p": gen_cfg.top_p,
    }

    sess = session or requests
    t0 = time.perf_counter()
    resp = sess.post(url, headers=headers, json=payload, timeout=vllm_cfg.timeout_s)
    elapsed_s = time.perf_counter() - t0

    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices", [])
    text = (choices[0].get("text") if choices else "") or ""
    text = text.strip()

    # nettoyage léger
    if "Span:" in text:
        span_text = text.split("Span:")[-1].strip()
    else:
        span_text = text.strip()

    if not span_text:
        span_text = "None"

    return span_text, elapsed_s


# =========================
# 7. Négation
# =========================

def detect_negation(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    negation_patterns = [
        r"\bnever had\b",
        r"\bunremarkable\b",
        r"\bnormal\b",
        r"\bno\b",
        r"\bwithout\b",
        r"\bdenies\b",
        r"\bnegative for\b",
    ]
    return re.search("|".join(negation_patterns), text, flags=re.IGNORECASE) is not None


# =========================
# 8. Ecriture batch append
# =========================

def append_batch_tsv(
    out_path: str,
    df_batch: pd.DataFrame,
    sep: str,
    write_header_if_new: bool,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "a"
    header = write_header_if_new and (not os.path.exists(out_path) or os.path.getsize(out_path) == 0)
    df_batch.to_csv(out_path, sep=sep, index=False, mode=mode, header=header)


# =========================
# 9. Main avec logs + reprise
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config_from_json(args.config)
    gen_cfg = cfg.generation
    io_cfg = cfg.io
    vllm_cfg = cfg.vllm
    rt_cfg = cfg.runtime

    model_suffix = vllm_cfg.model.replace("-", "_").replace(" ", "_").lower()
    os.makedirs(io_cfg.out_dir, exist_ok=True)

    log_path = os.path.join(io_cfg.out_dir, rt_cfg.log_file)
    logger = setup_logging(log_path, rt_cfg.log_level)

    logger.info("Config generation: %s", asdict(gen_cfg))
    logger.info("Config IO: %s", asdict(io_cfg))
    logger.info("Config vLLM: %s", asdict(vllm_cfg))
    logger.info("Config runtime: %s", asdict(rt_cfg))
    logger.info("Config prompt: %s", asdict(cfg.prompt))

    # --- Charger template prompt une seule fois ---
    prompt_template = load_prompt_template(cfg.prompt.template_path)
    prompt_var = cfg.prompt.sentence_var
    logger.info("Prompt template loaded: %s (var={%s})", cfg.prompt.template_path, prompt_var)

    df = load_dataset(io_cfg.data_path, sep=io_cfg.sep, encoding=io_cfg.encoding)

    if io_cfg.sentence_col not in df.columns:
        raise ValueError(
            f"Colonne '{io_cfg.sentence_col}' introuvable. Colonnes dispo : {list(df.columns)}"
        )

    pred_col_name = f"{io_cfg.pred_col_prefix}_{model_suffix}"
    time_col_name = f"latency_s_{model_suffix}"
    neg_col_name = "negation"

    out_path = os.path.join(io_cfg.out_dir, f"spans_{model_suffix}.tsv")
    ckpt = checkpoint_path(io_cfg.out_dir, model_suffix)

    total = len(df)
    start_row = load_checkpoint(ckpt) if rt_cfg.resume else 0
    start_row = max(0, min(start_row, total))

    if start_row > 0:
        logger.info("Reprise activée: start_row=%d / total=%d (checkpoint=%s)", start_row, total, ckpt)
    else:
        logger.info("Départ: start_row=0 / total=%d", total)

    batch_size = max(1, int(rt_cfg.batch_size))

    processed = start_row
    global_start = time.perf_counter()
    latencies_all: List[float] = []

    with requests.Session() as session:
        while processed < total:
            batch_start_idx = processed
            batch_end_idx = min(processed + batch_size, total)

            t_batch0 = time.perf_counter()
            df_batch = df.iloc[batch_start_idx:batch_end_idx].copy()

            preds: List[str] = []
            times_s: List[float] = []
            negs: List[bool] = []

            for i, sent in enumerate(df_batch[io_cfg.sentence_col].tolist(), start=batch_start_idx):
                span_pred, elapsed_s = generate_span_for_sentence_vllm(
                    sentence=str(sent),
                    prompt_template=prompt_template,
                    prompt_var=prompt_var,
                    vllm_cfg=vllm_cfg,
                    gen_cfg=gen_cfg,
                    session=session,
                )

                preds.append(span_pred)
                times_s.append(float(elapsed_s))
                negs.append(detect_negation(sent))
                latencies_all.append(float(elapsed_s))

                done = (i + 1)
                remaining = total - done
                mean_s = (sum(latencies_all) / len(latencies_all)) if latencies_all else 0.0
                eta_s = remaining * mean_s

                if done % 50 == 0 or done == batch_end_idx:
                    logger.info(
                        "Progress: %d/%d (remaining=%d) | avg=%.3fs/line | ETA=%.1fs",
                        done, total, remaining, mean_s, eta_s
                    )

            df_batch[pred_col_name] = preds
            df_batch[time_col_name] = times_s
            df_batch[neg_col_name] = negs

            append_batch_tsv(out_path, df_batch, io_cfg.sep, write_header_if_new=True)

            processed = batch_end_idx
            save_checkpoint(ckpt, processed)

            t_batch = time.perf_counter() - t_batch0
            batch_mean = (sum(times_s) / len(times_s)) if times_s else 0.0

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
