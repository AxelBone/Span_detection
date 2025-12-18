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


# =========================
# 1. Dataclasses de config
# =========================

@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50              # optionnel côté vLLM (pas toujours supporté via API OpenAI)
    do_sample: bool = False      # si temperature==0, c'est de facto déterministe
    repetition_penalty: float = 1.0  # optionnel côté vLLM (pas toujours supporté via API OpenAI)


@dataclass
class ModelConfig:
    # Conservé pour compat avec ton JSON existant (mais non utilisé en mode vLLM)
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
    api_key: str = ""  # optionnel (si un jour tu mets un proxy/auth)


@dataclass
class GlobalConfig:
    model: ModelConfig
    generation: GenerationConfig
    io: IOConfig
    vllm: VLLMConfig


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

    # ---- Conserver ta logique de résolution des chemins IO ----
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

    # ---- (optionnel) résout model_name pour compat (même si non utilisé ici) ----
    model_name = model_raw.get("model_name")
    models_root = paths.get("models_root")
    if model_name is not None and models_root:
        if not os.path.isabs(model_name):
            model_raw["model_name"] = os.path.join(models_root, model_name)

    model_cfg = _dict_to_dataclass(ModelConfig, model_raw)
    gen_cfg = _dict_to_dataclass(GenerationConfig, gen_raw)
    io_cfg = _dict_to_dataclass(IOConfig, io_raw)
    vllm_cfg = _dict_to_dataclass(VLLMConfig, vllm_raw)

    return GlobalConfig(model=model_cfg, generation=gen_cfg, io=io_cfg, vllm=vllm_cfg)


# =========================
# 3. Chargement des données
# =========================

def load_dataset(path: str, sep: str = "\t", encoding: str = "utf-8") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return pd.read_csv(path, sep=sep, encoding=encoding)


# =========================
# 4. Construction du prompt
# =========================

def build_prompt(sentence: str) -> str:
    persona = (
        "You are an experimented clinician with an exhaustive knowledge "
        "of human phenotypes ontology."
    )

    instruction = (
        "Given a sentence, you must identify the spans related to possible "
        "phenotypes, either explicitly or implicitly.\n"
        "You should keep in the span all words related to the phenotype that "
        "should be informative (such as negation or adjective).\n"
        "You may reformulate the span if needed.\n"
        "If you don't detect any span or if you don't know, don't try to "
        "make up an answer, just write 'None'."
    )

    examples = """
SENTENCE: Shortly after birth, he developed tachypnea, irritability and spastic movements of the upper limbs, and he was found to have mild hypocalcemia and hypomagnesemia.
==========
Span: tachypnea, irritability, spastic movements of the upper limbs, mild hypocalcemia, hypomagnesemia

SENTENCE: MR spectroscopy showed a region of increased myoinositol in the left thalamus indicating gliosis with no lactate peak. His TSH has been persistently mildly elevated; however, he is not on thyroxine.
==========
Span: increased myoinositol in the left thalamus, gliosis, TSH has been persistently mildly elevated

SENTENCE: On examination, she has a nonexpressive face with subtle dysmorphism and mild positional deformity of the chest wall. His physical examination was significant for hypertelorism, long thin and hyperextensible fingers and hypotonia. Other examinations were within normal limits. Brain MRI showed diffuse white matter T2 hyperintensity.
==========
Span: nonexpressive face, subtle dysmorphism, mild positional deformity of the chest wall, hypertelorism, long fingers, thin fingers, hyperextensible fingers, hypotonia, diffuse white matter T2 hyperintensity

SENTENCE: He has developed its first seizures at the age of 9 month and continues to seize daily.
==========
Span: seizures, seize daily

SENTENCE: She can only regard faces and smile.
==========
Span: None

SENTENCE: She barely can feel pain, has no tears when she cries even though she has normal sweating.
==========
Span: barely can feel pain, no tears, normal sweating

SENTENCE: She has subtle dysmorphia characterized as hypotelorism and tapering of fingers.
==========
Span: subtle dysmorphia, hypotelorism, tapering of fingers

SENTENCE: Lysosomal enzymes in cultured skin fibroblasts such as beta-galactactosidase or total beta-hexosaminidase were within normal limits.
==========
Span: None

SENTENCE: Skeletal survey showed 11 pairs of ribs.
==========
Span: 11 pairs of ribs

SENTENCE: It showed atrophied thalami and restricted water diffusion.
==========
Span: atrophied thalami, abnormal water regulation
""".strip()

    start = f"SENTENCE: {sentence}\n==========\nSpan: "
    return f"{persona}\n\n{instruction}\n\n{examples}\n\n{start}"


# =========================
# 5. Appel vLLM (OpenAI Completions) + timing
# =========================

def _build_headers(vllm_cfg: VLLMConfig) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if vllm_cfg.api_key:
        headers["Authorization"] = f"Bearer {vllm_cfg.api_key}"
    return headers


def generate_span_for_sentence_vllm(
    sentence: str,
    vllm_cfg: VLLMConfig,
    gen_cfg: GenerationConfig,
    session: Optional[requests.Session] = None,
) -> Tuple[str, float]:
    prompt = build_prompt(sentence)

    url = f"{vllm_cfg.base_url}/completions"
    headers = _build_headers(vllm_cfg)

    # OpenAI-compatible "completions"
    payload: Dict[str, Any] = {
        "model": vllm_cfg.model,
        "prompt": prompt,
        "max_tokens": gen_cfg.max_new_tokens,
        "temperature": gen_cfg.temperature,
        "top_p": gen_cfg.top_p,
    }

    # Certains déploiements vLLM acceptent ces champs (sinon ils sont ignorés ou rejetés).
    # Décommente si tu sais que ton endpoint les supporte :
    # payload["top_k"] = gen_cfg.top_k
    # payload["repetition_penalty"] = gen_cfg.repetition_penalty

    sess = session or requests
    t0 = time.perf_counter()
    resp = sess.post(url, headers=headers, json=payload, timeout=vllm_cfg.timeout_s)
    elapsed_s = time.perf_counter() - t0

    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices", [])
    text = (choices[0].get("text") if choices else "") or ""
    text = text.strip()

    # Le prompt termine par "Span: " donc idéalement vLLM retourne directement la suite.
    # Fallback si le modèle répète "Span:"
    if "Span:" in text:
        span_text = text.split("Span:")[-1].strip()
    else:
        span_text = text

    # Normalisation légère
    span_text = span_text.strip()
    if not span_text:
        span_text = "None"

    return span_text, elapsed_s


# =========================
# 6. Négation (fix + sortie claire)
# =========================

def detect_negation(text: Any) -> bool:
    if not isinstance(text, str):
        return False

    # À adapter selon tes critères
    negation_patterns = [
        r"\bnever had\b",
        r"\bunremarkable\b",
        r"\bnormal\b",
        r"\bno\b",
        r"\bwithout\b",
        r"\bdenies\b",
        r"\bnegative for\b",
    ]
    joined = "|".join(negation_patterns)
    return re.search(joined, text, flags=re.IGNORECASE) is not None


# =========================
# 7. Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config_from_json(args.config)
    gen_cfg = cfg.generation
    io_cfg = cfg.io
    vllm_cfg = cfg.vllm

    print("Configuration génération :", asdict(gen_cfg))
    print("Configuration IO :", asdict(io_cfg))
    print("Configuration vLLM :", asdict(vllm_cfg))

    df = load_dataset(io_cfg.data_path, sep=io_cfg.sep, encoding=io_cfg.encoding)

    if io_cfg.sentence_col not in df.columns:
        raise ValueError(
            f"Colonne '{io_cfg.sentence_col}' introuvable. "
            f"Colonnes dispo : {list(df.columns)}"
        )

    # suffix basé sur le nom du modèle vLLM
    model_suffix = vllm_cfg.model.replace("-", "_").replace(" ", "_").lower()
    pred_col_name = f"{io_cfg.pred_col_prefix}_{model_suffix}"
    time_col_name = f"latency_s_{model_suffix}"

    preds: List[str] = []
    times_s: List[float] = []

    # Session HTTP pour perf (keep-alive)
    with requests.Session() as session:
        for sent in df[io_cfg.sentence_col]:
            span_pred, elapsed_s = generate_span_for_sentence_vllm(
                sentence=str(sent),
                vllm_cfg=vllm_cfg,
                gen_cfg=gen_cfg,
                session=session,
            )
            preds.append(span_pred)
            times_s.append(elapsed_s)

    df[pred_col_name] = preds
    df[time_col_name] = times_s

    # Negation (sur la phrase)
    df["negation"] = [detect_negation(sent) for sent in df[io_cfg.sentence_col]]

    # Stats simples
    s = pd.Series(times_s, dtype="float64")
    print("\n--- Timing par requête (secondes) ---")
    print(
        f"n={len(s)} | mean={s.mean():.4f} | median={s.median():.4f} | "
        f"p95={s.quantile(0.95):.4f} | max={s.max():.4f}"
    )

    os.makedirs(io_cfg.out_dir, exist_ok=True)
    out_path = os.path.join(io_cfg.out_dir, f"spans_{model_suffix}.tsv")
    df.to_csv(out_path, sep=io_cfg.sep, index=False)

    print(f"Résultats sauvegardés dans : {out_path}")
    print(f"Hyperparamètres de génération : {asdict(gen_cfg)}")


if __name__ == "__main__":
    main()
