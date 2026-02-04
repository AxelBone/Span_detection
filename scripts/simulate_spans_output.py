#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simule un fichier de sortie `spans_long_<model>.tsv`
sans appel LLM, pour tests et évaluations.
"""

import random
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


# -------------------------
# Paramètres par défaut
# -------------------------

DEFAULT_PROMPTS = [
    "span_detection",
    "span_detection_with_examples",
    "negation_detection",
]

DEFAULT_SPANS = [
    "fever",
    "tachypnea",
    "vomiting",
    "abdominal pain",
    "hypotonia",
    "seizures",
    "respiratory distress",
]


# -------------------------
# Utilitaires
# -------------------------

def random_latency(mean=0.9, std=0.15, min_s=0.2) -> float:
    v = random.gauss(mean, std)
    return round(max(v, min_s), 3)


def simulate_spans(max_spans: int = 3, p_no_span: float = 0.3) -> List[str]:
    """
    Génère 0..N spans aléatoires.
    """
    if random.random() < p_no_span:
        return []

    n = random.randint(1, max_spans)
    return random.sample(DEFAULT_SPANS, k=n)


def simulate_raw_output(spans: List[str]) -> str:
    """
    Simule une sortie LLM JSON.
    """
    return json.dumps({"spans": spans}, ensure_ascii=False)


# -------------------------
# Génération principale
# -------------------------

def generate_simulated_output(
    df_input: pd.DataFrame,
    sentence_col: str,
    model_name: str,
    prompt_names: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, row in df_input.iterrows():
        sentence = str(row[sentence_col])

        for p_idx, prompt_name in enumerate(prompt_names):
            latency = random_latency()
            spans = simulate_spans()
            raw_output = simulate_raw_output(spans)
            spans_count = len(spans)

            if spans_count == 0:
                rows.append({
                    **row.to_dict(),
                    "model": model_name,
                    "prompt_name": prompt_name,
                    "prompt_index": p_idx,
                    "span_index": -1,
                    "span_text": "",
                    "spans_count": 0,
                    "raw_output": raw_output,
                    "latency_s": latency,
                })
            else:
                for s_idx, span_text in enumerate(spans):
                    rows.append({
                        **row.to_dict(),
                        "model": model_name,
                        "prompt_name": prompt_name,
                        "prompt_index": p_idx,
                        "span_index": s_idx,
                        "span_text": span_text,
                        "spans_count": spans_count,
                        "raw_output": raw_output,
                        "latency_s": latency,
                    })

    return pd.DataFrame(rows)


# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Simule un fichier de sortie spans_long_*.tsv")
    parser.add_argument("--input", required=True, help="Fichier TSV/CSV d'entrée")
    parser.add_argument("--sentence-col", default="sentence", help="Nom de la colonne phrase")
    parser.add_argument("--out", required=True, help="Fichier TSV de sortie simulée")
    parser.add_argument("--model", default="qwen3-32b-simulated", help="Nom du modèle simulé")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=DEFAULT_PROMPTS,
        help="Liste des noms de prompts",
    )
    parser.add_argument("--sep", default="\t", help="Séparateur (défaut: tab)")
    parser.add_argument("--seed", type=int, default=42, help="Seed aléatoire")

    args = parser.parse_args()
    random.seed(args.seed)

    df_input = pd.read_csv(args.input, sep=args.sep)

    if args.sentence_col not in df_input.columns:
        raise ValueError(
            f"Colonne '{args.sentence_col}' absente. Colonnes dispo: {list(df_input.columns)}"
        )

    df_out = generate_simulated_output(
        df_input=df_input,
        sentence_col=args.sentence_col,
        model_name=args.model,
        prompt_names=args.prompts,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, sep=args.sep, index=False)

    print(f"[OK] Fichier simulé écrit dans : {args.out}")
    print(f"     Lignes générées : {len(df_out)}")
    print(f"     Prompts         : {args.prompts}")


if __name__ == "__main__":
    main()
