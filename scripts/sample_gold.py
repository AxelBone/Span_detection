#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd


def stratified_sample_gold(
    df: pd.DataFrame,
    total_sentences: int,
    annotated_ratio: float,
    negated_ratio_within_annotated: float,
    random_state: Optional[int] = None,
    min_len_chars: Optional[int] = None,
    max_len_chars: Optional[int] = None,
) -> pd.DataFrame:
    """
    Tire un échantillon de phrases à partir du gold TSV avec :
    - total_sentences : nombre total de phrases souhaitées
    - annotated_ratio : proportion de phrases annotées (annotation=True)
    - negated_ratio_within_annotated : proportion de phrases annotées négatives
      (gold_negated=True) parmi les annotées

    L'échantillon est stratifié en 3 groupes :
    - ann_neg : annotation=True & gold_negated=True
    - ann_pos : annotation=True & gold_negated=False
    - no_ann  : annotation=False

    Quelques garde-fous :
    - si on demande plus de phrases que ce que le corpus filtré contient,
      on prend tout ce qu'il y a.
    - si on demande plus de phrases dans un groupe qu'il n'en existe,
      le surplus est redistribué aux autres groupes qui ont encore de la capacité.
    """

    df = df.copy()

    # Nettoyage / typage
    if "annotation" not in df.columns:
        raise ValueError("La colonne 'annotation' est absente du DataFrame.")
    if "gold_negated" not in df.columns:
        raise ValueError("La colonne 'gold_negated' est absente du DataFrame.")
    if "sentence" not in df.columns:
        raise ValueError("La colonne 'sentence' est absente du DataFrame.")

    # Filtre par longueur de phrase si demandé
    if min_len_chars is not None:
        df = df[df["sentence"].astype(str).str.len() >= min_len_chars]
    if max_len_chars is not None:
        df = df[df["sentence"].astype(str).str.len() <= max_len_chars]

    if len(df) == 0:
        raise ValueError("Après filtrage, le DataFrame est vide.")

    # Assurer un booléen propre pour annotation / gold_negated
    df["annotation"] = df["annotation"].astype(bool)
    # gold_negated peut être None pour les phrases non annotées
    df["gold_negated"] = df["gold_negated"].fillna(False).astype(bool)

    # Séparation en groupes
    df_ann = df[df["annotation"]]
    df_no_ann = df[~df["annotation"]]

    df_ann_neg = df_ann[df_ann["gold_negated"]]
    df_ann_pos = df_ann[~df_ann["gold_negated"]]

    # Capacités maximales
    cap = {
        "ann_neg": len(df_ann_neg),
        "ann_pos": len(df_ann_pos),
        "no_ann": len(df_no_ann),
    }

    total_available = sum(cap.values())
    if total_available == 0:
        raise ValueError("Aucune phrase disponible après filtrage pour les trois groupes.")

    # Si on demande plus que ce qui existe, on réduit
    if total_sentences > total_available:
        print(
            f"[WARN] total_sentences demandé = {total_sentences}, "
            f"mais seulement {total_available} phrases disponibles. "
            f"On prend tout le corpus filtré."
        )
        total_sentences = total_available

    # Calcul des cibles théoriques
    n_annotated = int(round(total_sentences * annotated_ratio))
    n_annotated = max(0, min(n_annotated, total_sentences))
    n_no_ann = total_sentences - n_annotated

    n_neg = int(round(n_annotated * negated_ratio_within_annotated))
    n_neg = max(0, min(n_neg, n_annotated))
    n_pos = n_annotated - n_neg

    target = {
        "ann_neg": n_neg,
        "ann_pos": n_pos,
        "no_ann": n_no_ann,
    }

    # Première passe : clamp à la capacité
    actual = {g: min(target[g], cap[g]) for g in target}
    total_actual = sum(actual.values())

    # Distribution du reste si on est en dessous de total_sentences
    remaining = total_sentences - total_actual

    # Capacité restante
    remaining_cap = {g: cap[g] - actual[g] for g in target}

    # Boucle simple de remplissage
    group_order = ["ann_neg", "ann_pos", "no_ann"]  # ordre de priorité

    while remaining > 0 and any(remaining_cap[g] > 0 for g in remaining_cap):
        for g in group_order:
            if remaining == 0:
                break
            if remaining_cap[g] > 0:
                actual[g] += 1
                remaining_cap[g] -= 1
                remaining -= 1

    # Si remaining > 0 ici, c'est qu'on n'a vraiment plus de capacité (total_available < total_sentences initial)
    # mais on a déjà ajusté total_sentences à total_available, donc normalement remaining doit être 0.

    # Debug / info
    print("[INFO] Capacités par groupe :", cap)
    print("[INFO] Cibles théoriques   :", target)
    print("[INFO] Quantités finales   :", actual)
    print("[INFO] Total final         :", sum(actual.values()))

    # Tirage aléatoire dans chaque groupe
    rng = random_state

    def sample_group(group_df: pd.DataFrame, n: int, seed_offset: int) -> pd.DataFrame:
        if n <= 0:
            return group_df.iloc[0:0]  # DF vide
        # on décale un peu la seed pour chaque groupe pour éviter les corrélations bizarres
        rs = None if rng is None else (rng + seed_offset)
        return group_df.sample(n=n, random_state=rs, replace=False)

    sampled_ann_neg = sample_group(df_ann_neg, actual["ann_neg"], seed_offset=0)
    sampled_ann_pos = sample_group(df_ann_pos, actual["ann_pos"], seed_offset=1)
    sampled_no_ann = sample_group(df_no_ann, actual["no_ann"], seed_offset=2)

    sampled = pd.concat([sampled_ann_neg, sampled_ann_pos, sampled_no_ann], axis=0)

    # Mélange global
    sampled = sampled.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Tirage stratifié d'un échantillon à partir du gold_spans.tsv"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Chemin du TSV d'entrée (gold_spans.tsv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Chemin du TSV de sortie (échantillon).",
    )
    parser.add_argument(
        "--total-sentences",
        type=int,
        required=True,
        help="Nombre total de phrases à tirer.",
    )
    parser.add_argument(
        "--annotated-ratio",
        type=float,
        required=True,
        help="Proportion de phrases annotées (0.0 - 1.0).",
    )
    parser.add_argument(
        "--negated-ratio-within-annotated",
        type=float,
        required=True,
        help="Proportion de négatives parmi les phrases annotées (0.0 - 1.0).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Seed pour rendre le tirage reproductible.",
    )
    parser.add_argument(
        "--min-len-chars",
        type=int,
        default=None,
        help="Longueur minimale de la phrase (en caractères) à inclure.",
    )
    parser.add_argument(
        "--max-len-chars",
        type=int,
        default=None,
        help="Longueur maximale de la phrase (en caractères) à inclure.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")

    sampled = stratified_sample_gold(
        df=df,
        total_sentences=args.total_sentences,
        annotated_ratio=args.annotated_ratio,
        negated_ratio_within_annotated=args.negated_ratio_within_annotated,
        random_state=args.random_state,
        min_len_chars=args.min_len_chars,
        max_len_chars=args.max_len_chars,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.output, sep="\t", index=False)

    print(f"[OK] Échantillon écrit dans : {args.output}")
    print(f"[OK] Nombre de lignes : {len(sampled)}")


if __name__ == "__main__":
    main()
