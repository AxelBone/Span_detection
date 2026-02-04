#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import spacy
import re


# Chemins à adapter
ANNOT_DIR = Path("data/CHU_50_eng/annotations")   # dossier des JSON annotés
TEXT_DIR = Path("data/CHU_50_eng/txt")              # dossier des .txt bruts
OUTPUT_TSV = Path("data/gold_spans.tsv")   # fichier de sortie

def load_french_nlp():
    """Charge le modèle français spaCy."""
    # python -m spacy download fr_core_news_sm
    return spacy.load("fr_core_news_sm")

def load_english_nlp():
    """Charge le modèle anglais spaCy."""
    # python -m spacy download en_core_web_sm
    return spacy.load("en_core_web_sm")


def get_sentences_with_offsets(nlp, text: str) -> List[Dict[str, Any]]:
    """
    Segmente le texte en phrases et renvoie une liste de dict :
      {
        "sentence_id": int,
        "text": str (exact slice du texte brut),
        "start_char": int,
        "end_char": int
      }
    """
    doc = nlp(text)
    sents = []
    for i, sent in enumerate(doc.sents):
        raw = text[sent.start_char:sent.end_char]  # fidèle au texte brut
        sents.append({
            "sentence_id": i,
            "text": raw,
            "start_char": sent.start_char,
            "end_char": sent.end_char,
        })
    return sents


def clean_sentence_for_csv(text: str) -> str:
    """Supprime retours chariot, tabs, espaces multiples, pour le CSV."""
    if text is None:
        return ""
    # remplace toute forme de whitespace (y compris \n, \t, etc.) par un espace
    return re.sub(r"\s+", " ", text).strip()


def find_sentence_for_span(
    sentences: List[Dict[str, Any]],
    span_start: int
) -> Optional[Dict[str, Any]]:
    """
    Trouve la phrase qui contient l'offset span_start (global, dans le texte entier).
    Condition: start_char <= span_start < end_char.
    Renvoie le dict de la phrase ou None si non trouvée.
    """
    for sent in sentences:
        if sent["start_char"] <= span_start < sent["end_char"]:
            return sent
    return None


def extract_hpo_id(ann: Dict[str, Any]) -> Optional[str]:
    """
    Récupère le premier hpoId si présent dans ann["hpoAnnotation"].
    """
    hpo_list = ann.get("hpoAnnotation", [])
    if isinstance(hpo_list, list) and len(hpo_list) > 0 and isinstance(hpo_list[0], dict):
        return hpo_list[0].get("hpoId")
    return None


def process_one_report(
    nlp,
    annot_path: Path,
    text_path: Path
) -> List[Dict[str, Any]]:
    """
    Traite un couple (JSON d'annotation, texte brut) et renvoie
    une liste de lignes pour le tableau d'évaluation.

    Nouvelle logique :
    - toutes les phrases du texte apparaissent au moins une fois
    - si une phrase n'a aucune annotation -> 1 ligne avec annotation=False, champs gold_* vides
    - si une phrase a k annotations -> k lignes avec annotation=True
    """
    rows: List[Dict[str, Any]] = []

    report_id = annot_path.stem

    # Charge le texte brut
    try:
        text = text_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = text_path.read_text(encoding="latin-1")

    # Segmentation en phrases
    sentences = get_sentences_with_offsets(nlp, text)

    # Charge les annotations
    data = json.loads(annot_path.read_text(encoding="utf-8"))
    annotations = data.get("annotations", [])

    # On prépare un mapping sentence_id -> liste d'annotations
    sent_to_anns: Dict[int, List[Dict[str, Any]]] = {s["sentence_id"]: [] for s in sentences}

    for ann in annotations:
        global_start = ann.get("start")
        if global_start is None:
            continue

        sent = find_sentence_for_span(sentences, global_start)
        if sent is None:
            # Annotation orpheline (pas de phrase correspondante) -> on peut l'ignorer ou logger
            # Ici on ignore silencieusement
            continue

        sentence_id = sent["sentence_id"]
        sent_to_anns.setdefault(sentence_id, []).append(ann)

    # Maintenant, on parcourt toutes les phrases
    for sent in sentences:
        sentence_id = sent["sentence_id"]
        raw_sentence = sent["text"]
        clean_sentence = clean_sentence_for_csv(raw_sentence)

        anns_for_sent = sent_to_anns.get(sentence_id, [])

        if not anns_for_sent:
            # Aucune annotation sur cette phrase -> 1 ligne "vide"
            rows.append({
                "report_id": report_id,
                "sentence_id": sentence_id,
                "sentence": clean_sentence,
                "gold_span_text": None,
                "start": None,
                "length": None,
                "gold_negated": None,
                "hpo_id": None,
                "annotation": False,  # flag demandé
            })
        else:
            # Au moins une annotation -> 1 ligne par annotation
            for ann in anns_for_sent:
                global_start = ann.get("start")
                length = ann.get("length")
                gold_span_text = ann.get("sentence", "").strip()
                gold_negated = bool(ann.get("negated", False))
                hpo_id = extract_hpo_id(ann)

                local_start = None
                if global_start is not None:
                    local_start = global_start - sent["start_char"]

                rows.append({
                    "report_id": report_id,
                    "sentence_id": sentence_id,
                    "sentence": clean_sentence,
                    "gold_span_text": gold_span_text if gold_span_text else None,
                    "start": local_start,
                    "length": length,
                    "gold_negated": gold_negated,
                    "hpo_id": hpo_id,
                    "gold_annotation": True,   # il y a une annotation sur cette phrase
                })

    return rows


def main():
    nlp = load_english_nlp()
    all_rows: List[Dict[str, Any]] = []

    for annot_file in sorted(ANNOT_DIR.glob("*.json")):
        text_file = TEXT_DIR / f"{annot_file.stem}.txt"
        if not text_file.exists():
            print(f"[WARN] Texte manquant pour {annot_file.name} -> {text_file} introuvable, on skip.")
            continue

        rows = process_one_report(nlp, annot_file, text_file)
        all_rows.extend(rows)
        print(f"[OK] {annot_file.name}: {len(rows)} lignes générées (phrases + annotations)")

    df = pd.DataFrame(all_rows)

    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_TSV, sep="\t", index=False)

    print(f"\n[OK] Gold standard écrit dans : {OUTPUT_TSV}")
    print(f"[OK] Nombre total de lignes : {len(df)}")


if __name__ == "__main__":
    main()
