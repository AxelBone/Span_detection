# %% [markdown]
# # Evaluation multi-modèles de détection de spans
#
# - Entrée : fichiers de prédictions au format "long" (une ligne = 1 span prédit)
# - Gold : colonnes gold_* déjà présentes dans ces fichiers
# - Span-level : Levenshtein normalisé + seuil
# - Agrégation : par modèle et par (modèle, prompt)

# %%
import os
import glob
import re
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# =========================
# 1. Config
# =========================

# Motif des fichiers de sortie des différents modèles
PREDICTION_PATTERN = "results/spans_long_*.tsv"  # à adapter si besoin

# Séparateur des TSV ("," si tu as mis sep="," dans ton script)
SEP = ","   # ou "\t"

# Seuil de similarité Levenshtein normalisée pour considérer un match comme TP
LEV_SIM_THRESHOLD = 0.8

# Colonnes clés pour identifier une phrase
KEY_COLS_SENTENCE = ["report_id", "sentence_id"]

# Nom de la colonne texte de la phrase (si tu veux l'afficher plus tard)
SENTENCE_COL_CANDIDATES = ["Sentence_en", "sentence", "Sentence"]

# Nom de la colonne gold des spans
GOLD_SPAN_COL = "gold_span_text"

# Nom de la colonne prédiction des spans
PRED_SPAN_COL = "span_text"

# %%
# =========================
# 2. Chargement des prédictions multi-modèles
# =========================

files = sorted(glob.glob(PREDICTION_PATTERN))
print(f"Nb de fichiers de prédiction trouvés : {len(files)}")
for f in files:
    print(" -", f)

if not files:
    raise RuntimeError("Aucun fichier de prédiction trouvé : vérifie PREDICTION_PATTERN.")

dfs = []
for f in files:
    df_tmp = pd.read_csv(f, sep=SEP)
    df_tmp["source_file"] = os.path.basename(f)
    dfs.append(df_tmp)

df_pred = pd.concat(dfs, ignore_index=True)
print("Shape totale des prédictions :", df_pred.shape)
df_pred.head()

# %%
# =========================
# 3. Vérifications de colonnes
# =========================

required_cols = set(KEY_COLS_SENTENCE + [GOLD_SPAN_COL, PRED_SPAN_COL, "model", "prompt_name", "span_index"])
missing = required_cols - set(df_pred.columns)
if missing:
    raise ValueError(f"Colonnes manquantes dans df_pred : {missing}")

# colonne phrase (optionnelle mais pratique)
sentence_col = None
for c in SENTENCE_COL_CANDIDATES:
    if c in df_pred.columns:
        sentence_col = c
        break
print("Colonne phrase utilisée :", sentence_col)

# %%
# =========================
# 4. Normalisation de texte & Levenshtein
# =========================

def normalize_span(text: Any) -> str:
    """
    Normalisation légère d'un span :
    - lower
    - strip
    - réduction des espaces
    - suppression ponctuation simple en bordure
    """
    if pd.isna(text):
        return ""
    s = str(text)
    s = s.lower()
    s = s.strip()
    # espaces multiples -> un seul
    s = re.sub(r"\s+", " ", s)
    # enlever ponctuation simple aux extrémités
    s = s.strip(",.;:()[]{}\"'")
    return s

def levenshtein_distance(a: str, b: str) -> int:
    """
    Distance de Levenshtein (édition) sur caractères.
    Implémentation simple O(len(a)*len(b)).
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    # DP sur une seule dimension
    prev_row = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = curr_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (ca != cb)
            curr_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = curr_row
    return prev_row[-1]

def levenshtein_similarity(a: str, b: str) -> float:
    """
    Similarité Levenshtein normalisée dans [0,1].
    sim = 1 - lev / max_len
    """
    a_norm = normalize_span(a)
    b_norm = normalize_span(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    d = levenshtein_distance(a_norm, b_norm)
    max_len = max(len(a_norm), len(b_norm))
    return 1.0 - d / max_len

# Petit test rapide
print("sim('hypotonie', 'hypotonia') =", levenshtein_similarity("hypotonie", "hypotonia"))

# %%
# =========================
# 5. Construction des gold spans par phrase
# =========================

# On suppose que les colonnes gold_* sont déjà présentes dans df_pred (copiées du dataset d'origine)
gold_df = df_pred[KEY_COLS_SENTENCE + [GOLD_SPAN_COL]].drop_duplicates()

# Normalisation et filtrage des gold spans non vides
gold_df["gold_span_norm"] = gold_df[GOLD_SPAN_COL].apply(normalize_span)

def collect_gold_spans(group: pd.DataFrame) -> List[str]:
    spans = [s for s in group["gold_span_norm"].tolist() if s]
    # uniq pour éviter les doublons
    return sorted(set(spans))

gold_by_sent: Dict[Tuple, List[str]] = (
    gold_df
    .groupby(KEY_COLS_SENTENCE, dropna=False)
    .apply(collect_gold_spans)
    .to_dict()
)

nb_sents_with_gold = sum(1 for v in gold_by_sent.values() if len(v) > 0)
print("Nb total de phrases :", len(gold_by_sent))
print("Nb de phrases avec au moins un gold span :", nb_sents_with_gold)

# %%
# =========================
# 6. Construction des spans prédits par phrase / modèle / prompt
# =========================

# Pour les spans prédits, on ignore les lignes sentinel avec span_index = -1
df_pred_spans = df_pred[df_pred["span_index"] >= 0].copy()
df_pred_spans["pred_span_norm"] = df_pred_spans[PRED_SPAN_COL].apply(normalize_span)

def collect_pred_spans(group: pd.DataFrame) -> List[str]:
    spans = [s for s in group["pred_span_norm"].tolist() if s]
    return spans  # on ne déduplique pas forcément ici

# Dict: (model, prompt_name, report_id, sentence_id) -> [pred spans]
pred_by_model_prompt_sent: Dict[Tuple, List[str]] = (
    df_pred_spans
    .groupby(["model", "prompt_name"] + KEY_COLS_SENTENCE, dropna=False)
    .apply(collect_pred_spans)
    .to_dict()
)

len(pred_by_model_prompt_sent)

# %%
# =========================
# 7. Fonction de comptage TP / FP / FN (span-level, Levenshtein)
# =========================

def count_tp_fp_fn_for_sentence(
    gold_spans: List[str],
    pred_spans: List[str],
    threshold: float,
) -> Tuple[int, int, int]:
    """
    Compte TP, FP, FN pour une phrase donnée (ensemble de gold_spans et de pred_spans)
    avec matching basé sur similarité Levenshtein >= threshold.

    NB : pas de matching one-to-one (Hungarian) => une même gold peut être utilisée
    pour plusieurs prédictions (on suit ta consigne "no Hungarian").
    """
    tp = 0
    fp = 0
    fn = 0

    # TP / FP (côté prédictions)
    for pred in pred_spans:
        if gold_spans and any(levenshtein_similarity(pred, g) >= threshold for g in gold_spans):
            tp += 1
        else:
            fp += 1

    # FN (côté gold)
    for gold in gold_spans:
        if not pred_spans or all(levenshtein_similarity(pred, gold) < threshold for pred in pred_spans):
            fn += 1

    return tp, fp, fn

# %%
# =========================
# 8. Evaluation span-level par (modèle, prompt)
# =========================

def evaluate_span_level_for_model_prompt(
    df: pd.DataFrame,
    gold_by_sent: Dict[Tuple, List[str]],
    threshold: float = 0.8,
) -> pd.DataFrame:
    """
    df : dataFrame complet des prédictions (multi-modèles)
    gold_by_sent : dict (report_id, sentence_id) -> [gold spans]
    Retourne un DataFrame de métriques par (model, prompt_name).
    """

    results = []

    # On re-filtre les spans prédits ici pour être sûr
    df_spans = df[df["span_index"] >= 0].copy()
    df_spans["pred_span_norm"] = df_spans[PRED_SPAN_COL].apply(normalize_span)

    # Boucle sur chaque (modèle, prompt)
    for (model, prompt_name), df_mp in df_spans.groupby(["model", "prompt_name"]):
        # dict local : (report_id, sentence_id) -> [pred spans]
        local_pred_by_sent = (
            df_mp
            .groupby(KEY_COLS_SENTENCE, dropna=False)["pred_span_norm"]
            .apply(list)
            .to_dict()
        )

        tp_total, fp_total, fn_total = 0, 0, 0

        # On considère toutes les phrases présentes dans le gold (pour comptage FN),
        # même si le modèle n'a rien prédit dessus
        for sent_key, gold_spans in gold_by_sent.items():
            pred_spans = local_pred_by_sent.get(sent_key, [])
            tp, fp, fn = count_tp_fp_fn_for_sentence(gold_spans, pred_spans, threshold)
            tp_total += tp
            fp_total += fp
            fn_total += fn

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            "model": model,
            "prompt_name": prompt_name,
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return pd.DataFrame(results)

df_metrics_mp = evaluate_span_level_for_model_prompt(df_pred, gold_by_sent, threshold=LEV_SIM_THRESHOLD)
df_metrics_mp.sort_values("f1", ascending=False)

# %%
# =========================
# 9. Agrégation par modèle (en moyennant sur les prompts)
# =========================

# Agrégation simple : on recompute TP/FP/FN par modèle (somme sur prompts)
agg = (
    df_metrics_mp
    .groupby("model", as_index=False)[["tp", "fp", "fn"]]
    .sum()
)

agg["precision"] = agg["tp"] / (agg["tp"] + agg["fp"])
agg["recall"] = agg["tp"] / (agg["tp"] + agg["fn"])
agg["f1"] = 2 * agg["precision"] * agg["recall"] / (agg["precision"] + agg["recall"])

df_metrics_model = agg.sort_values("f1", ascending=False)
df_metrics_model

# %%
# =========================
# 10. Visualisation : F1 par modèle
# =========================

plt.figure(figsize=(8, 4))
order = df_metrics_model.sort_values("f1")["model"].tolist()
plt.barh(df_metrics_model["model"], df_metrics_model["f1"])
plt.xlabel("F1 (span-level, Levenshtein)")
plt.title("Comparaison des modèles (tous prompts confondus)")
plt.tight_layout()
plt.show()

# %%
# =========================
# 11. Visualisation : F1 par (modèle, prompt)
# =========================

plt.figure(figsize=(10, 5))
x_labels = [f"{m}\n{p}" for m, p in zip(df_metrics_mp["model"], df_metrics_mp["prompt_name"])]
plt.bar(range(len(df_metrics_mp)), df_metrics_mp["f1"])
plt.xticks(range(len(df_metrics_mp)), x_labels, rotation=45, ha="right")
plt.ylabel("F1 (span-level, Levenshtein)")
plt.title("Comparaison modèles / prompts")
plt.tight_layout()
plt.show()

# %%
# =========================
# 12. Inspection d'exemples (optionnel)
# =========================

# Exemple : filtrer quelques phrases où un modèle donné a des FP ou FN (à inspecter à la main)

MODEL_TO_INSPECT = df_metrics_model.iloc[0]["model"]  # meilleur modèle par défaut
PROMPT_TO_INSPECT = df_metrics_mp.sort_values("f1", ascending=False).iloc[0]["prompt_name"]

print("Inspection pour :", MODEL_TO_INSPECT, "/", PROMPT_TO_INSPECT)

df_mp = df_pred[(df_pred["model"] == MODEL_TO_INSPECT) & (df_pred["prompt_name"] == PROMPT_TO_INSPECT)]

# Phrases avec gold spans mais aucune prédiction (FN typiques)
sent_with_gold = [k for k, v in gold_by_sent.items() if len(v) > 0]

# Recalcule dictionnaire locaux (comme plus haut)
df_mp_spans = df_mp[df_mp["span_index"] >= 0].copy()
df_mp_spans["pred_span_norm"] = df_mp_spans[PRED_SPAN_COL].apply(normalize_span)

pred_by_sent_local = (
    df_mp_spans
    .groupby(KEY_COLS_SENTENCE)["pred_span_norm"]
    .apply(list)
    .to_dict()
)

fn_sentences = []
for sent_key in sent_with_gold:
    gold_spans = gold_by_sent.get(sent_key, [])
    pred_spans = pred_by_sent_local.get(sent_key, [])
    _, _, fn = count_tp_fp_fn_for_sentence(gold_spans, pred_spans, LEV_SIM_THRESHOLD)
    if fn > 0:
        fn_sentences.append((sent_key, gold_spans, pred_spans))

print(f"Nb de phrases avec au moins un FN pour ce modèle/prompt : {len(fn_sentences)}")

# Affichage de quelques exemples
for (report_id, sentence_id), gold_spans, pred_spans in fn_sentences[:5]:
    row_example = df_mp[(df_mp["report_id"] == report_id) & (df_mp["sentence_id"] == sentence_id)].iloc[0]
    sent_text = row_example.get(sentence_col, "[[no sentence col]]")
    print("=" * 80)
    print(f"Report: {report_id} | sentence_id: {sentence_id}")
    print("Sentence:", sent_text)
    print("Gold spans:", gold_spans)
    print("Pred spans:", pred_spans)
