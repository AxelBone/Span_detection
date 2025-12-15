"""
run_spans.py

Script pour :
- charger les données (phrases)
- appeler un modèle local (ou distant, ex : epfl-llm/meditron-7b, Qwen-3-32B, etc.)
- produire un tableau avec les spans prédits

Toute la configuration (modèle, hyperparamètres, chemins) est externalisée
dans un fichier JSON passé en argument --config.
"""

import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import pandas as pd
import re
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
    model_name: str = "models/meditron-7b"
    device_map: str = "auto"
    dtype: str = "float16"  # "bfloat16" / "float16" / "float32"
    local_files_only: bool = False
    trust_remote_code: bool = False  # utile pour certains modèles comme Qwen


@dataclass
class IOConfig:
    data_path: str = "data/spans_dataset.tsv"
    sep: str = "\t"
    encoding: str = "utf-8"
    sentence_col: str = "Sentence_en"
    out_dir: str = "results"
    pred_col_prefix: str = "span_pred"  # le suffix sera basé sur le nom du modèle


@dataclass
class GlobalConfig:
    model: ModelConfig
    generation: GenerationConfig
    io: IOConfig


# =========================
# 2. Utilitaires de config
# =========================

def _dict_to_dataclass(dc_cls, data: Dict[str, Any]):
    # permet de gérer les valeurs manquantes ou en trop
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
            # remplace {models_root}, {data_root}, etc.
            return value.format(**paths)
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        return value

    model_raw = expand(raw.get("model", {}))

    model_name = model_raw.get("model_name")
    models_root = paths.get("models_root")

    if model_name is not None and models_root:
        if not os.path.isabs(model_name):
            model_raw["model_name"] = os.path.join(models_root, model_name)
    
    gen_raw = expand(raw.get("generation", {}))
    io_raw = expand(raw.get("io", {}))

    filename = io_raw.get("filename")
    data_root = paths.get("data_root")
    if filename is not None:
        if data_root and not os.path.isabs(filename):
            io_raw["data_path"] = os.path.join(data_root, filename)
        else:
            # filename déjà absolu ou pas de data_root
            io_raw["data_path"] = filename

    # 2) Construire out_dir relatif au project_root si nécessaire
    out_dir = io_raw.get("out_dir", "results")
    project_root = paths.get("project_root")
    if out_dir is not None and project_root and not os.path.isabs(out_dir):
        io_raw["out_dir"] = os.path.join(project_root, out_dir)

    model_cfg = _dict_to_dataclass(ModelConfig, model_raw)
    gen_cfg = _dict_to_dataclass(GenerationConfig, gen_raw)
    io_cfg = _dict_to_dataclass(IOConfig, io_raw)

    return GlobalConfig(model=model_cfg, generation=gen_cfg, io=io_cfg)



# =========================
# 3. Chargement des données
# =========================

def load_dataset(path: str,
                 sep: str = "\t",
                 encoding: str = "utf-8") -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path, sep=sep, encoding=encoding)
    return df


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

SENTENCE: Lysosomal enzymes in cultured skin fibroblasts such as beta-galactosidase or total beta-hexosaminidase were within normal limits.
==========
Span: None

SENTENCE: Skeletal survey showed 11 pairs of ribs.
==========
Span: 11 pairs of ribs

SENTENCE: It showed atrophied thalami and restricted water diffusion.
==========
Span: atrophied thalami, abnormal water regulation
"""

    start = f"SENTENCE: {sentence}\n==========\nSpan: "

    full_prompt = f"{persona}\n\n{instruction}\n\n{examples}\n{start}"
    return full_prompt


# =========================
# 5. Initialisation du modèle générique
# =========================

def load_model(model_cfg: ModelConfig):
    """
    Charge le tokenizer et le modèle (type causal LM) en local ou distant.
    Compatible avec Meditron, Qwen, etc.
    """

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(model_cfg.dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name,
        local_files_only=model_cfg.local_files_only,
        use_fast=True,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    # Certains modèles (LLaMA-like, Qwen, etc.) n'ont pas de pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name,
        device_map=model_cfg.device_map,
        torch_dtype=torch_dtype,
        local_files_only=model_cfg.local_files_only,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    model.eval()
    return tokenizer, model


# =========================
# 6. Fonction d'appel au modèle
# =========================

def generate_span_for_sentence(
    sentence: str,
    tokenizer,
    model,
    gen_cfg: GenerationConfig,
) -> str:

    prompt = build_prompt(sentence)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        top_k=gen_cfg.top_k,
        do_sample=gen_cfg.do_sample,
        repetition_penalty=gen_cfg.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extraction naïve de la partie après "Span:"
    if "Span:" in full_text:
        span_text = full_text.split("Span:")[-1].strip()
    else:
        # fallback : tout le texte généré après le prompt
        generated_only = full_text[len(prompt):].strip()
        span_text = generated_only

    return span_text


# =========================
# 7. Negative detection
# =========================

def detect_negation(text):
    if not isinstance(text, str):
        return text, False  # Return non-string values as is

    negation_patterns = ['never had', 'unremarkable', 'normal', 'no']  # Add more patterns as needed
    # exceptions = ['without particularity', 'without significance', 'was normal', 'not found']  # Add patterns that should not be removed at the beginning

    negation_found = False

    # Iterate over negation patterns and apply removal based on context
    for pattern in negation_patterns:
        # Create a regular expression pattern with word boundaries around negation words
        pattern_regex = r'\b' + re.escape(pattern) + r'\b'

        # Use re.sub to remove the negation part from the text
        text, count = re.subn(pattern_regex, '', text, flags=re.IGNORECASE)

        # Check if negation pattern was found
        if count > 0:
            negation_found = True

    return negation_found  # Remove leading/trailing whitespaces and return negation flag



# =========================
# 8. Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Chemin du fichier de configuration JSON",
    )
    args = parser.parse_args()

    # --- Chargement config ---
    cfg = load_config_from_json(args.config)
    model_cfg = cfg.model
    gen_cfg = cfg.generation
    io_cfg = cfg.io

    print("Configuration modèle :", asdict(model_cfg))
    print("Configuration génération :", asdict(gen_cfg))
    print("Configuration IO :", asdict(io_cfg))

    # --- Chargement du modèle ---
    tokenizer, model = load_model(model_cfg)

    # --- Chargement du dataset ---
    df = load_dataset(io_cfg.data_path, sep=io_cfg.sep, encoding=io_cfg.encoding)

    if io_cfg.sentence_col not in df.columns:
        raise ValueError(
            f"Colonne '{io_cfg.sentence_col}' introuvable dans le fichier. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    # --- Application du modèle sur chaque phrase ---
    preds = []
    for sent in df[io_cfg.sentence_col]:
        span_pred = generate_span_for_sentence(
            sentence=sent,
            tokenizer=tokenizer,
            model=model,
            gen_cfg=gen_cfg,
        )
        preds.append(span_pred)


    # suffix basé sur le nom du modèle (partie après le slash)
    model_suffix = model_cfg.model_name.split("/")[-1]
    model_suffix = model_suffix.replace("-", "_").replace(" ", "_").lower()

    pred_col_name = f"{io_cfg.pred_col_prefix}_{model_suffix}"
    df[pred_col_name] = preds


    # Negation
    neg_detected = [detect_negation(sent) for sent in df[io_cfg.sentence_col]]
    df["negation"] = neg_detected

    # chemin de sortie
    os.makedirs(io_cfg.out_dir, exist_ok=True)
    out_path = os.path.join(io_cfg.out_dir, f"spans_{model_suffix}.tsv")
    df.to_csv(out_path, sep=io_cfg.sep, index=False)

    print(f"Résultats sauvegardés dans : {out_path}")
    print(f"Hyperparamètres de génération : {asdict(gen_cfg)}")


if __name__ == "__main__":
    main()