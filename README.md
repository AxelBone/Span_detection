# Extraction de spans phénotypiques avec des LLMs

Ce dépôt contient des scripts pour **l’extraction automatique de spans phénotypiques** (et optionnellement la négation) à partir de phrases cliniques, en utilisant des **LLMs** :

- soit via un **appel local Python** (`transformers`, inférence dans le process),
- soit via un **serveur Ollama** (appel HTTP),
- avec une **configuration entièrement externalisée** en JSON,
- et des sorties exploitables pour l’évaluation (format long TSV).

---

## 📁 Arborescence du projet

```
.
├── README.md
├── configs
│ ├── configs-local.json # Config inférence locale (Transformers)
│ ├── configs-ollama.json # Config inférence via Ollama
│ └── obs # Anciennes configs / brouillons
├── data
│ ├── gold_sample_N500.tsv
│ └── gold_spans.tsv # Generated from CHU50
├── notebook
│ ├── evaluate_neg.ipynb
│ ├── evaluate_span_detection.ipynb
│ └── evaluate_span_detection.py
├── prompts
│ ├── prompt.txt
│ ├── small_models
│ ├── span_prompt_with_examples.txt
│ ├── span_prompt_with_examples_strict_version.txt
│ └── span_prompt_without_examples_strict_version.txt
└── scripts
├── run_spans.py # Script principal (local ou Ollama selon config)
├── neg.py
├── sample_gold.py
├── prepare_gold_standard_for_span_detection_and_negation_evaluation.py
├── evaluate_time_for_run_spans.py
├── evaluate_time_for_run_spans_ollama.py
└── evaluate_time_for_run_spans_vLLM.py
```

---

## 🎯 Fonctionnalités principales

- Lecture d’un dataset **1 phrase par ligne**
- Application de **plusieurs prompts** sur chaque phrase
- Appel :
  - soit d’un **modèle local** (`transformers`)
  - soit d’un **modèle distant via Ollama**
- Sortie au **format long** :
  - 1 ligne = 1 span prédit
  - conservation de la sortie brute du modèle
- Gestion :
  - des logs
  - de la reprise sur checkpoint
  - des temps d’inférence

---

## ⚙️ Pré-requis

### Python
- Python **3.9+** recommandé

### Dépendances minimales
```bash
pip install torch transformers accelerate pandas
```

⚠️ Pour les modèles volumineux (Qwen 32B, etc.), un GPU avec suffisamment de VRAM est fortement recommandé.
Le paramètre device_map="auto" est supporté.


### 📄 Format des données en entrée

Le script lit un fichier TSV / CSV contenant une colonne de phrases.

Exemple minimal :
Sentence_en
Shortly after birth, he developed tachypnea...
MR spectroscopy showed a region of increased...

Le nom de la colonne est configurable via :

"io": {
  "sentence_col": "Sentence_en"
}

### ▶️ Utilisation
Lancer une extraction

`python scripts/run_spans.py --config configs/configs-local.json`

Ou avec Ollama :

`python scripts/run_spans.py --config configs/configs-ollama.json`

### 🧩 Configuration JSON

Toute la logique est pilotée par un fichier JSON.
Exemple : inférence locale (configs/configs-local.json)

```
{
  "paths": {
    "project_root": "/home/prollier/ext/Span_detection/",
    "models_root": "/home/prollier/models/",
    "data_root": "/home/prollier/output/for_span_detection_formatted/"
  },

  "model": {
    "model_name": "{models_root}/qwen3-32b",
    "device_map": "auto",
    "dtype": "float16",
    "local_files_only": true,
    "trust_remote_code": true
  },

  "generation": {
    "max_new_tokens": 64,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 50,
    "do_sample": false,
    "repetition_penalty": 1.0
  },

  "io": {
    "filename": "gold_sample_N500.tsv",
    "sep": "\t",
    "encoding": "utf-8",
    "sentence_col": "Sentence_en",
    "out_dir": "results"
  },

  "runtime": {
    "batch_size": 100,
    "log_file": "run_local.log",
    "log_level": "INFO",
    "resume": true
  },

  "prompt": {
    "template_paths": [
      "prompts/small_models/span_detection.txt",
      "prompts/small_models/span_detection_with_examples.txt"
    ],
    "sentence_var": "sentence"
  }
}
```

Exemple : inférence Ollama (configs/configs-ollama.json)

{
  "ollama": {
    "base_url": "https://compute-01.odh.local/ollama",
    "model": "deepseek-r1:8b-llama-distill-q4_K_M",
    "timeout_s": 120.0,
    "verify_ssl": false
  }
}

➡️ Cette config est combinée avec les sections communes (paths, io, prompt, etc.).

### 📤 Format des sorties

Les résultats sont écrits dans :

results/spans_long_<model>.tsv

Colonnes importantes

    model
    prompt_name
    prompt_index
    span_index
    span_text
    spans_count
    raw_output
    latency_s

toutes les colonnes originales du dataset

👉 Format long : une ligne par span (ou une ligne vide si aucun span).
🧪 Évaluation

Les notebooks et scripts d’évaluation sont disponibles dans :

notebook/

    evaluate_span_detection.ipynb
    evaluate_neg.ipynb

Ils permettent de comparer les prédictions aux gold standards présents dans data/.
🧠 Modèles compatibles

    LLaMA / derivatives
    Meditron
    Qwen (souvent trust_remote_code=true)
    Tout modèle compatible AutoModelForCausalLM

🚀 Extensions possibles

    Quantisation 4-bit / 8-bit (bitsandbytes)
    vLLM
    batching multi-phrases
    fallback automatique Ollama → local
    parsing structuré JSON strict

📌 Notes

    Aucun code n’est spécifique à une langue : EN / FR supportés
    Les prompts sont entièrement externalisés
    Le script est conçu pour des runs longs et reproductibles

👤 Auteur / Contact
Projet interne — adapté pour l’expérimentation LLM en extraction clinique.
