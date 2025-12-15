# run_spans.py — Extraction de *spans* phénotypiques avec un LLM

Ce dépôt contient `run_spans.py`, un script qui :
- charge un dataset de **phrases** (1 ligne = 1 phrase / document),
- appelle un **modèle LLM** (local ou distant : ex. `epfl-llm/meditron-7b`, `Qwen-3-32B`, etc.),
- produit un fichier de sortie avec les **spans prédits** (phénotypes) + un flag de **négation**,
- **externalise toute la configuration** (modèle, hyperparamètres, chemins) dans un fichier JSON passé via `--config`.

---

## 1) Pré-requis

### Python & dépendances
- Python 3.9+ recommandé
- `torch`
- `transformers`
- `pandas`

Installation typique :
```bash
pip install -U torch transformers pandas
````

> ⚠️ Pour les modèles volumineux, prévois une machine GPU et suffisamment de VRAM, ou utilise `device_map="auto"` avec accélération adaptée.

---

## 2) Format attendu des données

Le script lit un fichier TSV/CSV (par défaut TSV) contenant une colonne de phrases (par défaut `Sentence_en`).

Exemple minimal :

| Sentence_en                                       |
| ------------------------------------------------- |
| "Shortly after birth, he developed tachypnea..."  |
| "MR spectroscopy showed a region of increased..." |

> Le nom de la colonne est configurable via `io.sentence_col`.

---

## 3) Utilisation rapide

### Lancer le script

```bash
python run_spans.py --config configs/run_spans.json
```

Le script :

1. charge la config JSON,
2. charge le tokenizer + modèle (`AutoTokenizer`, `AutoModelForCausalLM`),
3. génère une prédiction pour chaque phrase,
4. écrit un fichier TSV dans `out_dir`.

---

## 4) Configuration JSON

Toute la configuration est dans un fichier JSON.

### Exemple (fourni)

```json
{
  "paths": {
    "project_root": "/home/me/code/phenotype-project",
    "models_root": "/mnt/big_disk/models",
    "data_root": "/mnt/shared_data/chu50_en_v2"
  },

  "model": {
    "model_name": "meditron-7b",
    "device_map": "auto",
    "dtype": "float16",
    "local_files_only": true,
    "trust_remote_code": false
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
    "filename": "spans_dataset.tsv",
    "sep": "\t",
    "encoding": "utf-8",
    "sentence_col": "Sentence_en",
    "out_dir": "results",
    "pred_col_prefix": "span_pred"
  }
}
```

### Mécanisme des chemins

Le script supporte une section `paths` pour construire automatiquement des chemins :

* `models_root` : racine des modèles locaux
* `data_root` : racine des données
* `project_root` : racine projet (utile pour `out_dir`)

Règles importantes :

* Si `model.model_name` n’est pas absolu et que `paths.models_root` existe, alors :

  * `model_name` devient `os.path.join(models_root, model_name)`
* Si `io.filename` est fourni :

  * `io.data_path` devient `os.path.join(data_root, filename)` (si relatif)
* Si `io.out_dir` est relatif et `project_root` existe :

  * `out_dir` devient `os.path.join(project_root, out_dir)`

> Remarque : si tu préfères, tu peux aussi fournir directement `io.data_path` (chemin complet) au lieu de `filename`.

---

## 5) Sorties générées

### Colonnes ajoutées

* Une colonne de prédiction :
  `span_pred_<model_suffix>`

Où :

* `model_suffix` = dernier segment de `model_name`, normalisé (minuscules, `-` → `_`)

Exemple :

* modèle `.../meditron-7b` ⇒ colonne `span_pred_meditron_7b`

### Négation

Le script ajoute aussi :

* `negation` : un booléen indiquant si un motif de négation a été détecté dans la phrase.

> ⚠️ Note : la fonction `detect_negation` retire des motifs via regex mais retourne uniquement un booléen (`negation_found`). Si tu veux conserver le texte nettoyé + le flag, il faut adapter cette fonction et son appel.

### Fichier de sortie

Le fichier est écrit dans :

* `io.out_dir/spans_<model_suffix>.tsv`

Exemple :

* `results/spans_meditron_7b.tsv`

---

## 6) Prompting / comportement attendu du modèle

Le prompt (dans `build_prompt`) :

* définit une persona clinique,
* donne des consignes d’extraction de spans phénotypiques (avec négation/adjectifs),
* fournit plusieurs exemples “SENTENCE → Span: …”,
* exige `None` en absence de span / incertitude.

La génération extrait ensuite naïvement ce qui se trouve après `"Span:"`.

---

## 7) Modèles compatibles

Le script utilise :

* `AutoTokenizer.from_pretrained`
* `AutoModelForCausalLM.from_pretrained`

Il fonctionne généralement avec :

* modèles type LLaMA/Meditron,
* Qwen (souvent besoin de `trust_remote_code=true` selon le modèle).

Paramètres utiles :

* `model.local_files_only=true` si modèles présents localement,
* `model.trust_remote_code=true` pour certains repos HF,
* `model.dtype` : `float16` / `bfloat16` / `float32`.

