# Geneclass25

Geneclass25 is a genomic sequence classification project for identifying pathogen groups from FASTA sequences. It trains fast machine learning models for bacteria, fungi, and RNA viruses using k-mer TF-IDF features with Logistic Regression.

The project is designed for research and educational use. It is not a medical diagnosis tool and should not replace clinical confirmation.

## What It Does

- Imports pathogen sequences from NCBI into local FASTA folders.
- Trains genus-level classifiers for bacteria, fungi, and RNA viruses.
- Trains a kingdom router to choose between Bacteria, Fungi, and Viruses.
- Predicts new FASTA sequences using either manual model selection or automatic routing.
- Tracks training runs with metrics, confusion matrices, predictions, and saved models.
- Provides Streamlit apps for prediction and training history review.

## Project Structure

```text
Geneclass25/
  DNA/
    Bacteria/
    Fungi/
  RNA/
    Viruses/
  runs/
    saved_models/
    training_logs/
  training/
    config.py
    core.py
    artifacts.py
  import file.py
  Main_Train.py
  Train_Kingdom_Router.py
  Central_Code_Reader.py
  Judge_App.py
  Training_Dashboard.py
  central_reader.py
  dataset_loader.py
```

## Requirements

Install the main Python libraries:

```bash
python3 -m pip install pandas numpy scikit-learn biopython joblib streamlit
```

Optional but recommended for NCBI SSL certificate handling:

```bash
python3 -m pip install certifi
```

## Data Source

The dataset is pulled from NCBI Nucleotide using Biopython Entrez. The import script searches for pathogen sequences and stores them as FASTA files under:

```text
DNA/Bacteria/<Genus>/<Species>/
DNA/Fungi/<Genus>/<Species>/
RNA/Viruses/<Genus>/<Species>/
```

Run the importer:

```bash
python3 "import file.py"
```

Useful import options:

```bash
DATASET_TARGETS=rna python3 "import file.py"
AUDIT_ONLY=1 python3 "import file.py"
RNA_SAMPLES_PER_SPECIES=20 python3 "import file.py"
RNA_REQUIRE_COMPLETE_GENOME=1 python3 "import file.py"
```

## Training Models

Train a target model:

```bash
TRAIN_TARGET=bacteria TRAIN_PRESET=fast python3 Main_Train.py
TRAIN_TARGET=fungi TRAIN_PRESET=fast python3 Main_Train.py
TRAIN_TARGET=rna TRAIN_PRESET=accuracy python3 Main_Train.py
```

Train the kingdom router:

```bash
python3 Train_Kingdom_Router.py
```

Models are saved in:

```text
runs/saved_models/
```

Each training run creates artifacts in:

```text
runs/training_logs/run_<id>_<target>_<label_level>/
```

Important artifacts include:

- `summary.json`
- `class_counts.csv`
- `confusion_matrix_val.csv`
- `classification_report_val.csv`
- `predictions_val.csv`
- `accuracy_timeline_by_genus.csv`
- `model_pipeline.joblib`

## Main Training Options

The project is controlled mainly through environment variables.

```bash
TRAIN_TARGET=bacteria|fungi|rna
TRAIN_PRESET=turbo|fast|balanced|accuracy
LABEL_LEVEL=genus|species
SPLIT_MODE=group_species
ENABLE_CV=0|1
DEDUP_EXACT=0|1
KMER_MIN=5
KMER_MAX=5
KMER_MAX_FEATURES=50000
LR_MAX_ITER=5000
LR_CLASS_WEIGHT=balanced
```

RNA-specific options:

```bash
RNA_MIN_UNIQUE_GENOMES_PER_LABEL=5
RNA_MIN_SAMPLES_PER_LABEL=0
RNA_COLLAPSE_NESTED_SPECIES=1
RNA_FAMILY_TOP_K=2
RNA_HIERARCHICAL_WEIGHT=0.0
RNA_USE_FAMILY_FRAGMENT_LEN=0
```

## Prediction

Use the command-line central reader:

```bash
python3 Central_Code_Reader.py --input path/to/input.fasta --source best
```

Force a specific target model:

```bash
python3 Central_Code_Reader.py --input path/to/input.fasta --force-target rna
```

Launch the prediction app:

```bash
streamlit run Judge_App.py
```

Launch the training dashboard:

```bash
streamlit run Training_Dashboard.py
```

## How The Model Works

The model treats a genome sequence like biological text. It breaks each sequence into overlapping k-mers, such as 5-letter nucleotide patterns, then converts those patterns into TF-IDF features.

The classifier is Logistic Regression. This keeps training fast, reduces lag, and works well with sparse k-mer features.

For example, with `k=5`:

```text
ACGTTAGC -> ACGTT, CGTTA, GTTAG, TTAGC
```

The model learns which k-mer patterns are more common in each genus.

## Why Bacteria, Fungi, And RNA Viruses Behave Differently

Bacteria usually performs best because 16S rRNA is a strong marker gene. Fungi is harder because ITS, 18S, and 28S records can be more mixed. RNA viruses are the hardest because they mutate quickly, have uneven genome lengths, may be segmented, and often have noisy or nested labels.

For RNA viruses, the project adds extra data-cleaning methods:

- Require complete genomes when importing, when possible.
- Remove bad or partial records.
- Normalize labels such as Influenza subtype names.
- Drop classes with too few genomes or samples.
- Deduplicate exact sequences.
- Use group-aware validation splitting.

## Validation And Metrics

The project reports:

- Accuracy
- Balanced Accuracy
- F1 Macro
- F1 Weighted
- MCC
- LogLoss when probability output is valid

Validation uses `group_species` splitting by default. That means records are grouped by genus/species during train-validation splitting to reduce leakage from highly similar species records.

Cross-validation can be enabled with:

```bash
ENABLE_CV=1 python3 Main_Train.py
```

It is disabled by default in most fast workflows because it increases training time.

## Important Notes

- High validation accuracy does not always mean the model is clinically reliable.
- RNA virus accuracy is lower because the biology and labels are harder.
- Dropping low-sample RNA classes can increase accuracy, but it also makes the task easier.
- A true external test set is still recommended before any serious deployment.
- The Streamlit clinical mapping is for research display only, not diagnosis.

## Resetting Training State

Archive current models and logs, then start fresh:

```bash
python3 reset_training_state.py
```

This moves old artifacts into:

```text
runs/archive/
```

## Recommended Workflow

```bash
python3 "import file.py"
python3 Train_Kingdom_Router.py
TRAIN_TARGET=bacteria TRAIN_PRESET=fast python3 Main_Train.py
TRAIN_TARGET=fungi TRAIN_PRESET=fast python3 Main_Train.py
TRAIN_TARGET=rna TRAIN_PRESET=accuracy python3 Main_Train.py
streamlit run Judge_App.py
streamlit run Training_Dashboard.py
```

