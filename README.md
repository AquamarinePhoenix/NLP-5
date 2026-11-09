# Task 5: Emotion Classification Using Machine Learning 
 
This  laboratory  exercise  aims  to  investigate  emotion  classification  in  text  using  machine  learning 
techniques.  

1. Text dataset preparation 
Use  the  metadata.csv  file  from  the  Lithuanian  Emotional  Speech  Corpus  (EMO)  to  create  a  text 
classification dataset. 
− Extract the text samples and their corresponding emotion labels. 
− Identify and exclude all texts associated with multiple emotion labels. 
− Remove duplicate texts, ensuring that each unique text appears only once in the dataset. 
− Save the resulting dataset in CSV format, containing the text samples and their emotion labels, for 
further analysis and classification. 
 
2. Text Analysis 
Analyze the linguistic characteristics of the prepared text dataset and investigate differences across the 
five emotion classes. 
− Text length: Visualize and compare the distributions of text lengths (in words) across the emotion 
classes. 
− Vocabulary analysis: Apply lemmatization and stop-word removal. Identify and compare the most 
frequent words in each emotion class. Generate a word cloud for each emotion class to visualize 
its most frequent words. 
− Part-of-speech  (POS)  analysis:  Analyze  and  compare  the  distributions  of  POS  tags  across  the 
emotion classes. 
 
3. Text Representation (Feature Extraction) 
− Split the dataset into training and test sets using stratified sampling. 
− Select and implement three text representation methods, choosing one from each category: 
    1.  Frequency-based representations: Bag-of-Words (BoW), TF-IDF, or other method. 
    2.  Static word embeddings: Word2Vec, GloVe, FastText, or another model. 
    3.  Contextual embeddings: BERT, Sentence-BERT, or other pretrained transformer model.
− Convert each text into a fixed-length feature vector suitable for classification. Ensure that any data-
dependent feature extraction is fitted using only the training set. 
 
4. Emotion Classification 
Train a Logistic Regression classifier to predict the five emotion classes using each of the three selected 
text representations. 
− Use the same training and test sets for all three representations. 
− Train a separate classifier for each representation. 
− Evaluate each classifier using accuracy, precision, recall, and F1-score. 
− Generate and compare confusion matrices for all three models. 
 
 
In the report:  
✓ Report the number of texts associated with multiple emotion labels and excluded from the 
dataset. Report the number of text samples before and after duplicate removal. Present the 
distribution of texts across the five emotion classes.  
✓ Summarize the main findings from the text analysis part, highlighting notable similarities and 
differences between the emotion classes. 
✓ Describe and justify the selected text representation methods. Compare the resulting 
representations in terms of dimensionality, sparsity, and their ability to capture semantic 
information. 
✓ Present the classification results in a comparative table. Discuss which text representation 
performs best and analyze the most common classification errors. 
 
Report requirements: 
 
The Python code should be included in the appendix. 
Scope for report: 4-6 pages (excluding the appendix and title page) 
 
The report may be prepared using MS Word, LaTeX, or another document preparation environment of 
your choice. The final report must be submitted in PDF format. The report should follow the general 
formatting requirements applied to academic text. In particular, all figures and tables must be numbered 
and have informative captions; all figures and tables must be referred to and discussed in the text; etc.

# Code summary

The project is executed from `main.py`, which coordinates the complete
data-processing, analysis, feature-extraction, and classification workflow.
The main stages are:

- **Data loading and configuration (`nlplib/data.py`, `nlplib/config.py`)**:
  the input file is read line by line from `data/input/metadata.csv`. Each
  record is parsed into its raw value, emotion label, and normalized text.
  The configurable test mode uses a deterministic random seed and can limit
  execution to a selected number of rows.
- **Database preparation (`nlplib/db.py`, `nlplib/sql.py`)**: SQLite tables
  store raw metadata, grouped text, emotion definitions, text-length
  distributions, and serialized feature vectors. The grouped table enforces
  unique text samples and records dataset statistics such as duplicates,
  multi-label texts, and class distribution. The processed dataset is
  exported to `data/output/output.csv`.
- **Text preprocessing (`nlplib/transform.py`)**: spaCy's Lithuanian model
  performs tokenization, lemmatization, alphabetic-token filtering, and
  stop-word removal. Processing is batch-based and parallelized with worker
  processes, and the transformed text is written back to SQLite.
- **Exploratory analysis (`nlplib/statistics.py`, `nlplib/plot.py`)**:
  the workflow calculates text-length statistics from the SQLite
  `LENGTH(TEXT_SAMPLE)` values, including skewness, kurtosis, and an
  Anderson-Darling comparison between emotion groups. It also creates
  emotion-specific word clouds, POS-tag distributions, and UMAP projections
  in `data/output/`.
- **Feature extraction (`nlplib/features.py`)**: the data is split into
  stratified training and test sets using a fixed random state. Three
  representations are generated using training data: TF-IDF vectors,
  locally computed static embeddings based on a token co-occurrence matrix
  reduced with truncated SVD, and contextual sentence embeddings from
  `paraphrase-multilingual-MiniLM-L12-v2`. The three extractors run in
  parallel, and their dimensions and sparsity are logged.
- **Classification (`nlplib/classification.py`)**: each representation is
  evaluated with multinomial logistic regression and a 200-tree random
  forest. Accuracy, macro-precision, macro-recall, and macro-F1 are logged.
  Confusion matrices, common classification errors, logistic probability
  plots, and representative random-forest trees are saved to
  `data/output/`.
- **Logging and output management (`nlplib/log.py`)**: previous logs and PNG
  files are cleared at the start of a run. Progress messages, dataset
  statistics, model metrics, and validation information are written to
  `data/output/log.txt` and printed to the console.

The dependencies used by the pipeline are listed in `requirements.txt`.
Install them before running the pipeline:

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
    check=True,
)
```

The normal entry point is:

```python
from main import main

if __name__ == "__main__":
    main()
```

Alternatively, run the complete workflow from a terminal:

```text
python main.py
```

## Configuration and usage notes

> [!NOTE]
> The input dataset must be available at `data/input/metadata.csv`. The
> parser expects each non-empty line to contain an emotion label between
> slashes and the text after the final `|` character.

> [!WARNING]
> `main.py` installs packages from `requirements.txt` every time it starts.
> A virtual environment is recommended so the project dependencies do not
> modify the system Python installation.

> [!NOTE]
> `nlplib/config.py` has `TEST = True` and `TEST_ROWS = 5000` by default.
> This limits the run to a reproducible sample of two emotion classes.
> Set `TEST = False` to process the complete dataset.

```python
# nlplib/config.py
TEST = False
TEST_ROWS = 5000  # Used only when TEST is True.
```

> [!WARNING]
> Text preprocessing uses spaCy's `lt_core_news_sm` Lithuanian model.
> If it is not installed, the program attempts to download it automatically,
> so the first run requires internet access and additional disk space.

The pipeline creates its output directory automatically. Important generated
files include:

```text
data/output/
├── metadata.db
├── output.csv
├── log.txt
├── confusion_matrices.png
├── text_length_distribution.png
├── pos_distribution.png
├── wordclouds/
├── umap/
├── logistic_cdf/
└── random_forest/
```

> [!NOTE]
> Existing log files and PNG visualizations in `data/output/` are cleared at
> the beginning of each run. Copy any results that must be preserved before
> starting another run.
