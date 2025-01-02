# NLP Demo: Word Embeddings and Text Analysis

## Overview
This repository contains three Jupyter notebooks demonstrating various applications of Natural Language Processing (NLP) using word embeddings, specifically with Word2Vec models. The demos showcase interactive text analysis, detailed word embedding exploration, and text visualisation techniques.

## Notebooks
1. **Gensim_interactive_demo.ipynb**  
   An interactive demonstration allowing users to experiment with word embeddings in real-time. Enter words of your choice and observe how the model processes and relates them to other terms in its vocabulary.

2. **word_embedding_explorer.ipynb**  
   A comprehensive tool for conducting detailed analysis of word embeddings. This notebook enables precise testing and evaluation of semantic relationships between words, offering insights into the underlying structure of the language model.

3. **word_embeddings_plotter.ipynb**  
   Visualise text representations in both 2D and 3D spaces using Principal Component Analysis (PCA). This notebook uses a selection of Shakespeare's sonnets as sample text, demonstrating how complex linguistic relationships can be mapped into visual representations.

## Prerequisites

### Required Libraries
- Gensim (version 4.3.3)
- spaCy
- Additional Python dependencies (specified in requirements.txt)

### Model Installation

#### Word2Vec Model
```python
import gensim
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

# List available models
print(list(gensim.downloader.info()['models'].keys()))

# Download and save the pre-trained model
wv = api.load('word2vec-google-news-300')
wv.save(your_path+'word2vec-google-news-300.kv')
```

#### spaCy Models
Install either or both of the following spaCy models using terminal commands. These can also be downloaded by running requirements.txt. 
```bash
# For the larger, more comprehensive model
python -m spacy download en_core_web_lg

# For the smaller, faster model
python -m spacy download en_core_web_sm
```

### Text Corpus
The visualisation notebook uses Shakespeare's sonnets (specifically sonnets 25-50) as sample text. You can download the complete works from:
[Shakespeare's Complete Works](https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt)

## Usage Instructions

### Loading Word2Vec Model
```python
# Load the pre-trained model
model = api.load(your_path+"word2vec-google-news-300")

# Access word vectors
word_vectors = model.wv
```

### Loading spaCy Model
```python
import spacy
nlp = spacy.load("en_core_web_sm")
```

## Additional Resources

For more detailed information about the models used in this project:

- [Word2Vec Google News Model Documentation](https://huggingface.co/fse/word2vec-google-news-300)
- [spaCy English Model Documentation](https://huggingface.co/spacy/en_core_web_sm)

## Notes
- The Word2Vec model used is the Google News pre-trained model, which contains 300-dimensional word vectors trained on a substantial corpus of news articles.
- The spaCy models offer different trade-offs between accuracy and performance - choose based on your specific needs.
- The visualisation notebook specifically uses sonnets 25-50 to provide a focused and manageable dataset for demonstration purposes.


T. Mehta, 02/01/25