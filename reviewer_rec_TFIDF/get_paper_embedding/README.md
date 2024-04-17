# Get Paper Embedding

A simple toolbox for getting a representation vector of a paper.

## Requirements and installation 
Requirements:
* Python version >= 3.6
* PyTorch version >= 1.6.0
* Network for initial run

We also use the following packages, which you may install via `pip install -r requirements.txt`.
* tqdm
* scikit-learn
* joblib
* numpy
* torch
* cogdl

## Usage

First, please modify the `config.py`, and fill the file path of corpus.

Currently, we support two types of embedding methods (i.e. `OAG-BERT` and `TF-IDF`) and a mixture of them.
To use this toolbox, just clone this repo and copy the `paper_emb` folder to your project, and call the embedding function like the format we wrote in `example.py`:

```python
from paper_emb.oag_tfidf_emb import get_oag_emb, get_tfidf_emb, get_oag_tfidf_emb

# get the embedding
oag_emb = get_oag_emb(paper_example)
idf_emb = get_tfidf_emb(paper_example)
mix_emb = get_oag_tfidf_emb(paper_example)

# shape
print("OAG:{}|IDF:{}|MIX:{}".format(oag_emb.shape, idf_emb.shape, mix_emb.shape))
# OAG:(768,)|IDF:(100,)|MIX:(868,)
```

The `paper_example` is a dict:

```python
paper_example = {
    "title": "A paper title.", # Mandatory
    "abstract": "A paper abstract.", # Optional, used for both OAG and TF-IDF
    "keywords": ["word1", "word2", "word3"], # Optional, only required in OAG mode
    "venue": "a possible venue", # Optional, only required in OAG mode
    "authors": ["author 1", "author 2"], # Optional, only required in OAG mode
    "affiliations": ["org 1", "org 2", "org 3"] # Optional, only required in OAG mode
}
```

## Notes

For the first time to use this toolbox, some pretained models will be downloaded. 
If the downloading is failed, you may ask the author to get the pre-downloaded model.


