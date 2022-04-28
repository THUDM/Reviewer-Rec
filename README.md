# Reviewer-Rec

## Task

Here are two simple examples of reviewer matching by using TD-IDF, LightGCN and GF-CF.
Please refer to teh follwoing steps to run our codes

## Requirements

- Python version >= 3.6
- PyTorch version >= 1.6.0
- Network for initial run

We use the following packages, you need to install it.

- tqdm
- scikit-learn
- joblib
- numpy
- torch

## Usage

### TF-IDF

We use the embedding method of TF-IDF, the pretrained vectorizer and model is provided via [this url](https://cloud.tsinghua.edu.cn/f/534a960c0f7d479fb832/?dl=1), and put this folder as `/reviewer_rec_TFIDF/get_paper_embedding/embedding_data`.
Also, download the data from [Here](https://cloud.tsinghua.edu.cn/f/8708cd1fc7a54dd6b446/?dl=1), and put this folder as `/reviewer_rec_TFIDF/get_paper_embedding/data`.


Please run:
```bash
cd reviewer_rec_TFIDF
python parse_paper_information.py
``` 

to get the results.

### GF-CF and LightGCN

This code is heavliy bulit on the official implementation of [GF-CF](https://github.com/yshenaw/GF_CF) and [LightGCN](https://github.com/gusye1234/LightGCN-PyTorch).

download the data from [Here](https://cloud.tsinghua.edu.cn/f/5851cdac725743b4bef4/?dl=1), and put this folder as `/reviewer_rec_LightGCN/data/reviewer_rec`.

Please modify the `ROOT_PATH` in `world.py`.

To run LightGCN, use the following command:

```python
cd reviewer_rec_LightGCN
python main.py --dataset="reviewer_rec" --topks="[20,5,10,50]" --model "lgn" --gpu_id 0
```

To run GF-CF, use the following command:

```python
cd reviewer_rec_LightGCN
python main.py --dataset="reviewer_rec" --topks="[20,5,10,50]" --simple_model "gf-cf" --gpu_id 0
```
