# Reviewer-Rec

## Task

Here are several examples of reviewer matching by using TD-IDF, LightGCN and GF-CF.
Please refer to the following steps to run our codes.

## Requirements

- Python version >= 3.6
- PyTorch version >= 1.6.0
- Network for initial run
- ```pip install -r requirements.txt``` (Note: sparsesvd can be installed from source)

## Usage

### TF-IDF

We use the embedding method of TF-IDF, the pretrained vectorizer and model is provided via [this url](https://cloud.tsinghua.edu.cn/f/534a960c0f7d479fb832/?dl=1), and put this folder as `reviewer_rec_TFIDF/get_paper_embedding/embedding_data`.
Also, download the data from [Here](https://cloud.tsinghua.edu.cn/f/8708cd1fc7a54dd6b446/?dl=1), and put this folder as `reviewer_rec_TFIDF/get_paper_embedding/data`.

Moreover, download `tfidf_model.pkl` from [Baidu Pan](https://pan.baidu.com/s/1e7fcTmw41nOgbzM5o-LdCQ?pwd=jey2) with password jey2 and `svd_model.pkl` from [Baidu Pan](https://pan.baidu.com/s/1i4_Ijt6gf6eK7pqPM3ob8g?pwd=qcng) with password qcng. Put these two models in into `reviewer_rec_TFIDF/get_paper_embedding/`.

Download `paper_embedding.json` from https://pan.baidu.com/s/1mvNnpRY6fWOM4mUE3WsZQQ?pwd=7suq and `training_reviewer_embedding.json` from https://pan.baidu.com/s/1ish6ofqTm5dPiz0PpDQ9Hg?pwd=8jy8. Put these two embedding files into `reviewer_rec_TFIDF/get_paper_embedding/embedding_data`.


Please run:
```bash
cd reviewer_rec_TFIDF
python parse_paper_information.py
``` 

to get the results.

### GF-CF and LightGCN

This code is heavliy bulit on the official implementation of [GF-CF](https://github.com/yshenaw/GF_CF) and [LightGCN](https://github.com/gusye1234/LightGCN-PyTorch).

Download the data from [Here](https://cloud.tsinghua.edu.cn/f/5851cdac725743b4bef4/?dl=1), and put this folder as `/reviewer_rec_LightGCN/data/reviewer_rec`.


To run LightGCN, use the following command:

```bash
cd reviewer_rec_LightGCN
python main.py --dataset="reviewer_rec" --topks="[20,5,10,50]" --model "lgn" --gpu_id 0
```

To run GF-CF, use the following command:

```bash
cd reviewer_rec_LightGCN
python main.py --dataset="reviewer_rec" --topks="[20,5,10,50]" --simple_model "gf-cf" --gpu_id 0
```
