# News Headline Selection

**Transformers for Headline Selection for Russian News Clusters**
Pavel Voropaev, Olga Sopilnyak

**Abstract:** In this paper, we explore various multilingual and Russian pre-trained transformer-based models for the Dialogue Evaluation 2021 shared task on headline selection. Our experiments show that the combined approach issuperior to individual multilingual and monolingual models. We present an analysis of a number of ways to obtainsentence embeddings and learn a ranking model on top of them. We achieve the result of 87.28% and 86.60% accuracy for the public and private test sets respectively.

## Download training and test sets

    wget https://www.dropbox.com/s/jpcwryaeszqtrf9/titles_markup_0525_urls.tsv
    wget https://www.dropbox.com/s/v3c0q6msy2pwvfm/titles_markup_test.tsv

## Download pre-trained embedding weights

[Google Drive](https://drive.google.com/drive/u/1/folders/1frkNOP1IkHWC7Mq6RnnyFh6YfVdohGo3)

## Train a classifier

    python train.py [path/to/pre-trained]
