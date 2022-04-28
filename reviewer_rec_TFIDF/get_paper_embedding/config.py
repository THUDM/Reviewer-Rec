import os

basedir = os.path.dirname(__file__)

TFIDF_MODEL = os.path.join(basedir, 'tfidf_model.pkl')
SVD_MODEL = os.path.join(basedir, 'svd_model.pkl')

# fn_paper_text = os.path.join(basedir, 'corpus', 'pub_info.json')
# fn_train = os.path.join(basedir, 'corpus', 'trainset.json')
# fn_valid = os.path.join(basedir, 'corpus', 'validset.json')
# fn_test = os.path.join(basedir, 'corpus', 'testset.json')

