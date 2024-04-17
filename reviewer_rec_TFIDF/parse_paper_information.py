import joblib
import json
import os
from tqdm import tqdm

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# try:
#     from cogdl.oag import oagbert
# except:
#     from cogdl import oagbert

from get_paper_embedding.config import TFIDF_MODEL, SVD_MODEL #, fn_paper_text, fn_train, fn_valid, fn_test, 
from data_preprocess.util_data_preprocess import read_json_file, read_list_from_txt, write_json_file

basedir = os.path.dirname(__file__)


def pretain_LSA(tfidf_model_filename, svd_mode_filename):
    print('Using LSA to pretrain language for initial run.')
    paper_to_idx = {}
    corpus = []
    paper_info = json.load(open(fn_paper_text, 'r'))
    file_names = [fn_train, fn_valid, fn_test]
    count = 0
    for each_filename in file_names:
        records = json.load(open(each_filename, 'r'))
        for each_user in tqdm(records):
            for each_record in each_user:
                if each_record['item'] not in paper_to_idx:
                    if each_record['item'] in paper_info:
                        text = paper_info[each_record['item']]['title']
                        if text[-1] != '.':
                            text += '.'
                        if 'abstract' in paper_info[each_record['item']]:
                            text += paper_info[each_record['item']]['abstract']
                        paper_to_idx[each_record['item']] = count
                        corpus.append(text)
                        count += 1
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    X = svd.fit_transform(X)
    joblib.dump(vectorizer, tfidf_model_filename)
    joblib.dump(svd, svd_mode_filename)
    return vectorizer, svd


if os.path.exists(TFIDF_MODEL):
    vectorizer = joblib.load(TFIDF_MODEL)
    svd = joblib.load(SVD_MODEL)
else:
    vectorizer, svd = pretain_LSA(TFIDF_MODEL, SVD_MODEL)

# tokenizer, bert_model = oagbert("oagbert-v2")


def get_tfidf_emb(input_dict):
    text = input_dict['title']
    if len(text) > 0 and text[-1] != '.':
        text += '.'
    if 'abstract' in input_dict and input_dict['abstract'] is not None:
        text += input_dict['abstract']
    tf_idf_emb = vectorizer.transform([text])
    tf_idf_emb_zip = svd.transform(tf_idf_emb)
    return tf_idf_emb_zip.reshape(-1)


def load_paper_id_text_from_file(filepath):
    id_text_dict = {}
    paper_detail = read_json_file(filepath)
    for each_paper in paper_detail:
        id_text_dict[each_paper['_id']] = {
            'title': each_paper['Title'],
            'abstract': each_paper['Abstract'],
        }
    return id_text_dict


def get_paper_embedding(id_text_dict: dict, paper_mapping_dict: dict):
    paper_embedding_dict = {}
    total_number = len(id_text_dict)
    count = 1
    for key, value in id_text_dict.items():
        embedding = get_tfidf_emb(value)
        if embedding is not None and key in paper_mapping_dict.keys():
            print(str(count) + '/' + str(total_number), paper_mapping_dict[key], embedding.shape)
            paper_embedding_dict[paper_mapping_dict[key]] = list(embedding)
        count = count + 1
    return paper_embedding_dict


def get_reviewer_embedding(paper_embedding_dict, reviewer_record_dict):
    reviewer_embedding_dict = {}
    for key, value in reviewer_record_dict.items():
        reviewer_record = np.array([])
        for each_paper in value:
            if each_paper in paper_embedding_dict.keys():
                paper_embedding = paper_embedding_dict[each_paper]
                if len(reviewer_record) == 0:
                    reviewer_record = np.array(paper_embedding)
                else:
                    reviewer_record = np.vstack((reviewer_record, np.array(paper_embedding)))
        reviewer_embedding = np.mean(reviewer_record, axis=0)
        if isinstance(reviewer_embedding, np.ndarray):
            reviewer_embedding_dict[key] = list(reviewer_embedding)
    return reviewer_embedding_dict


def parse_paper_mapping_relationship(filepath):
    paper_mapping_dict = {}
    paper_mapping_list: list = read_list_from_txt(filepath)
    for each_paper_mapping in paper_mapping_list:
        paper_mapping_dict[each_paper_mapping.split(' ')[0]] = each_paper_mapping.split(' ')[1]
    return paper_mapping_dict


def get_reviewer_history(filepath):
    reviewer_record_dict = {}
    review_record_list: list = read_list_from_txt(filepath)
    for each_review_record in review_record_list:
        each_review_record_split = each_review_record.split(' ')
        if len(each_review_record_split) >= 2:
            reviewer_list = each_review_record_split[1:]
            paper_id = each_review_record_split[0]
            for each_reviewer in reviewer_list:
                if each_reviewer not in reviewer_record_dict.keys():
                    reviewer_record_dict[each_reviewer] = [paper_id]
                else:
                    reviewer_record_dict[each_reviewer].append(paper_id)
    return reviewer_record_dict


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


def find_top_k(matrix, k, axis=-1, largest=True, sorted=True):
    if axis is None:
        axis_size = matrix.size
    else:
        axis_size = matrix.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(matrix)
    if largest:
        index_array = np.argpartition(a, axis_size - k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k) - 1, axis=axis)
    else:
        index_array = np.argpartition(a, k - 1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis)
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices


def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]
    for i in range(pred_data.shape[0]):
        if np.sum(pred_data[i, :]) != 0:
            print(pred_data[i, :])
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg) / pred_data.shape[0]


def abc(reviewer_embedding_dict: dict,
        paper_embedding_dict: dict,
        top_k,
        review_history_dict: dict,
        review_history_training_dict: dict,
        ):
    paper_embedding_matrix = np.array([])
    reviewer_embedding_matrix = np.array([])
    paper_id_list = []
    reviewer_id_list = []

    for key, value in reviewer_embedding_dict.items():
        if len(reviewer_embedding_matrix) == 0:
            reviewer_embedding_matrix = np.array(value)
        else:
            reviewer_embedding_matrix = np.vstack((reviewer_embedding_matrix, np.array(value)))
        reviewer_id_list.append(key)
    for key, value in paper_embedding_dict.items():
        if len(paper_embedding_matrix) == 0:
            paper_embedding_matrix = np.array(value)
        else:
            paper_embedding_matrix = np.vstack((paper_embedding_matrix, np.array(value)))
        paper_id_list.append(key)
    cos_similar_matrix = get_cos_similar_matrix(paper_embedding_matrix, reviewer_embedding_matrix)

    _, topk_indices = find_top_k(cos_similar_matrix, top_k, largest=True, sorted=True)

    recall_total = 0
    precision_total = 0
    paper_number = 0
    label_all = []
    pred_all = []
    for i in range(len(paper_id_list)):
        paper_id = paper_id_list[i]
        top_k_reviewer_index_list = topk_indices[i]
        top_k_reviewer_id_list = np.array(reviewer_id_list)[np.array(top_k_reviewer_index_list)]

        assert len(top_k_reviewer_id_list) == top_k
        top_k_reviewer_id_list_float = []
        for top_k_reviewer_id in top_k_reviewer_id_list:
            top_k_reviewer_id_list_float.append(float(top_k_reviewer_id))

        if paper_id in review_history_training_dict.keys():
            training_reviewer_id_list = review_history_training_dict[paper_id]
            top_k_reviewer_id_list = list(set(top_k_reviewer_id_list) - set(training_reviewer_id_list))

            reviewer_ground_truth_list = review_history_dict[paper_id]
            recall = len(list(set(reviewer_ground_truth_list).intersection(set(top_k_reviewer_id_list)))) / len(
                reviewer_ground_truth_list)
            precision = len(list(set(reviewer_ground_truth_list).intersection(set(top_k_reviewer_id_list)))) / len(
                top_k_reviewer_id_list)

            label_one = []
            for i in reviewer_ground_truth_list:
                if i != '':
                    label_one.append(int(i))
            label_all.append(label_one)
            pred_one = []
            for i in top_k_reviewer_id_list:
                if str(int(i)) in reviewer_ground_truth_list:
                    pred_one.append(1)
                else:
                    pred_one.append(0)
            while len(pred_one) < 20:
                pred_one.append(0)
            pred_all.append(pred_one)

            recall_total = recall_total + recall
            precision_total += precision
            paper_number = paper_number + 1
    ndcg = NDCGatK_r(label_all, np.array(pred_all), top_k)
    return recall_total / paper_number, precision_total / paper_number, ndcg


def parse_all_reviewer_for_one_submission(filepath):
    review_record_list = read_list_from_txt(filepath)
    reviewer_dict = {}
    # submission: reviewer_1, reviewer_2, reviewer_3, reviewer_4
    for each_review_record in review_record_list:
        if len(each_review_record.split(' ')) >= 2:
            reviewer_dict[each_review_record.split(' ')[0]] = each_review_record.split(' ')[1:]
    return reviewer_dict


if __name__ == '__main__':
    paper_filepath = r'get_paper_embedding/data/paper_detail.json'
    paper_reviewer_training_part_filepath = r'get_paper_embedding/data/train.txt'
    paper_reviewer_full_filepath = r'get_paper_embedding/data/full.txt'

    submission_history = r'get_paper_embedding/data/submission_list.txt'
    paper_mapping_dict = parse_paper_mapping_relationship(submission_history)
    id_text_dict = load_paper_id_text_from_file(paper_filepath)
    print('data loaded')

    top_k = 20
    training_reviewer_embedding_dict = read_json_file(
        r'get_paper_embedding/embedding_data/training_reviewer_embedding.json')
    paper_embedding_dict = read_json_file(
        r'get_paper_embedding/embedding_data/paper_embedding.json')
    print('Embedding data loaded')
    review_history_dict: dict = parse_all_reviewer_for_one_submission(paper_reviewer_full_filepath)
    review_history_training_dict: dict = parse_all_reviewer_for_one_submission(paper_reviewer_training_part_filepath)

    recall_total, precision_total, ndcg_total = abc(training_reviewer_embedding_dict, paper_embedding_dict, top_k,
                                                    review_history_dict, review_history_training_dict)
    print(recall_total, precision_total, ndcg_total)
    # 20：recall-0.0016197136362882253; precision-0.00026145657143040074; ndcg-0.0010986330642514023
    # 50：recall-0.003551578413456859
