from tpms import *
from metrics import *
from parse import parse_args

import numpy as np
import json
import os

parse = parse_args()

test_inner_file = parse.test_inter_path
train_inner_file = parse.train_inter_path

paper_info_file = parse.pap_info_path
reviewer_info_file = parse.rev_info_path

with open(reviewer_info_file, encoding='utf-8') as f:
    reviewer_info = json.load(f)
rev_id_dict = {}
for i in range(len(reviewer_info)):
    if reviewer_info[i]['u_newid'] not in list(rev_id_dict.keys()):
        rev_id_dict[reviewer_info[i]['u_newid']] = reviewer_info[i]

with open(paper_info_file, encoding='utf-8') as f:
    paper_info = json.load(f)
paper_id_dict = {}
for i in range(len(paper_info)):
    if paper_info[i]['_newid'] not in list(paper_id_dict.keys()):
        paper_id_dict[paper_info[i]['_newid']] = paper_info[i]


train_inner_pap_rev_dict = {}
test_inner_pap_rev_dict = {}
train_inner_rev_pap_dict = {}
test_inner_rev_pap_dict = {}

try:
    file = open(train_inner_file, encoding='utf-8')  # 所有paper对应的reviewer的交互记录文件，第一列是paperID，后几列是reviewerID
except FileNotFoundError:
    print("file is not found")
else:
    contents = file.readlines()
    for content in contents:  # 依次处理full文件中的每一行交互记录，即一篇paper对应的reviewer记录
        interaction = content.split()  # 以空格为分隔符，同时将元素存储为列表interaction的元素
        pap_id = interaction[0]
        rev_id_list = interaction[1:]
        if pap_id not in train_inner_pap_rev_dict.keys():
            train_inner_pap_rev_dict[pap_id] = rev_id_list
        for rev_id in rev_id_list:
            if rev_id not in train_inner_rev_pap_dict.keys():
                train_inner_rev_pap_dict[rev_id] = []
            train_inner_rev_pap_dict[rev_id].append(pap_id)

try:
    file = open(test_inner_file, encoding='utf-8')  # 所有paper对应的reviewer的交互记录文件，第一列是paperID，后几列是reviewerID
except FileNotFoundError:
    print("file is not found")
else:
    contents = file.readlines()
    for content in contents:  # 依次处理full文件中的每一行交互记录，即一篇paper对应的reviewer记录
        interaction = content.split()  # 以空格为分隔符，同时将元素存储为列表interaction的元素
        pap_id = interaction[0]
        rev_id_list = interaction[1:]
        if pap_id not in test_inner_pap_rev_dict.keys():
            test_inner_pap_rev_dict[pap_id] = rev_id_list
        for rev_id in rev_id_list:
            if rev_id not in test_inner_rev_pap_dict.keys():
                test_inner_rev_pap_dict[rev_id] = []
            test_inner_rev_pap_dict[rev_id].append(pap_id)


paper_id_dict_test = {}
paper_id_dict_train = {}
rev_id_dict_test = {}

for pap_id_test in test_inner_pap_rev_dict.keys():
    paper_id_dict_test[pap_id_test] = paper_id_dict[pap_id_test]
for pap_id_train in train_inner_pap_rev_dict.keys():
    paper_id_dict_train[pap_id_train] = paper_id_dict[pap_id_train]
for rev_id_test in test_inner_rev_pap_dict.keys():
    rev_id_dict_test[rev_id_test] = rev_id_dict[rev_id_test]

paper_profiles = get_paper_profiles(paper_id_dict_test, 'title')
reviewer_profiles = get_reviewer_profiles(rev_id_dict, paper_id_dict_train, train_inner_rev_pap_dict, 'title')

similarities = compute_similarities(paper_profiles, reviewer_profiles)


ndcg20 = []
recall20 = []

for pap_id in similarities.keys():
    pap_rev_ids = list(similarities[pap_id].keys())
    pap_rev_score = list(similarities[pap_id].values())
    sorted_lists = sorted(zip(pap_rev_ids, pap_rev_score), key=lambda x: x[1], reverse=True)
    pap_rev_ids_pred = [x[0] for x in sorted_lists]

    ndcg20.append(calculate_ndcg_at_k(pap_rev_ids_pred, test_inner_pap_rev_dict[pap_id], k=20))
    recall20.append(calculate_recall_at_k(pap_rev_ids_pred, test_inner_pap_rev_dict[pap_id], k=20))

print("NDCG@20: {:.2f}, Recall@20: {:.2f}".format(np.mean(ndcg20), np.mean(recall20)))