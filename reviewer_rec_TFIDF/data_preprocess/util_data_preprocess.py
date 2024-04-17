import bson
import json
import re


def extract_author_id_from_url(url):
    return re.findall(r"\d+\.?\d*", url)[0]


def read_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as load_f:
        json_file = json.load(load_f)
    return json_file


def write_json_file(filename, dict_obj):
    with open(filename, 'w', encoding='utf-8') as write_f:
        write_f.write(json.dumps(dict_obj, indent=4))
    return True


def read_bson_file(filepath):
    bson_file = open(filepath, 'rb')
    bson_data = bson.decode_all(bson_file.read())
    return bson_data


def extract_author_id(json_data):
    collected_data = set()
    for key, value in json_data.items():
        for each_author_id in value:
            collected_data.add(each_author_id)
    collected_data = list(collected_data)
    return collected_data


def write_list_to_txt(file_path, list_data):
    with open(file_path, "a+") as f:
        for each_data in list_data:
            f.writelines(each_data + '\n')
    f.close()


def read_list_from_txt(filepath):
    datalist = []
    f = open(filepath)
    line = f.readline().rstrip('\n')
    while line:
        datalist.append(line)
        line = f.readline().rstrip('\n')
    f.close()
    return datalist


