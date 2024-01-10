import os
import json

def save_list_to_file(input_list: list, file_path):
    file = open(file_path, 'w')
    file.write("\n".join(str(item) for item in input_list))
    file.close()

def save_string_to_file(text, file_path):
    file = open(file_path, 'w')
    file.write(text)
    file.close()

def read_file_to_set(file_path):
    content = set()
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            for l in  file.readlines():
                content.add(l.strip())
    return content

def read_file_to_list(file_path):
    content = list()
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            for l in file.readlines():
                content.append(l.strip().replace("\n", ""))
    return content

def read_json_file(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data


def path_exists(dir_path):
    return os.path.exists(dir_path)

def create_folder(dir_path):
    try:
        os.mkdir(dir_path)
    except:
        pass
