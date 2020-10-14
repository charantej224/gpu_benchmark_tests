import json


def write_file(filename, content):
    with open(filename, 'a+') as f:
        f.write(content)
        f.close()


def write_dict(filename, dict_json):
    with open(filename, 'w') as f:
        json.dump(dict_json, f, indent=2)
        f.close()
