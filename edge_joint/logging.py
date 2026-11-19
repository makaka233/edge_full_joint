import json


def emit(path, row):
    with path.open('a', encoding='utf8', newline='\n') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
