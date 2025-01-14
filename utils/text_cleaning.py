import re

def remove_substrings(s, substrings_to_remove, replacement_string):
    for sub in substrings_to_remove:
        s = s.replace(sub, replacement_string)
    return s

def collapse_spaces(s):
    return re.sub(r'\s+', '\n', s).strip()

def read_text_file(file_path, encoding='utf-8'):
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()

def clean_text(text):
    text = remove_substrings(text, {"/C20", "\n"}, " ")
    text = collapse_spaces(text)
    return text

if __name__ == "__main__":
    pass