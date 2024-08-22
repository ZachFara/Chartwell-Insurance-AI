import re

def remove_substrings(s, substrings_to_remove, replacement_string):
    for sub in substrings_to_remove:
        s = s.replace(sub, replacement_string)
    return s

def collapse_spaces(s):
    return re.sub(r'\s+', ' ', s).strip()

if __name__ == "__main__":
    pass