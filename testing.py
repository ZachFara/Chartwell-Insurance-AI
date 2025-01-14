import re
from typing import List, Union

def read_file(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        return file.read()

def split_by_headers(input_text):
    lines = input_text.split('\n')
    groups = []
    current_group = []

    for line in lines:
        # Remove punctuation for the uppercase check
        stripped_line = re.sub(r'[^\w\s]', '', line)  # Remove non-alphanumeric and non-space characters
        if stripped_line.isupper() and stripped_line.strip():  # Check if stripped line is uppercase and not empty
            if current_group:
                groups.append(' '.join(current_group))  # Save the current group
                current_group = []
        current_group.append(line)

    if current_group:
        groups.append(' '.join(current_group))  # Add the last group

    return groups

def merge_short_sentences(text_list, char_limit=40):
    merged_list = []
    buffer = ""

    for text in text_list:
        if len(buffer + " " + text if buffer else text) <= char_limit:
            buffer = buffer + " " + text if buffer else text
        else:
            if buffer:
                merged_list.append(buffer.strip())
            buffer = text

    if buffer:
        merged_list.append(buffer.strip())
    
    return merged_list

def merge_bullet_points(text_list):
    def parse_enum(x):
        x = x.strip()
        if len(x) >= 2 and x[1] == '.':
            if x[0].isdigit(): 
                return ('digit', x[0])
            if x[0].isalpha():
                return ('letter', 'upper' if x[0].isupper() else 'lower', x[0])
        return None

    merged = []
    current = []
    for text in text_list:
        e = parse_enum(text)
        if e:
            current.append(text)
        else:
            if current:
                if text.strip():  # Include only if text is non-empty
                    current.append(text)
                merged.append(current)
                current = []
            elif text.strip():
                merged.append([text])
    if current:
        merged.append(current)
    return merged

def split_on_enumeration(text):
    def is_enumeration(line):
        return bool(re.match(r"^\s*[A-Za-z0-9]+\.", line.strip()))

    lines = [l for l in text.split('\n') if l.strip()]
    groups, current = [], []
    random_count = 0

    for line in lines:
        if is_enumeration(line):
            if random_count == 1 and current:
                groups.append("\n".join(current))
                current = []
            current.append(line)
            random_count = 0
        else:
            random_count += 1
            if random_count > 1 and current:
                groups.append("\n".join(current))
                current = []
                random_count = 1
            current.append(line)

    if current: groups.append("\n".join(current))
    return groups

def flatten_list(nested_list):
    result = []
    def recursive_flatten(sublist):
        for item in sublist:
            if isinstance(item, list):
                recursive_flatten(item)
            else:
                result.append(item)
    recursive_flatten(nested_list)
    return result

def list_to_text(text_list, printout:bool = True, seperator:str = '\n'):

    text = seperator.join(text_list)

    if printout:
        print(text)

    return text

def move_ending_all_caps_line(chunks):
    def is_all_caps_line(line):
        return line.strip().isupper()
    
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        lines = chunk.split("\n")
        if lines and is_all_caps_line(lines[-1]):
            last_line = lines.pop()
            if i + 1 < len(chunks):  # If there's a next chunk
                next_chunk_lines = chunks[i + 1].split("\n")
                next_chunk_lines.insert(0, last_line)
                chunks[i + 1] = "\n".join(next_chunk_lines)
            chunk = "\n".join(lines)
        processed_chunks.append(chunk)
    return processed_chunks

def remove_duplicate_chunks(chunks):
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk not in seen:
            unique_chunks.append(chunk)
            seen.add(chunk)
    return unique_chunks

class ChunkingHandler:
    def __init__(self, text) -> None:
        self.text = text
        self.chunks = None

    def remove_empty(self, text_list: List[str]):
        self.chunks = [elem for elem in text_list if elem.strip()]
        return self

    def list_to_text(self, text_list=None, printout: bool = False, seperator: str = '\n'):
        if text_list is None:
            assert self.chunks is not None
            text_list = self.chunks
        self.text = seperator.join(text_list)
        if printout:
            print(self.text)
        return self

    def split_by_headers(self):
        assert self.text is not None
        lines = self.text.split('\n')
        groups = []
        current_group = []

        for line in lines:
            stripped_line = re.sub(r'[^\w\s]', '', line)
            if stripped_line.isupper() and stripped_line.strip():
                if current_group:
                    groups.append(' '.join(current_group))
                    current_group = []
            current_group.append(line)

        if current_group:
            groups.append(' '.join(current_group))

        self.chunks = groups
        return self

    def remove_header_notation(self, headers: Union[str, List[str]]):
        assert self.chunks is not None
        if isinstance(headers, str):
            if headers == "WHITESPACE":
                self.chunks = [chunk.lstrip() for chunk in self.chunks]
            else:
                self.chunks = [chunk.lstrip(headers) for chunk in self.chunks]
        else:
            for header in headers:
                if header == "WHITESPACE":
                    self.chunks = [chunk.lstrip() for chunk in self.chunks]
                else:
                    self.chunks = [chunk.lstrip(header) for chunk in self.chunks]
        return self

    def split_on_enumeration(self, text=None):
        if text is None:
            text = self.text

        def is_enumeration(line):
            return bool(re.match(r"^\s*[A-Za-z0-9]+\.", line.strip()))

        lines = [l for l in text.split('\n') if l.strip()]
        groups, current = [], []
        random_count = 0

        for line in lines:
            if is_enumeration(line):
                if random_count == 1 and current:
                    groups.append("\n".join(current))
                    current = []
                current.append(line)
                random_count = 0
            else:
                random_count += 1
                if random_count > 1 and current:
                    groups.append("\n".join(current))
                    current = []
                    random_count = 1
                current.append(line)

        if current:
            groups.append("\n".join(current))

        self.chunks = groups
        return self

    def merge_short_sentences(self, char_limit=40):
        assert self.chunks is not None
        merged_list = []
        buffer = ""

        for text in self.chunks:
            candidate = (buffer + " " + text) if buffer else text
            if len(candidate) <= char_limit:
                buffer = candidate
            else:
                if buffer:
                    merged_list.append(buffer.strip())
                buffer = text

        if buffer:
            merged_list.append(buffer.strip())

        self.chunks = merged_list
        return self

    def move_ending_all_caps_line(self):
        assert self.chunks is not None
        def is_all_caps_line(line):
            return line.strip().isupper()

        processed_chunks = []
        for i, chunk in enumerate(self.chunks):
            lines = chunk.split("\n")
            if lines and is_all_caps_line(lines[-1]):
                last_line = lines.pop()
                if i + 1 < len(self.chunks):
                    nxt_lines = self.chunks[i + 1].split("\n")
                    nxt_lines.insert(0, last_line)
                    self.chunks[i + 1] = "\n".join(nxt_lines)
                chunk = "\n".join(lines)
            processed_chunks.append(chunk)

        self.chunks = processed_chunks
        return self

    def remove_duplicates(self):
        assert self.chunks is not None
        seen = set()
        unique_chunks = []
        for chunk in self.chunks:
            if chunk not in seen:
                unique_chunks.append(chunk)
                seen.add(chunk)
        self.chunks = unique_chunks
        return self
    

def main2():
    document_name = 'document_texts.txt'
    text = read_file(document_name)

    # 1) Initialize the handler
    handler = ChunkingHandler(text)
    
    # 2) Remove empty lines from the raw file input
    handler.remove_empty(handler.text.split('\n')) \
           .list_to_text(seperator='\n') \
    # 3) Split by headers (all-caps blocks)
    handler.split_by_headers() \
    # 4) Remove '#' at the beginning and extra whitespace
    handler.remove_header_notation('#') \
           .remove_header_notation('WHITESPACE') \
           .list_to_text(seperator='\n') \
    # 5) Split on enumerations (A. 1. a. etc.)
    handler.split_on_enumeration() \
           .merge_short_sentences(char_limit=40) \
           .move_ending_all_caps_line() \
           .remove_empty(handler.chunks) \
           .remove_duplicates()

    # 10) Write the final chunks
    with open("chunking_testing_output.txt", mode='w') as file:
        for i, chunk in enumerate(handler.chunks, 1):
            file.write(f"Chunk {i}:\n{chunk}\n{'-' * 80}\n")

    print("Complete, status: 0")
   
def main():
    # Read the document in as a text file
    document_name = 'document_texts.txt'
    text = read_file(document_name)

    # Split the text by newline
    text_list = text.split('\n')
    
    # Remove some empty strings that I am not sure how they got created
    text_list = [elem for elem in text_list if len(elem) > 0]

    # Split the text up by places where there is a all caps header
    text = '\n'.join(text_list)
    text_list = split_by_headers(text)

    # If there is a # at the beginning of a element let's remove it since that can mess up the split on enumeration function
    text_list = [elem.lstrip("#") for elem in text_list]
    
    x = 0
    
    text_list = [elem.lstrip() for elem in text_list]
    
    x = 0


    # If separate chunks starts with # A. B. C. in a row then those should be part of the same chunk
    # Same thing for 1. 2. 3. and I. II. etc.
    text = list_to_text(text_list, printout = False)
    text_list = split_on_enumeration(text)

    text_list = merge_short_sentences(text_list)

    text_list = move_ending_all_caps_line(text_list)

    text_list = [elem for elem in text_list if len(elem) > 0]

    text_list = remove_duplicate_chunks(text_list)
    
    with open("chunking_testing_output.txt", mode='w') as file:
        for i, chunk in enumerate(text_list, 1):
            file.write(f"Chunk {i}:\n{chunk}\n{'-' * 80}\n")
    
    print("Complete, status: 0")

if __name__ == "__main__":
    main2()


