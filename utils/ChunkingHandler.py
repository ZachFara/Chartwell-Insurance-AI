import re
from typing import List, Union

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

    def overlap_wordcount_chunking(self, max_words=100, overlap=0):
        combined_text = ' '.join(self.chunks) if self.chunks else self.text
        words = combined_text.split()
        final_chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i + max_words]
            chunk_text = ' '.join(chunk_words)
            final_chunks.append(chunk_text)
            if overlap > 0:
                i += max_words - overlap
            else:
                i += max_words
        self.chunks = [c.strip() for c in final_chunks if c.strip()]
        return self

if __name__ == '__main__':
    print('Imported ChunkingHandler.py')
