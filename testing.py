import re

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
    # Define the regex pattern for enumerations
    pattern = r"(?<!\w)(\d+\.|[A-Za-z]\.)(?=\s)"
    # Use re.split to split on the pattern
    parts = re.split(pattern, text)
    # Combine enumeration markers with their respective text
    result = []
    for i in range(0, len(parts) - 1, 2):
        result.append(parts[i].strip() + " " + parts[i + 1].strip())
    if len(parts) % 2 != 0:
        result.append(parts[-1].strip())
    return result

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

def list_to_text(text_list, printout:bool = True):

    text = '\n'.join(text_list)

    if printout:
        print(text)

    return None

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

    list_to_text(text_list)

    # If separate chunks starts with # A. B. C. in a row then those should be part of the same chunk
    # Same thing for 1. 2. 3. and I. II. etc.
    text_list = [split_on_enumeration(elem) for elem in text_list]
    text_list = flatten_list(nested_list=text_list)
    text_list = merge_bullet_points(text_list)
    text_list = [string for sublist in text_list for string in sublist]

    list_to_text(text_list)

    # So if a piece of text is short than 40 characters, we append it to the next chunk
    text_list = merge_short_sentences(text_list)

    # At this point we will anticipate having some chunks which are too long to be parsed so we will have to develop a traditional overlap chunking method

    text_list = merge_short_sentences(text_list)
    
    with open("chunking_testing_output.txt", mode='w') as file:
        for i, chunk in enumerate(text_list, 1):
            file.write(f"Chunk {i}:\n{chunk}\n{'-' * 80}\n")
    
    print("Complete, status: 0")

if __name__ == "__main__":
    main()

