import tiktoken  # Used for counting tokens
import openai

def chunk_text(text, max_tokens=4000):
    # Initialize the tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Tokenize the entire text
    tokens = enc.encode(text)
    
    # Split the tokens into chunks of max_tokens size
    token_chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    
    # Convert each chunk of tokens back into text for embedding
    text_chunks = [enc.decode(chunk) for chunk in token_chunks]
    
    return text_chunks

def get_embeddings(text, client):
    chunks = chunk_text(text)
    embeddings = []
    
    for chunk in chunks:
        response = openai.Embedding.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        embeddings.append(response['data'][0]['embedding'])
    
    return embeddings

if __name__ == "__main__":
    pass