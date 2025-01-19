import tiktoken  # Used for counting tokens
import openai
import time
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

def get_embeddings(text, client, model="text-embedding-3-small"):
    chunks = chunk_text(text)
    embeddings = []
    max_retries = 3
    
    for chunk in chunks:
        for attempt in range(max_retries):
            try:
                response = openai.Embedding.create(
                    input=chunk,
                    model=model
                )
                embeddings.append(response['data'][0]['embedding'])
                # Add a small delay between requests
                time.sleep(1)
                break  # Success, exit retry loop
            except openai.error.RateLimitError:
                if attempt == max_retries - 1:  # Last attempt
                    raise  # Re-raise if all retries failed
                time.sleep(20 * (attempt + 1))  # Wait longer between retries
    
    return embeddings

if __name__ == "__main__":
    pass
