import openai
import spacy

def retrieve_contexts(index, vector, top_k=10):
    res = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [item['metadata']['text'] for item in res['matches']]

def retrieve_contexts_with_metadata(index, vector, top_k=10):
    res = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return res['matches']

def filter_contexts(contexts, keyword):
    return [context for context in contexts if keyword in context]


def filter_contexts2(contexts, keyword, similarity_threshold = .7):
    
    assert isinstance(similarity_threshold, float)
    assert similarity_threshold <= 1.00
    
    nlp = spacy.load("en_core_web_sm")

    keyword_doc = nlp(keyword)
    filtered_contexts = []
    for context in contexts:
        context_doc = nlp(context)
        if context_doc.similarity(keyword_doc) > similarity_threshold:  # Similarity threshold
            filtered_contexts.append(context)
    return filtered_contexts

def augment_query(query, filtered_contexts):
    augmented_query = "\n\n---\n\n".join([f"Context {i+1}:\n{context}" for i, context in enumerate(filtered_contexts)]) + "\n\n-----\n\n" + query
    return augmented_query

def generate_response(primer, augmented_query, client, model="gpt-4o-mini"):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": primer},
                    {"role": "user", "content": augmented_query}
                ]
            )
            return res['choices'][0]['message']['content']
        except openai.error.RateLimitError:
            if attempt == max_retries - 1:  # Last attempt
                raise  # Re-raise if all retries failed
            time.sleep(20 * (attempt + 1))  # Wait longer between retries