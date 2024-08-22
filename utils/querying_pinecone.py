
def retrieve_contexts(index, vector, top_k=10):
    res = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [item['metadata']['text'] for item in res.matches]

def filter_contexts(contexts, keyword):
    return [context for context in contexts if keyword in context]

def augment_query(query, filtered_contexts):
    augmented_query = "\n\n---\n\n".join(filtered_contexts) + "\n\n-----\n\n" + query
    return augmented_query

def generate_response(primer, augmented_query, client):
    res = client.chat.completions.create(model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ])

    return res.choices[0].message.content
