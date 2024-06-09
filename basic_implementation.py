import os
import torch
from PyPDF2 import PdfReader, PdfWriter
from sentence_transformers import SentenceTransformer
from transformers import pipeline

'''
This is an example of an RAG framework at a very raw level using minimal package
In-Memory implementation of a vector DB
Using a sentence embedder and a in house logic to perform similarity search and returning the matched document to 
support the user query


'''
file_name = "ConceptsofBiology-WEB.pdf"
os.makedirs(os.path.join(os.getcwd(), "data_store"), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), "processed_data"), exist_ok=True)
inp_file = os.path.join(os.getcwd(), "data_store", file_name)
out_file = os.path.join(os.getcwd(), "processed_data", "Concepts_of_Biology.pdf")

# Task 1 To make indexing faster, you can pick any 2 chapters from the pdf and treat it as a
# source.
# Breaking down the first pdf into the first two chapters [Page 18-68 ]
writer = PdfWriter()
with open(inp_file, 'rb') as inp:
    reader = PdfReader(inp)
    page = 18
    while page < 68:
        writer.add_page(reader.pages[page])
        page += 1
with open(out_file, 'wb') as outfile:
    writer.write(outfile)


inp_reader = PdfReader(out_file)
page_data = []
# Reading the pdf
for i in range(len(inp_reader.pages)):
    pg = inp_reader.pages[i]  # Page Object
    page_txt = pg.extract_text()
    # print(len(page_txt))
    page_data.append(page_txt)

complete_text = ' '.join(page_data)

# Defining the chuning logic
CHUNK_LENGTH = 2000

chunks = [complete_text[i:i+CHUNK_LENGTH] for i in range(0, len(complete_text), CHUNK_LENGTH)]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Vector embedding storage
chunk_vector_db = {f"chunk_{i}": {chunk: embedding_model.encode(chunk)} for i, chunk in enumerate(chunks)}

# chunk_embeddings = {embedding_model.encode(chunk): chunk for chunk in chunks}  # Vector embedding storage

user_query = input("Enter your query")
user_query_embedding = embedding_model.encode(user_query)

top_embedding = ''
matched_similarity = {'matched_context': ''}
max_similarity = 0
for embed, matched_embedding_chunk in chunk_vector_db.items():
    embedding = list(matched_embedding_chunk.values())[0]
    chunk = list(matched_embedding_chunk.keys())[0]
    similarity_score = embedding_model.similarity(embedding, user_query_embedding)[0][0].tolist()
    if similarity_score > max_similarity:
        matched_similarity['matched_context'] = chunk
print(matched_similarity)
model_id = "google/flan-t5-base"
pipelines = pipeline("text2text-generation", model=model_id, device='cuda',
                     model_kwargs={"torch_dtype": torch.bfloat16})
system_prompt = "Given the context answer the question that follow"
context = matched_similarity['matched_context']
prompt = f'''
{system_prompt}
context: {context}
question: {user_query}
'''
with torch.no_grad():
    result = pipelines(prompt, temperature=0.01, output_scores=True)
    print(result[0]["generated_text"])
