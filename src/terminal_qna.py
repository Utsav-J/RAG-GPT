import os
import openai
import yaml
from utils.load_config import LoadConfig
from typing import List, Tuple
from pyprojroot import here
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

CONFIG = LoadConfig()
with open(here("configs/app_config.yml")) as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)

embeddings = OpenAIEmbeddings()
vector_db = Chroma(persist_directory=CONFIG.persist_directory, embedding_function=embeddings)
print("Number of vectors in vector_db:", vector_db._collection.count())

llm_engine = app_config['llm_config']['engine']
llm_system_role = app_config['llm_config']['llm_system_role']
llm_temperature = app_config['llm_config']['temperature']

persist_directory = str(here(app_config['directories']['persist_directory']))
k = app_config['retrieval_config']['k']

while True:
    question = input("\n\nEnter your question or press 'q' to exit: ")
    if question.lower() =='q':
        break
    question = "# user new question:\n" + question
    docs = vector_db.similarity_search(question, k=CONFIG.k)
    retrieved_docs_page_content: List[Tuple] = [
        str(x.page_content)+"\n\n" for x in docs]
    retrived_docs_str = "# Retrieved content:\n\n" + str(retrieved_docs_page_content)
    prompt = retrived_docs_str + "\n\n" + question
    response = openai.chat.completions.create(
        messages=[
            {"role": "system", "content": CONFIG.llm_system_role},
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo"
    )
    print(response.choices[0].message.content)