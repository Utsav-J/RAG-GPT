import gradio as gr
import time
import re
import os
import ast, html
import openai
from langchain.vectorstores.chroma import Chroma
from utils.load_config import LoadConfig
from typing import List, Tuple

CONFIG = LoadConfig()
URL = "https://github.com/Farzad-R/LLM-Zero-to-Hundred/tree/master/RAG-GPT"
hyperlink = f"[RAG-GPT user guideline]({URL})"

class Chatbot:
    """ Retrieval and answer generation abilities """
    @staticmethod
    def respond(chatbot:List, message:str, data_type:str = "Preprocessed docs", temperature:float = 0.0) -> Tuple:
        """
        Parameters:
            chatbot (List): List representing the chatbot's conversation history.
            message (str): The user's query.
            data_type (str): Type of data used for document retrieval ("Preprocessed doc" or "Upload doc: Process for RAG").
            temperature (float): Temperature parameter for language model completion.
        
        Returns:
            Tuple: A tuple containing an empty string, the updated chat history, and references from retrieved documents.
        """
        if data_type == "Preprocessed docs":
            if os.path.exists(CONFIG.persist_directory):
                vectordb = Chroma(persist_directory=CONFIG.persist_directory, embedding_function= CONFIG.embedding_model)
            else:
                chatbot.append("VectorDB doesnt exist, please first upload the files from the upload_files_manually.py")
                return "",chatbot,None
        elif data_type == "Upload doc: Process for RAG":
            if os.path.exists(CONFIG.persist_directory):
                vectordb = Chroma(embedding_function=CONFIG.embedding_model, persist_directory=CONFIG.persist_directory)
            else:
                chatbot.append("No vector DB was found, upload your files using the upload button to proceeed")
                return "", chatbot, None

        docs = vectordb.similarity_search(message, CONFIG.k)
        print(docs)
        question = "# User new question:\n" + message
        retrieved_content = Chatbot.clean_references(docs)
        chat_history = f"Chat history:\n {str(chatbot[-CONFIG.number_of_q_a_pairs])}\n\n"
        prompt = f"{chat_history}{retrieved_content}{question}"
        print("+++++++++++++++++")
        print(prompt)


        response = openai.chat.completions.create(
            messages=[
                {"role": "system", "content": CONFIG.llm_system_role},
                {"role": "user", "content": prompt}
            ],
            model="gpt-3.5-turbo",
            temperature=temperature
        )
        chatbot.append(message, response.choices[0].message.content)
        time.sleep(2)

        return "",chatbot,retrieved_content
    
    @staticmethod
    def clean_references(docs: List) -> str:
        """
        Clean and format the reference documents that are gathered from the similarity search
        """

        server_url = "http://127.0.0.1:7860"
        documents = [str(x)+"\n\n" for x in docs]
        markdown_documents = ""
        counter = 1

        for i in documents:
            # content after "page_content=" up until the " metadata=" part
            # The entire metadata part, including the curly braces.
            content,metadata = re.match(r"page_content=(.?)(metadata=\{.\})").groups()
            metadata = metadata.split('=',1)[1]
            metadata_dict = ast.literal_eval(metadata)

            content = bytes(content, "utf-8").decode("unicode_escape")

            content = re.sub(r"\\n",'\n', content)
            content = re.sub(r'\s*<EOS>\s*<pad>\s*', ' ', content)
            content = re.sub(r'\s+', ' ', content).strip()

            content = html.unescape(content)
            content = content.encode('latin1').decode('utf-8', 'ignore')

            # Remove or replace special characters and mathematical symbols
            # This step may need to be customized based on the specific symbols in your documents
            content = re.sub(r'â', '-', content)
            content = re.sub(r'â', '∈', content)
            content = re.sub(r'Ã', '×', content)
            content = re.sub(r'ï¬', 'fi', content)
            content = re.sub(r'â', '∈', content)
            content = re.sub(r'Â·', '·', content)
            content = re.sub(r'ï¬', 'fl', content)

            pdf_url = f"{server_url}/{os.path.basename(metadata_dict['source'])}"

            markdown_documents += f"# Retrieved content {counter}:\n" + content + "\n\n" + \
                f"Source: {os.path.basename(metadata_dict['source'])}" + " | " +\
                f"Page number: {str(metadata_dict['page'])}" + " | " +\
                f"[View PDF]({pdf_url})" "\n\n"
            counter += 1
        return markdown_documents
