from utils.prepare_vectordb import PrepareVectorDB
from typing import List, Tuple
from utils.load_config import LoadConfig
from utils.summarizer import Summarizer

CONFIG = LoadConfig()


class UploadFile:

    @staticmethod
    def process_uploaded_files(files_dir: List, chatbot: List, rag_with_dropdown: str) -> Tuple:

        if rag_with_dropdown == "Upload doc: Process for RAG":
            prepare_vectordb_instance = PrepareVectorDB(data_directory=files_dir,
                                                        persist_directory=CONFIG.custom_persist_directory,
                                                        embedding_model_engine=CONFIG.embedding_model_engine,
                                                        chunk_size=CONFIG.chunk_size,
                                                        chunk_overlap=CONFIG.chunk_overlap)
            prepare_vectordb_instance.prepare_and_save_vectordb()
            chatbot.append(
                (" ", "Uploaded files are ready. Please ask your question"))
        elif rag_with_dropdown == "Upload doc: Give Full summary":
            final_summary = Summarizer.summarize_the_pdf(file_dir=files_dir[0],
                                                         max_final_token=CONFIG.max_final_token,
                                                         token_threshold=CONFIG.token_threshold,
                                                         gpt_model=CONFIG.llm_engine,
                                                         temperature=CONFIG.temperature,
                                                         summarizer_llm_system_role=CONFIG.summarizer_llm_system_role,
                                                         final_summarizer_llm_system_role=CONFIG.final_summarizer_llm_system_role,
                                                         character_overlap=CONFIG.character_overlap)
            chatbot.append(
                (" ", final_summary))
        else:
            chatbot.append(
                (" ", "If you would like to upload a PDF, please select your desired action in 'rag_with' dropdown."))
        return "", chatbot