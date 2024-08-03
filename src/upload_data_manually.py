import os
from utils.prepare_vectordb import PrepareVectorDB
from utils.load_config import LoadConfig

CONFIG = LoadConfig()

def upload_data_manually() -> None: 

    """
    
    """
    prepareVectorDbInstance = PrepareVectorDB(
        data_directory= CONFIG.data_directory,
        chunk_overlap= CONFIG.chunk_overlap,
        chunk_size= CONFIG.chunk_size,
        embedding_model_engine= CONFIG.embedding_model_engine,
        persist_directory= CONFIG.persist_directory
    )

    # if not len(os.listdir(CONFIG.persist_directory)) != 0:
    if len(os.listdir(CONFIG.persist_directory)) == 0:
        prepareVectorDbInstance.prepare_and_save_vectorDB()
    else:
        print(f"VectorDB already exists in the persist directory {CONFIG.persist_directory}")
    return None

if __name__ == '__main__':
    upload_data_manually()