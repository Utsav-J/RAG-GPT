from langchain_text_splitters import RecursiveCharacterTextSplitter

class PrepareVectorDB:
    def __init__(
            self,
            data_directory:str,
            persist_directory:str,
            embedding_model_engine:str,
            chunk_size:str,
            chunk_overlap:str
    )-> None:
        """
        Initialize the PrepareVectorDB instance.

        Parameters:
            data_directory (str or List[str]): The directory or list of directories containing the documents.
            persist_directory (str): The directory to save the VectorDB.
            embedding_model_engine (str): The engine for OpenAI embeddings.
            chunk_size (int): The size of the chunks for document processing.
            chunk_overlap (int): The overlap between chunks.

        """
        self.embedding_model_engine = embedding_model_engine
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators= ['\n','\n\n'," ",""]
        )