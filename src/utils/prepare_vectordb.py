from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
import os
from langchain.vectorstores.chroma import Chroma

class PrepareVectorDB:
    def __init__(
            self,
            data_directory:str,
            persist_directory:str,
            embedding_model_engine:str,
            chunk_size:str,
            chunk_overlap:str
    )-> None :
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
        self.data_directory = data_directory;
        self.persist_directory = persist_directory;
        self.embedding_model_engine = OpenAIEmbeddings()
    
    def __load_all_documents(self) -> list :

        """
        
        """ 
        doc_count = 0
        if(isinstance(self.data_directory, list)):
            docs = []
            print("Loading all the documents in the directory")
            for data_dir in self.data_directory:
                docs.extend(PyPDFLoader(data_dir).load())
                doc_count +=1
            print(f"Number of docs loaded : {doc_count}")
            print(f"Number of pages : {len(docs)} \n\n")
        else:
            print("Loading documents manually")
            doc_list = os.listdir(self.data_directory) 
            docs = []
            for doc in doc_list:
                docs.extend(PyPDFLoader(os.path.join(self.data_directory, doc)).load())
                doc_count+=1
            print(f"Number of documents loaded : {doc_count}")
            print(f"Number of pages : {len(docs)} \n\n")
        return docs
    
    def __chunk_documents(self, docs : list) -> list:
        """
        
        """
        print("Making chunks of documents")
        chunked_documents = self.text_splitter.split_documents(docs)
        print(f"Number of chunks created: {len(chunked_documents)} \n\n")
        return chunked_documents
    
    def prepare_and_save_vectorDB(self):
        """
        
        """
        docs = self.__load_all_documents()
        chunks = self.__chunk_documents(docs)
        print("Preparing vector database")

        vectorDB = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model_engine,
            persist_directory=self.persist_directory
        )
        print("VectorDB has been created")
        print(f"Number of documents in the database: {vectorDB._collection.count()} \n\n ")
        return vectorDB