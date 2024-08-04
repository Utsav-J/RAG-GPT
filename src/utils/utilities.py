import tiktoken

def count_tokens(text:str, model:str)-> int:
    """
    Returns the number of tokens in the text
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))   