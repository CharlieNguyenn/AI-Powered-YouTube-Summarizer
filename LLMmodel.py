import os
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_community.vectorstores import FAISS

def setup_credentials():    
    model_id = "ibm/granite-3-2-8b-instruct"
   
    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com",
        api_key=os.getenv("IBM_API_KEY")
    )
       
    client = APIClient(credentials)
   
    project_id = os.getenv("IBM_PROJECT_ID")
   
    return model_id, credentials, client, project_id
 
def define_parameters():    
    return {
        # Set the decoding method to GREEDY for generating text
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
       
        # Specify the maximum number of new tokens to generate
        GenParams.MAX_NEW_TOKENS: 900,
    }
 
# Create and return an instance of the WatsonxLLM with the specified configuration 
def initialize_watsonx_llm(model_id, credentials, project_id, parameters):    
    return WatsonxLLM(
        model_id=model_id,
        url=credentials.url,
        apikey=credentials.api_key,
        project_id=project_id,
        params=parameters
    )
 
# Create and return an instance of WatsonxEmbeddings with the specified configuration 
def setup_embedding_model(credentials, project_id):    
    return WatsonxEmbeddings(
        model_id='ibm/slate-30m-english-rtrvr-v2',
        url=credentials.url,
        apikey=credentials.api_key,
        project_id=project_id
    )

# Use the FAISS library to create an index from the provided text chunks 
def create_faiss_index(chunks, embedding_model):
    """
    Create a FAISS index from text chunks using the specified embedding model.
   
    :param chunks: List of text chunks
    :param embedding_model: The embedding model to use
    :return: FAISS index
    """
    return FAISS.from_texts(chunks, embedding_model)