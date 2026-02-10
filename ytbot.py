import gradio as gr
import re 
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes 
from ibm_watsonx_ai import APIClient, Credentials  
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods 
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes  
from langchain_community.vectorstores import FAISS  
from langchain.chains import LLMChain  
from langchain.prompts import PromptTemplate  

# Get the video ID from a YouTube URL
def get_video_id(url):        
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

# Get the transcript of a YouTube video
def get_transcript(url):
    video_id = get_video_id(url)
    ytt_api = YouTubeTranscriptApi()
    transcripts = ytt_api.list(video_id)
    transcript = ""
    for t in transcripts:        
        if t.language_code == 'en':
            if t.is_generated:
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                transcript = t.fetch()
                break 
    
    return transcript if transcript else None