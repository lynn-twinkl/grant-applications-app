import logging
from src.prompts import PROMPTS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_usage(docs, max_concurrency: int = 110) -> List[List[str]]:
    logging.info(f"Starting extract_usage function with {len(docs)} documents and max_concurrency={max_concurrency}")

    llm = ChatOpenAI(
        model='gpt-5-nano',
        temperature=1 # Only 1 supported by gpt 5
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPTS['usage_extraction_system']),
        ("human", "{input}")
        ])

    chain = prompt | llm

    responses = chain.batch(docs, config={"max_concurrency": max_concurrency})
    
    result = []

    for response in responses:
        # Extract the content from AIMessage
        content = response.content
        
        items = [item.strip() for item in content.split(',')]
        
        result.append(items)
    
    return result
