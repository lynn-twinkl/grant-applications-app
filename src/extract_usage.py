from src.prompts import PROMPTS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

def extract_usage(docs, max_concurrency: int = 20) -> List[List[str]]:

    llm = ChatOpenAI(
        model='gpt-4.1-mini',
        temperature=0.2
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
