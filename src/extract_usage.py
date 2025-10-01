from src.prompts import PROMPTS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any
from dotenv import load_dotenv
from src.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

def extract_usage(docs, max_concurrency: int = 110) -> List[List[str]]:
    logger.info(f"Extracting usage items for {len(docs)} documents using AI (max_concurrency={max_concurrency})")

    try:

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

        logger.info(f"Successfully extracted usage items from applications")

    except Exception as e:
        logger.error(f"Failed to extract usage items: {str(e)}")
        raise
    
    try:
        result = []

        for response in responses:
            # Extract the content from AIMessage
            content = response.content
            items = [item.strip() for item in content.split(',')]
            result.append(items)
        
        return result

    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        raise
