from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

def extract_needs(docs, max_concurrency: int = 10) -> List[List[str]]:

    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                """
                You are a data extraction assistant helping a grant review team evaluate funding applications from schools. Your task is to extract a structured list of specific items or services that the school is requesting funding for.

                ## Instructions:

                - Users will submit excerpts from grant application letters.
                - From each letter, extract only the tangible items or clearly defined services the school wants to use the grant for.
                - Output the extracted items as a **clean, comma-separated list**, with no additional explanation or formatting.
                - Do not include abstract goals or general program names (e.g., "Arts Award program" or "student development").
                - Focus on concrete nouns that represent resources or services the grant would directly fund (e.g., "paint", "laptops", "counseling sessions", "sports equipment").

                ## Example:

                **User Input:**
                _I work in an alternative provision supporting disadvantaged children with SEND. Many of our students face significant challenges in communication and emotional expression.  
                Art is a powerful tool that allows them to process feelings, build confidence, and develop essential life skills.  
                With your support, we would purchase paints, canvases, clay, and sketchbooks to run the Arts Award program.  
                This would give our students a creative voice and a sense of achievement. Additionally, we could bring in an art therapist weekly, providing vital emotional support.  
                Your funding would transform lives, giving these children hope and opportunity._
                
                **Your Output:**
                paints, canvases, clay, sketchbooks, art therapist
                """
                 ),
                ("human", "{input}")
            ]
        )

    chain = prompt | llm

    responses = chain.batch(docs, config={"max_concurrency": max_concurrency})
    
    result = []

    for response in responses:
        # Extract the content from AIMessage
        content = response.content
        
        items = [item.strip() for item in content.split(',')]
        
        result.append(items)
    
    return result
