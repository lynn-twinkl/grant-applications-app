import streamlit as st
import re
import string
import torch
import spacy

from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
import contractions
from tqdm import tqdm


from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech
import openai
import numpy as np

import os
from dotenv import load_dotenv
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../",".env")))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#################################
# OpenAI Topic Representation
#################################
def create_openai_model():
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            prompt = """
            I have a topic that contains the following documents:
            [DOCUMENTS]

            The topic is described by the following keywords: [KEYWORDS]

            Based on the information above, extract a short yet descriptive topic label of at most 4 words. The labels should be interpretable enough to stakeholders that don't have access to the raw data. Make sure it is in the following format:

            topic: <topic label>
            """
            openai_model = OpenAI(client, model="gpt-4o-mini", exponential_backoff=True, chat=True, prompt=prompt)
            return openai_model

#############################################
# Convert OpenAI Representation to CustomName
#############################################

def ai_labeles_to_custom_name(model):
    chatgpt_topic_labels = {topic: " | ".join(list(zip(*values))[0]) for topic, values in model.topic_aspects_["OpenAI"].items()}
    chatgpt_topic_labels[-1] = "Outlier Topic"
    model.set_topic_labels(chatgpt_topic_labels)

"""
-----------------------------------
Lemmatization & Stopword Removal
-----------------------------------

"""
def topicModeling_preprocessing(df, spacy_model="en_core_web_lg"):

    base_stopwords = set(stopwords.words('english')) 

    custom_stopwords = {
        'material', 'materials', 'resources', 'resource', 'activity',
        'activities', 'sheet', 'sheets', 'worksheet', 'worksheets',
        'teacher', 'teachers', 'teach', 'high school', 'highschool',
        'middle school', 'grade', 'grades', 'hs', 'level', 'age', 'ages',
        'older', 'older kid', 'kid', 'student', "1st", "2nd", "3rd", "4th", '5th', '6th',
        '7th', '8th', '9th'
        }

    stopword_set = base_stopwords.union(custom_stopwords)

    stopword_pattern = r'\b(?:' + '|'.join(re.escape(word) for word in stopword_set) + r')\b'

    nlp = spacy.load(spacy_model)

    def clean_lemmatize_text(text):
        if not isinstance(text, str):
            return None
        
        text = contractions.fix(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(stopword_pattern, '', text)

        doc = nlp(text)
        tokens = [token.lemma_ for token in doc]
        
        clean_text = " ".join(tokens).strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)

        return clean_text if clean_text else None


    df['processedForModeling'] = df['preprocessedBasic'].apply(clean_lemmatize_text)

    # Drop rows where cleaned text is empty or None
    df = df.dropna(subset=['processedForModeling'])

    return df

"""
--------------------------
 Load Transformer Model
--------------------------
"""

def load_embedding_model(model_name):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    return SentenceTransformer(model_name, device=device)


"""
-------------------------
Batch Embedding Creation
-------------------------
"""

def encode_content_documents(embedding_model, content_documents, batch_size=20):
    embeddings_batches = []
    total_batches = range(0, len(content_documents), batch_size)

    with tqdm(total=len(total_batches), desc="Encoding Batches") as pbar:
        for i in total_batches:
            batch_docs = content_documents[i:i + batch_size]
            batch_embeddings = embedding_model.encode(batch_docs, convert_to_numpy=True, show_progress_bar=False)
            embeddings_batches.append(batch_embeddings)
            pbar.update(1)

    return np.vstack(embeddings_batches)

"""
-----------------------------
Topic Modeling with BERTopic
-----------------------------
"""

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stopwords = list(stopwords.words('english')) + [
        'activities',
        'activity',
        'class',
        'classroom',
        'material',
        'materials',
        'membership',
        'memberships',
        'pupil',
        'pupils',
        'resource',
        'resources',
        'sheet',
        'sheets',
        'student',
        'students',
        'subscription',
        'subscriptions',
        'subscribe',
        'subscribed',
        'recommend',
        'recommendation',
        'teach',
        'teacher',
        'teachers',
        'tutor',
        'tutors',
        'twinkl',
        'twinkls',
        'twinkle',
        'worksheet',
        'worksheets',
    ]

######### --------------- BERTOPIC ----------------- #############
def bertopic_model(docs, embeddings, _embedding_model, _umap_model, _hdbscan_model):
    
    main_representation_model = KeyBERTInspired()
    aspect_representation_model1 = MaximalMarginalRelevance(diversity=.3)

    # OpenAI Representation Model
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = """
    I have a topic that contains the following documents:
    [DOCUMENTS]

    The topic is described by the following keywords: [KEYWORDS]

    Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make sure it is in the following format:

    topic: <topic label>
    """
    openai_model = OpenAI(client, model="gpt-4o-mini", exponential_backoff=True, chat=True, prompt=prompt)

    representation_model = {
        "Main": main_representation_model,
        "Secondary Representation": aspect_representation_model1,
    }

    vectorizer_model = CountVectorizer(min_df=2, max_df=0.60, stop_words=stopwords)

    seed_topic_list = [
            ["autism", "special needs", "special education needs", "special education", "adhd", "autistic", "dyslexia", "dyslexic", "sen"],
            ]

    topic_model = BERTopic(
        verbose=True,
        embedding_model=_embedding_model,
        umap_model=_umap_model,
        hdbscan_model = _hdbscan_model,
        vectorizer_model=vectorizer_model,
        #seed_topic_list = seed_topic_list,
        representation_model=representation_model,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings)
    return topic_model, topics, probs

##################################
# TOPIC MERGING
##################################

def merge_specific_topics(topic_model, sentences, 
                          cancellation_keywords=["cancel", "cancellation", "cancel", "canceled"],
                          thanks_keywords=["thank", "thanks", "thank you", "thankyou", "ty", "thx"],
                          expensive_keywords=["can't afford", "price", "expensive", "cost"]):


    topic_info = topic_model.get_topic_info()
    
    # Identify cancellation-related topics by checking if any cancellation keyword appears in the topic name.
    cancellation_regex = '|'.join(cancellation_keywords)
    cancellation_topics = topic_info[
        topic_info['Name'].str.contains(cancellation_regex, case=False, na=False)
    ]['Topic'].tolist()
    
    # Identify thank-you-related topics similarly.
    thanks_regex = '|'.join(thanks_keywords)
    thanks_topics = topic_info[
        topic_info['Name'].str.contains(thanks_regex, case=False, na=False)
    ]['Topic'].tolist()
    
    # Identify expensive-related topics.
    expensive_regex = '|'.join(expensive_keywords)
    expensive_topics = topic_info[
        topic_info['Name'].str.contains(expensive_regex, case=False, na=False)
    ]['Topic'].tolist()
    
    # Exclude the outlier topic (-1) if it appears.
    cancellation_topics = [t for t in cancellation_topics if t != -1]
    thanks_topics = [t for t in thanks_topics if t != -1]
    expensive_topics = [t for t in expensive_topics if t != -1]
    
    # Create a list of topics to merge
    topics_to_merge = []
    
    if len(cancellation_topics) > 1:
        print(f"Merging cancellation topics: {cancellation_topics}")
        topics_to_merge.append(cancellation_topics)
    
    if len(thanks_topics) > 1:
        print(f"Merging thank-you topics: {thanks_topics}")
        topics_to_merge.append(thanks_topics)
        
    if len(expensive_topics) > 1:
        print(f"Merging expensive topics: {expensive_topics}")
        topics_to_merge.append(expensive_topics)
    
    # Call merge_topics
    if topics_to_merge:
        topic_model.merge_topics(sentences, topics_to_merge)
    
    return topic_model


##################################
# Topic to Dataframe Mapping
#################################

def update_df_with_topics(df, mapping, sentence_topics, topic_label_map):
    topics_by_row = {}
    for i, row_idx in enumerate(mapping):
        topic = sentence_topics[i]
        topics_by_row.setdefault(row_idx, set()).add(topic)
    
    updated_df = df.copy()
    
    def map_topics(row_idx):
        topic_ids = topics_by_row.get(row_idx, set())
        topic_names = [topic_label_map.get(t, str(t)) for t in topic_ids if t != -1]
        return ", ".join(sorted(topic_names))
    
    updated_df['Topics'] = updated_df.index.map(map_topics)
    return updated_df

