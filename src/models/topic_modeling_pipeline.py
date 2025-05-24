import openai
import numpy as np
import streamlit as st
import re
import string
import torch
import spacy
from spacy.cli import download
import importlib.util
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, OpenAI


import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


#############################################
# Convert OpenAI Representation to CustomName
#############################################

###################################
# HELPER FUNCTIONS
###################################

## ---------- LOAD SPACY MODEL ---------

def load_spacy_model(model_name="en_core_web_md"):
    """
    This model is used for sentence tokenization
    as well as stopword generation
    """
    if importlib.util.find_spec(model_name) is None:
        print(f"Model '{model_name}' not found. Downloading now...")
        download(model_name)

    return spacy.load(model_name)

## -------- SENTENCE TOKENIZER -------

def spacy_sent_tokenize(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

## -------- LOAD TRANSFORMER MODEL -------

def load_embedding_model(model_name):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    return SentenceTransformer(model_name, device=device)


## ------------ GENERATE EMBEDDINGS ------------

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

## ------- BUILDING STOPWORDS LIST ------

nlp = load_spacy_model()
stopwords = list(nlp.Defaults.stop_words)


custom_stopwords = [
    'thank you', 'thankyou', 'thanks', 'thank'
    'children', 'child', 'students',
    'twinkl',
    'funding'

]

stopwords.extend(custom_stopwords)


## --------- INSTANTIATE OPENAI MODEL ---------

def create_openai_model():
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            prompt = """
            This topic contains the following documents:
            [DOCUMENTS]

            The topic is described by the following keywords: [KEYWORDS]

            Based on the information above, extract a short yet descriptive topic label.
            """
            
            openai_model = OpenAI(
                    client,
                    model="gpt-4.1-nano",
                    exponential_backoff=True,
                    chat=True, prompt=prompt,
                    system_prompt= """ **Task**: As a topic modeling expert, your responsibility is to generate concise yet comprehensive topic labels from rows in a BertTopic `topic_info` dataframe. These topics have been derived from grant application forms submitted by schools, tutors, or other institutions participating in Twinkl giveaways.\n\n**Objective**: Your goal is to create labels for the extracted topics that accurately and clearly describe each topic within the specified context. These labels should be easily interpretable by the members of the Community Collections team.\n\n**Instructions**: \n\n1. **Understand the Context**: The topics relate to grant applications and are relevant to educational institutions participating in Twinkl giveaways.\n\n2. **Generate Labels**:\n- Create labels that are short yet capture the essence of each topic.\n- Ensure that the labels are contextually appropriate and provide clarity.\n- Focus on making the labels easily understandable for the Community Collections team. \n\n3. **Considerations**:\n- Each label should succinctly convey the main idea of the topic.\n- Avoid overly technical language unless necessary for precision.\n- Ensure the labels align with the overall educational and grant-related context."""
            )
            return openai_model

## ---------- AI LABELS TO TOPIC NAME ----------

def ai_labels_to_custom_name(model):
    chatgpt_topic_labels = {topic: " | ".join(list(zip(*values))[0]) for topic, values in model.topic_aspects_["OpenAI"].items()}
    chatgpt_topic_labels[-1] = "Outlier Topic"
    model.set_topic_labels(chatgpt_topic_labels)


#############################
# BERTOPIC MODELING
#############################

def bertopic_model(docs, embeddings, _embedding_model, _umap_model, _hdbscan_model):
    
    main_representation_model = KeyBERTInspired()

    openai_model = create_openai_model()

    representation_model = {
        "Main": main_representation_model,
        "OpenAI": openai_model
    }


    vectorizer_model = CountVectorizer(stop_words=stopwords, ngram_range=(1,2))

    topic_model = BERTopic(
        verbose=True,
        umap_model=_umap_model,
        representation_model=representation_model,
        vectorizer_model=vectorizer_model,
        hdbscan_model=_hdbscan_model,
        embedding_model=_embedding_model,
        nr_topics='auto'
    )
    
    topics, probs = topic_model.fit_transform(docs, embeddings)

    return topic_model, topics, probs




##################################
# TOPIC TO DATAFRAME MAPPING
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

