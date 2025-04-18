import spacy
import pandas as pd
import sys

csv_path = sys.argv[1]
custom_model_path = sys.argv[2]

df = pd.read_csv(csv_path)
texts = df['Additional Info'].to_list()

trained_nlp = spacy.load(custom_model_path)

for text in texts:
    doc = trained_nlp(text)
    print(f"TEXT: {text}")
    print()
    print("ENTITIES:", [(ent.text, ent.label_) for ent in doc.ents])
    print('-'*60) 
