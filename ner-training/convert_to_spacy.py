import spacy
from spacy.tokens import DocBin
from prepare_data import load_data
import sys

json_file_path = sys.argv[1]
outpath = sys.argv[2]

def create_spacy_binary(json_path, output_path, nlp):
    """
    Convert raw training examples into spaCy DocBin.
    """
    db = DocBin()

    # Load your custom data
    training_data = load_data(json_path)

    for text, ann in training_data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in ann["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                # If spaCy can't align the tokenization, skip or handle carefully
                print(f"Skipping misaligned entity: '{text[start:end]}'")
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)

    db.to_disk(output_path)
    print(f"Created spaCy binary file: {output_path}")


if __name__ == "__main__":
    nlp = spacy.blank("en")  # blank English pipeline
    create_spacy_binary(json_file_path, outpath, nlp)
