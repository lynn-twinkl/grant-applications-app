import json
import sys

raw_data = sys.argv[1]

def load_data(json_path):
    """
    Load your custom JSON with 'additional_info' and 'label' fields.
    Returns a list of (text, {"entities": [(start, end, label), ...]}) tuples.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # If your JSON is a list of records
    # If it's a single record, wrap it in [data] or handle accordingly
    if not isinstance(data, list):
        data = [data]

    training_data = []

    for record in data:
        text = record["additional_info"]
        spans = []
        for annotation in record["label"]:
            # Each annotation can have multiple "labels", but typically there's just one
            label = annotation["labels"][0]
            start = annotation["start"]
            end = annotation["end"]
            spans.append((start, end, label))
        # Append in spaCy's format
        training_data.append((text, {"entities": spans}))

    return training_data

if __name__ == "__main__":
    # Example usage
    TRAIN_DATA = load_data(raw_data)
    
    print(TRAIN_DATA[:2])
