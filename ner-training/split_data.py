import json
import random
import sys

json_file_path = sys.argv[1]

# Load your full dataset (make sure it's a list of records)
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Shuffle and then split the data 80/20
random.shuffle(data)
split_index = int(len(data) * 0.8)
train_data = data[:split_index]
dev_data = data[split_index:]

# Save the train and dev JSON files
with open("train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2)

with open("dev.json", "w", encoding="utf-8") as f:
    json.dump(dev_data, f, indent=2)

print(f"Train examples: {len(train_data)}, Dev examples: {len(dev_data)}")
