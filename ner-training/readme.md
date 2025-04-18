# Appropriate Usage for NER Training

## Cleaning and Debugging Training Data

We first need to debug our raw labeled data from Label Studio. Sometimes, labeled data has trailing whitespaces or punctuation, which Spacy _really_ doesn't like. So we need to remove it.

`python3 debug_labeled_data.py raw_labeded_data_path text_key_to_debug outdir`

This will create a new debugged json file in the specified directory. **Use this file for the next step.**

## Preparing Data For Training

Now, we need to convert this raw labeled data into Spacy's binary format. Before doing so however, we must make sure to split the data into training and dev sets for testing.

1. `python3 split_data.py debugged_json_path`

This will create `train.json` and `dev.json` files in the current working directory.

2. Move these file into the trianing_data dir:  `mv *.json training_data/`

3. Convert both sets into Spacy's binary format:

`python3 convert_to_spacy.py training_data/train.json training_data/train.spacy`
`python3 convert_to_spacy.py training_data/dev.json training_data/dev.spacy`

## Training

To start training the data from the CLI, we simply run the following command:

`
python -m spacy train transformer.cfg \ 
--paths.train training_data/train.spacy \
--paths.dev training_data/dev.spacy \
--gpu-id 0 \
--output ./roberta_model 
`
