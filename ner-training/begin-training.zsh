#!/bin/zsh

source ner_venv/bin/activate

train_spacy_file=$1
dev_spacy_file=$2
model_outdir=$3

python3 -m spacy train transformer.cfg \
--paths.train "$train_spacy_file" \
--paths.dev "$dev_spacy_file"  \
--gpu-id 0 \
--output "$model_outdir"

