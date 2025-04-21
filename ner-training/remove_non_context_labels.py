import json                                                                   
import sys

file_to_filter = sys.argv[1]
tag_to_keep = sys.argv[2]
outpath = sys.argv[3]

with open(file_to_filter, 'r') as input_file:
    dataset = json.load(input_file)
                                                                              
def filter_context_labels(dataset):                                           
    for item in dataset:                                                      
        item['label'] = [l for l in item['label'] if tag_to_keep in l['labels']]
    return dataset                                                            
                                                                              
filtered_data = filter_context_labels(dataset)

with open(outpath, 'w') as output_file: 
    json.dump(filtered_data, output_file, indent=2)
