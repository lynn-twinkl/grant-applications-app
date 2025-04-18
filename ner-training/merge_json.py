import json                                                                   
import sys

original_json_path = sys.argv[1]
additional_json_path = sys.argv[2]
                                                                              
with open(additional_json_path, 'r') as source_file:                                 
    source_data = json.load(source_file)                                      
                                                                              
# Load data from target.json                                                  
with open(original_json_path, 'r') as target_file:                                 
    target_data = json.load(target_file)                                      
                                                                              
# Ensure both source_data and target_data are lists                           
if isinstance(source_data, list) and isinstance(target_data, list):           
    # Append records from source_data to target_data                          
    target_data.extend(source_data)                                           
else:                                                                         
    print("The JSON data must be a list of records in both files.")           
                                                                              
# Write updated data back to target.json                                      
with open(original_json_path, 'w') as target_file:                                 
    json.dump(target_data, target_file, indent=4)                             
                                                                              
print("Records have been appended successfully.")         
