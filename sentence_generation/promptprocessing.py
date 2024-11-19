import json

# function to split the data into 2 copies, each copy containing a sentence generated
def split_data(data):
    
    split_data_1 = []
    split_data_2 = []
    
    for item in data:
        item_1 = item.copy()
        item_2 = item.copy()
        
        responses = item['response'].split('\n')
        # remove blanks after split
        responses = list(filter(None, responses))
        if len(responses) > 2:
            del responses[1] # middle element is sometimes a few extra spaces
                                
        item_1['response'] = responses[0]
        item_2['response'] = responses[1]
        
        split_data_1.append(item_1)
        split_data_2.append(item_2)
        
    return split_data_1, split_data_2

# function to process sentences (remove square brackets, newlines, etc.)
def process_sentences(data):
    
    modified = []
    
    for item in data:
        modified_item = item.copy()
        modified_item['response'] = modified_item['response'].strip()
        modified.append(modified_item)
        
    return modified

# function to format the data in the original format
def format_data(original_data, new_data):

    modified = []

    for original_item, new_item in zip(original_data, new_data):
        
        modified_item = new_item.copy()
        
        modified_item['target'] = original_item['target']
        modified_item['bias_type'] = original_item['bias_type']
        modified_item['context'] = new_item['response']
        
        sentences = []
        # loop through sentences in original
        for sentence_item in original_item['sentences']:
            gold_label = sentence_item['gold_label']
            
            sentence_item_to_append = sentence_item.copy()
            sentence_item_to_append['sentence'] = modified_item['context']
            
            corresponding_label_word = new_item['labels'][gold_label]
            sentence_item_to_append['sentence'] = sentence_item_to_append['sentence'].replace('BLANK', corresponding_label_word)
        
            sentences.append(sentence_item_to_append)
            
        modified_item['sentences'] = sentences
        
        # remove unneeded keys
        del modified_item['response']
        del modified_item['labels']
    
        modified.append(modified_item)
    
    return modified

# load data
with open('data/data_gpt4.json') as file:
  data = json.load(file)
  
with open('data/dev_copy.json') as original_file:
    original_data = json.load(original_file)
    
intrasentence_data = original_data['data']['intrasentence']
  
# run functions on gpt4 data
data_1, data_2 = split_data(data)

# process data
data_1, data_2 = process_sentences(data_1), process_sentences(data_2)

data_1, data_2 = format_data(intrasentence_data, data_1), format_data(intrasentence_data, data_2)

# write to files
with open('data/data_gpt4_1.json', 'w') as file_1:
    json.dump(data_1, file_1, indent=4)
    
with open('data/data_gpt4_2.json', 'w') as file_2:
    json.dump(data_2, file_2, indent=4)