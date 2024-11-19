from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm

def evaluate_sentence_likelihood(sentence):
    
    # Tokenize the input sentence and convert to tensor
    inputs = tokenizer(sentence, return_tensors="pt")
    
    # Move input to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Get model's output for sentence
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    # Extract the total loss
    loss = outputs.loss
    # Convert loss to likelihood (lower loss = higher likelihood)
    likelihood = torch.exp(-loss)
    
    return likelihood.item()

def get_key(val, dict):
    for key, value in dict.items():
        if val == value:
            return key

load_dotenv()

torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

token = os.getenv('HUGGINGFACE_TOKEN')

device = "cpu"

model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, token=token).to(device)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token)

# load data
with open('data/data_gpt4_1.json') as file_1:
  data_1 = json.load(file_1)
  
with open('data/data_gpt4_2.json') as file_2:
  data_2 = json.load(file_2)
  
with open('data/data_gpt4.json') as original_file:
  original_data = json.load(original_file)

# loop through data
i = 0
item_data_list = []
for item in tqdm(data_2):
    
    # get candidate words    
    original_item = original_data[i]
    words = [original_item['labels']['anti-stereotype'], original_item['labels']['unrelated'], original_item['labels']['stereotype']]
    base_sentence = item['context']
    
    item_data = {}
    item_data['base_sentence'] = base_sentence
    
    likelihoods = {}
    gold_labels = {}
    
    print(f'i = {i}')
    
    for word in words:
        
        # replace BLANK by word
        sentence = base_sentence.replace('BLANK', word)
        print(sentence)
        # get sentence likelihood
        likelihoods[word] = evaluate_sentence_likelihood(sentence)
        
        # get gold labels
        gold_labels[word] = get_key(word, original_item['labels'])
        

    # get word with the highest likelihood
    most_likely_word = max(likelihoods, key=likelihoods.get)
    likelihoods['most_likely_word'] = most_likely_word
    
    # get predicted label of most likely word
    predicted_label = get_key(most_likely_word, original_item['labels'])
    likelihoods['predicted_label'] = predicted_label
    
    item_data['likelihoods'] = likelihoods
    item_data['gold_labels'] = gold_labels

    print("Likelihoods:", likelihoods)
    print("Most likely word:", most_likely_word)
    item_data_list.append(item_data)
    
    i += 1
    print('------')
    
with open('data/llama_preliminary_2.json', 'w') as llama_preliminary:
    json.dump(item_data_list, llama_preliminary, indent=4)