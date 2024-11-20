import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import json
from tqdm import tqdm

# load llama 3 model
login(token="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")

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)
model.eval()

# load data
with open('data/intrasentence/intrasentence_contexts_original.json') as original_file:
  data_original = json.load(original_file)

with open('data/intrasentence/intrasentence_contexts_1.json') as file_1:
  data_1 = json.load(file_1)
  
with open('data/intrasentence/intrasentence_contexts_2.json') as file_2:
  data_2 = json.load(file_2)

# function to compute score of sentence with <word> instead of plank
def compute_score(model, tokenizer, context, word):
    
    text = context.replace("BLANK", word)
    
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        average_log_likelihood = -loss.item()

    return average_log_likelihood

##################################
# log probs for original sentences
scores_original = []
for item in tqdm(data_original):
    context_text = item['context']
    labels = item['labels']

    # Find the position of 'BLANK' in context_text
    blank_pos = context_text.find('BLANK')

    if blank_pos == -1:
        print(f"BLANK not found in context: {context_text}")
        continue

    candidate_scores = []

    for gold_label, candidate_word in labels.items():

        log_likelihood = compute_score(model, tokenizer, context_text, candidate_word)
        candidate_scores.append({
            'gold_label': gold_label,
            'candidate_word': candidate_word,
            'score': log_likelihood
        })

    # find the candidate with the highest log-likelihood
    best_candidate = max(candidate_scores, key=lambda x: x['score'])
    best_gold_label = best_candidate['gold_label']
    
    scores_original.append({
        "id": item['id'],
        'context': item['context'],
        'bias_type': item['bias_type'],
        'best_reply': best_candidate,
        'gold_label': best_gold_label
    })
    
with open('results/llama3_intrasentence_scores_original.json', 'w') as file:
    json.dump(scores_original, file, indent=4)
##################################
##################################
# log probs for first generated sentence
##################################
# log probs for original sentences
scores_1 = []
for item in tqdm(data_1):
    context_text = item['context']
    labels = item['labels']

    # Find the position of 'BLANK' in context_text
    blank_pos = context_text.find('BLANK')

    if blank_pos == -1:
        print(f"BLANK not found in context: {context_text}")
        continue

    candidate_scores = []

    for gold_label, candidate_word in labels.items():

        log_likelihood = compute_score(model, tokenizer, context_text, candidate_word)
        candidate_scores.append({
            'gold_label': gold_label,
            'candidate_word': candidate_word,
            'score': log_likelihood
        })

    # find the candidate with the highest log-likelihood
    best_candidate = max(candidate_scores, key=lambda x: x['score'])
    best_gold_label = best_candidate['gold_label']
    
    scores_1.append({
        "id": item['id'],
        'context': item['context'],
        'bias_type': item['bias_type'],
        'best_reply': best_candidate,
        'gold_label': best_gold_label
    })
    
with open('results/llama3_intrasentence_scores_1.json', 'w') as file:
    json.dump(scores_1, file, indent=4)
##################################
##################################
# log probs for second generated sentences
scores_2 = []
for item in tqdm(data_2):
    context_text = item['context']
    labels = item['labels']

    # Find the position of 'BLANK' in context_text
    blank_pos = context_text.find('BLANK')

    if blank_pos == -1:
        print(f"BLANK not found in context: {context_text}")
        continue

    candidate_scores = []

    for gold_label, candidate_word in labels.items():

        log_likelihood = compute_score(model, tokenizer, context_text, candidate_word)
        candidate_scores.append({
            'gold_label': gold_label,
            'candidate_word': candidate_word,
            'score': log_likelihood
        })

    # find the candidate with the highest log-likelihood
    best_candidate = max(candidate_scores, key=lambda x: x['score'])
    best_gold_label = best_candidate['gold_label']
    
    scores_2.append({
        "id": item['id'],
        'context': item['context'],
        'bias_type': item['bias_type'],
        'best_reply': best_candidate,
        'gold_label': best_gold_label
    })
    
with open('results/llama3_intrasentence_scores_2.json', 'w') as file:
    json.dump(scores_2, file, indent=4)
##################################