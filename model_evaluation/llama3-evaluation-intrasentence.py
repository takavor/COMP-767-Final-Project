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

# function to compute log likelihood for word
def compute_continuation_log_likelihood(model, tokenizer, context_prefix, continuation):
    
    # Concatenate context_prefix and continuation
    text = context_prefix + continuation
    # Tokenize the input
    inputs = tokenizer(text, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Get the length of the context_prefix
    context_inputs = tokenizer(context_prefix, return_tensors='pt').to(device)
    context_length = context_inputs['input_ids'].shape[1]

    # Compute the outputs
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Shift logits and labels for next-token prediction
    logits = logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    # Sum log probabilities for continuation tokens only
    continuation_log_probs = target_log_probs[:, context_length - 1:].sum().item()
    return continuation_log_probs

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

    context_prefix = context_text[:blank_pos]
    context_suffix = context_text[blank_pos + len('BLANK'):]

    candidate_scores = []

    for gold_label, candidate_word in labels.items():
        # Create the continuation by adding the candidate word and context_suffix
        continuation = candidate_word + context_suffix

        # Ensure proper spacing
        if not continuation.startswith(' '):
            continuation = ' ' + continuation

        log_likelihood = compute_continuation_log_likelihood(model, tokenizer, context_prefix, continuation)
        candidate_scores.append({
            'gold_label': gold_label,
            'candidate_word': candidate_word,
            'log_likelihood': log_likelihood
        })

    # find the candidate with the highest log-likelihood
    best_candidate = max(candidate_scores, key=lambda x: x['log_likelihood'])
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
scores_1 = []
for item in tqdm(data_1):
    context_text = item['context']
    labels = item['labels']

    # Find the position of 'BLANK' in context_text
    blank_pos = context_text.find('BLANK')

    if blank_pos == -1:
        print(f"BLANK not found in context: {context_text}")
        continue

    context_prefix = context_text[:blank_pos]
    context_suffix = context_text[blank_pos + len('BLANK'):]

    candidate_scores = []

    for gold_label, candidate_word in labels.items():
        # Create the continuation by adding the candidate word and context_suffix
        continuation = candidate_word + context_suffix

        # Ensure proper spacing
        if not continuation.startswith(' '):
            continuation = ' ' + continuation

        log_likelihood = compute_continuation_log_likelihood(model, tokenizer, context_prefix, continuation)
        candidate_scores.append({
            'gold_label': gold_label,
            'candidate_word': candidate_word,
            'log_likelihood': log_likelihood
        })

    # find the candidate with the highest log-likelihood
    best_candidate = max(candidate_scores, key=lambda x: x['log_likelihood'])
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
# log probs for second generated sentence
scores_2 = []
for item in tqdm(data_2):
    context_text = item['context']
    labels = item['labels']

    # Find the position of 'BLANK' in context_text
    blank_pos = context_text.find('BLANK')

    if blank_pos == -1:
        print(f"BLANK not found in context: {context_text}")
        continue

    context_prefix = context_text[:blank_pos]
    context_suffix = context_text[blank_pos + len('BLANK'):]

    candidate_scores = []

    for gold_label, candidate_word in labels.items():
        # Create the continuation by adding the candidate word and context_suffix
        continuation = candidate_word + context_suffix

        # Ensure proper spacing
        if not continuation.startswith(' '):
            continuation = ' ' + continuation

        log_likelihood = compute_continuation_log_likelihood(model, tokenizer, context_prefix, continuation)
        candidate_scores.append({
            'gold_label': gold_label,
            'candidate_word': candidate_word,
            'log_likelihood': log_likelihood
        })

    # find the candidate with the highest log-likelihood
    best_candidate = max(candidate_scores, key=lambda x: x['log_likelihood'])
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