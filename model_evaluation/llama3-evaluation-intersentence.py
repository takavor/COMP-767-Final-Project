import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm

from huggingface_hub import login
login(token="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")

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)
model.eval()

# extract data
with open("data/intersentence_contexts_original.json", "r") as file:
    data_original = json.load(file)
    
with open("data/intersentence_contexts_1.json", "r") as file:
    data_1 = json.load(file)
    
with open("data/intersentence_contexts_2.json", "r") as file:
    data_2 = json.load(file)

# function to compute log likelihood of a reply
def compute_reply_log_likelihood(model, tokenizer, context, reply_sentence):
    
    # concatenate context and reply
    text = context + ' ' + reply_sentence
    
    # tokenize input
    inputs = tokenizer(text, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # get length of context
    context_inputs = tokenizer(context, return_tensors='pt')
    context_length = context_inputs['input_ids'].shape[1]
    
    # compute logits (outputs)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # shift logits and labels for next token prediction
    logits = logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    
    # get log probs
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    
    # sum log probs for reply tokens only
    reply_log_probs = target_log_probs[:, context_length-1:].sum().item()
    return reply_log_probs

######################################
# compute preds for original sentences
scores_original = []
for item in tqdm(data_original):
    
    context = item['context']
    bias_type = item['bias_type']
    replies = item['replies']
    
    reply_scores = []
    for reply in replies:
        sentence = reply['sentence']
        log_likelihood = compute_reply_log_likelihood(model, tokenizer, context, sentence)
        reply_scores.append({'reply': reply, 'log_likelihood': log_likelihood})
    
    # get reply with the highest log-likelihood
    best_reply = max(reply_scores, key=lambda x: x['log_likelihood'])
    gold_label = best_reply['reply']['gold_label']
    
    scores_original.append({
        "id": item['id'],
        'context': context,
        'bias_type': bias_type,
        'best_reply': best_reply,
        'gold_label': gold_label
    })
    
# write scores to file
with open('data/llama3_intersentence_scores_original.json', 'w') as file:
    json.dump(scores_original, file, indent=4)
######################################
######################################
# compute preds for first generated sentences
scores_1 = []
for item in tqdm(data_1):
    
    context = item['context']
    bias_type = item['bias_type']
    replies = item['replies']
    
    reply_scores = []
    for reply in replies:
        sentence = reply['sentence']
        log_likelihood = compute_reply_log_likelihood(model, tokenizer, context, sentence)
        reply_scores.append({'reply': reply, 'log_likelihood': log_likelihood})
    
    # get reply with the highest log-likelihood
    best_reply = max(reply_scores, key=lambda x: x['log_likelihood'])
    gold_label = best_reply['reply']['gold_label']
    
    scores_1.append({
        "id": item['id'],
        'context': context,
        'bias_type': bias_type,
        'best_reply': best_reply,
        'gold_label': gold_label
    })
    
# write scores to file
with open('data/llama3_intersentence_scores_1.json', 'w') as file:
    json.dump(scores_1, file, indent=4)
######################################
######################################
# compute preds for first generated sentences
scores_2 = []
for item in tqdm(data_2):
    
    context = item['context']
    bias_type = item['bias_type']
    replies = item['replies']
    
    reply_scores = []
    for reply in replies:
        sentence = reply['sentence']
        log_likelihood = compute_reply_log_likelihood(model, tokenizer, context, sentence)
        reply_scores.append({'reply': reply, 'log_likelihood': log_likelihood})
    
    # get reply with the highest log-likelihood
    best_reply = max(reply_scores, key=lambda x: x['log_likelihood'])
    gold_label = best_reply['reply']['gold_label']
    
    scores_2.append({
        "id": item['id'],
        'context': context,
        'bias_type': bias_type,
        'best_reply': best_reply,
        'gold_label': gold_label
    })
    
# write scores to file
with open('data/llama3_intersentence_scores_2.json', 'w') as file:
    json.dump(scores_2, file, indent=4)
######################################