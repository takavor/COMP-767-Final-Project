from tqdm import tqdm
import json
from openai import OpenAI
import os

client = OpenAI()

# load intersentence contexts
with open('data/dev.json') as file:
  data = json.load(file)

intersentence_list = data["data"]["intersentence"]

# extract contexts
contexts = [{"id": item["id"], "context": item["context"], "target": item["target"]} for item in intersentence_list]

print(f"Number of sentences: {len(contexts)}")

# get previously generated contexts, if they exist
processed_contexts_file = "data/generated_intersentence_contexts.json"
if os.path.exists(processed_contexts_file):
  with open(processed_contexts_file, 'r') as file:
    processed_contexts = json.load(file)
  
  processed_ids = set([context['id'] for context in processed_contexts])
else:
  processed_contexts = []
  processed_ids = set()

# function to generate similar sentence given context
def generate_similar_context(context, target):
  
  prompt=f"""
            Your task is to generate two sentences that are similar in meaning to a given sentence. The sentences you generate MUST include
            the target word, which will be provided to you.
            ---
            Sentence: {context}
            ---
            Target word: {target}
            ---
            Separate your sentences by a newline.
            Make sure that the sentences you generate contains the target word. Never change the target word with any other word.
            """
            
  completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {"role": "user", "content": prompt}
    ]
  )
  
  response = completion.choices[0].message.content
  return response

# generate a new context per context
save_interval = 50
for idx, context in enumerate(tqdm(contexts)):
  
  # skip if already generated
  if context['id'] in processed_ids:
    continue
  
  response = generate_similar_context(context['context'], context['target'])
  context['response'] = response
  
  processed_contexts.append(context)
  processed_ids.add(context['id'])
  
  # save
  if idx % save_interval == 0:
    with open(processed_contexts_file, 'w') as file:
      json.dump(processed_contexts, file, indent=4)
    
    print(f"Saved checkpoint")    
  
  
