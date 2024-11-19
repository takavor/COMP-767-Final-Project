from tqdm import tqdm
import json
from openai import OpenAI

client = OpenAI()

# load data
with open('data/data.json') as file:
  data = json.load(file)

# function to generate similar sentences given context
def generate_similar_contexts(context, labels):
  
  prompt=f"Generate two sentences by paraphrasing the following sentence while maintaining its semantic meaning:\n'{context}'\nThe sentences you generate must also contain a [BLANK] token, which could be replaced by the following words which are labeled as stereotype, anti-stereotype, or unrelated: {labels}.\nSeparate the sentences by a newline."

  completion = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
      {"role": "user", "content": prompt}
    ]
  )
  
  response = completion.choices[0].message.content
  return response


for item in tqdm(data):
  response = generate_similar_contexts(item['context'], item['labels'])
  item['response'] = response

# save
with open('data/data_gpt4.json', 'w') as file:
  json.dump(data, file, indent=4)
  
  
  
