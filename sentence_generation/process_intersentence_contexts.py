import json

# this file contains the generated intersentence contexts
with open("data/generated_intersentence_contexts.json", "r") as file:
    data = json.load(file)

# this file contains the possible replies to the sentences, that the models will choose from
with open("data/original_intersentence_contexts.json", "r") as file:
    original_data = json.load(file)

processed_contexts = []

intersentence_contexts_original = []
intersentence_contexts_1 = []
intersentence_contexts_2 = []

for context in data:
    
    response = context["response"]
    
    
    
    # extract two sentences
    sentences = response.split("\n")
    if (len(sentences) != 2):
        print("Skipped id", context["id"])
        continue
    
    # remove whitespaces
    sentences = [s.strip() for s in sentences]
    
    # remove unnecessary keys
    context.pop("response", None)
    
    # extract possible replies
    original_item = original_data[context["id"]]
    original_replies = original_item["sentences"]
    
    # add bias type as well
    context["bias_type"] = original_item["bias_type"]
    
    replies = []
    for original_reply in original_replies:
        replies.append({
            "sentence": original_reply["sentence"],
            "gold_label": original_reply["gold_label"]
        })
    
    context["replies"] = replies
    
    # add contexts
    context_1 = context.copy()
    context_2 = context.copy()
    
    context_1['context'] = sentences[0]
    context_2['context'] = sentences[1]
    
    intersentence_contexts_original.append(context)
    intersentence_contexts_1.append(context_1)
    intersentence_contexts_2.append(context_2)
    
# write to files
with open("data/intersentence_contexts_original.json", "w") as file:
    json.dump(intersentence_contexts_original, file, indent=4)
    
with open("data/intersentence_contexts_1.json", "w") as file:
    json.dump(intersentence_contexts_1, file, indent=4)
    
with open("data/intersentence_contexts_2.json", "w") as file:
    json.dump(intersentence_contexts_2, file, indent=4)
    