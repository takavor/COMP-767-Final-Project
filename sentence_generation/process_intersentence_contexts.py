import json

# this file contains the generated intersentence contexts
with open("data/generated_intersentence_contexts.json", "r") as file:
    data = json.load(file)

# this file contains the possible replies to the sentences, that the models will choose from
with open("data/original_intersentence_contexts.json", "r") as file:
    original_data = json.load(file)

processed_contexts = []

for context in data:
    
    
    response = context["response"]
    # extract two sentences
    sentences = response.split("\n")
    if (len(sentences) != 2):
        print("Skipped id", context["id"])
        continue
    
    # remove whitespaces
    sentences = [s.strip() for s in sentences]
    
    # add to context
    context["generated_contexts"] = sentences
    
    # remove unnecessary keys
    context.pop("response", None)
    
    # extract possible replies
    original_item = original_data[context["id"]]
    original_replies = original_item["sentences"]
    
    replies = []
    for original_reply in original_replies:
        replies.append({
            "sentence": original_reply["sentence"],
            "gold_label": original_reply["gold_label"]
        })
    
    context["replies"] = replies
    processed_contexts.append(context)
    
# write to file
with open("data/processed_intersentence_contexts.json", "w") as file:
    json.dump(processed_contexts, file, indent=4)
    