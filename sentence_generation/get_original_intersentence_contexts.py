# open dev file
import json
with open("data/dev.json", "r") as file:
    data = json.load(file)
    
# convert array of intersentences to dict, with key being ID
intersentences = data["data"]["intersentence"]
d = {}    

for item in intersentences:
    d[item["id"]] = item

# write dict to json file
with open("data/original_intersentence_contexts.json", "w") as file:
    json.dump(d, file, indent=4)