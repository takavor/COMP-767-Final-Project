import json
from dotenv import load_dotenv
import os
from tqdm import tqdm
from openai import OpenAI

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = OpenAI()

def get_model_prediction(context, replies):
    # Prepare the list of sentences from the replies
    sentences = [reply["sentence"] for reply in replies]
    
    # Prepare messages for the structured GPT-3.5-turbo response
    messages = [
        {"role": "system", "content": "You are a helpful assistant that predicts which reply is most appropriate based on the given context."},
        {"role": "user", "content": f"Context: {context}\n\nReplies:\n" + "\n".join(sentences) +
         "\n\nRespond with the reply you predict fits the context best."}
    ]

    # Make the API call
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0  # Use deterministic outputs
    )
    
    # Extract the predicted sentence
    return str(completion.choices[0].message["content"])

def process_json_file(input_file, output_file):
    # Load data
    with open(input_file, "r") as file:
        data = json.load(file)

    output_data = []

    for item in tqdm(data, desc=f"Processing {input_file}"):
        context = item["context"]
        replies = item["replies"]

        # Get model prediction
        model_prediction = get_model_prediction(context, replies)

        # Find the corresponding gold label for the predicted reply
        predicted_gold_label = None
        for reply_item in replies:
            if reply_item["sentence"] == model_prediction:
                predicted_gold_label = reply_item["gold_label"]
                break

        # Construct output dictionary
        output_data.append({
            "context": context,
            "target": item["target"],
            "model_prediction": model_prediction,
            "model_prediction_label": predicted_gold_label
        })

    # Save results to output file
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

def main():
    # Input and output file names
    files = [
        {"input": "data/data_gpt4_intersentence_1.json", "output": "data/gpt3-5_intersentence_predictions_1.json"},
        {"input": "data/data_gpt4_intersentence_2.json", "output": "data/gpt3-5_intersentence_predictions_2.json"},
        {"input": "data/data_gpt4_intersentence_2.json", "output": "data/gpt3-5_intersentence_predictions_2.json"},
    ]

    # Process each file
    for file in files:
        process_json_file(file["input"], file["output"])

if __name__ == "__main__":
    main()