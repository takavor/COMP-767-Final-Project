import json
from dotenv import load_dotenv
import os
from tqdm import tqdm
from openai import OpenAI

# Load environment variables
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = OpenAI()

def get_model_prediction(context, replies):
    # Prepare the list of sentences from the replies
    sentences = [reply["sentence"] for reply in replies]
    
    # Prepare messages for the structured GPT-3.5-turbo response
    messages = [
        {"role": "system", "content": "You are an assistant that predicts which reply is most appropriate based on the given context."},
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
    return str(completion.choices[0].message.content)

def process_json_file(input_file, output_file):
    # Load data
    with open(input_file, "r") as file:
        data = json.load(file)

    output_data = []

    for item in tqdm(data, desc=f"Processing {input_file}"):
        try:
            context = item["context"]
            replies = item["replies"]

            # Get model prediction
            model_prediction = get_model_prediction(context, replies)
            
            for reply in replies:
                if reply["sentence"].lower() in model_prediction.lower():
                    predicted_label = reply["gold_label"]
                    break
                
            # Construct output dictionary
            output_data.append({
                "context": context,
                "target": item["target"],
                "bias_type": item["bias_type"],
                "model_prediction_label": predicted_label
            })
            
        except Exception as e:
            print(f"Error processing item: {item}")

    # Save results to output file
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

def main():
    # Input and output file names
    files = [
       # {"input": "data/intersentence/intersentence_contexts_1.json", "output": "results/gpt3-5_intersentence_predictions_1.json"},
        {"input": "data/intersentence/intersentence_contexts_2.json", "output": "results/gpt3-5_intersentence_predictions_2.json"},
        {"input": "data/intersentence/intersentence_contexts_original.json", "output": "results/gpt3-5_intersentence_predictions_original.json"},
    ]

    # Process each file
    for file in files:
        process_json_file(file["input"], file["output"])

if __name__ == "__main__":
    main()