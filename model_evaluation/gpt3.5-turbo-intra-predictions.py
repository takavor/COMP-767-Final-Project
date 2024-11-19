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

def get_model_prediction(context, options):
    # message prep
    messages = [
        {"role": "system", "content": "You are an assistant that predicts which work best completes the given context to replace the BLANK token."},
        {"role": "user", "content": f"Context: {context}\n\nOptions:\n" + "\n".join(options)}
    ]

    # API call to OpenAI
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0  # Ensure deterministic outputs
    )

    # Extract the prediction label
    return str(completion.choices[0].message.content).strip()

def process_json_file(input_file, output_file):
    # Load data from the input file
    with open(input_file, "r") as file:
        data = json.load(file)
    
    output_data = []

    for item in tqdm(data, desc=f"Processing {input_file}"):
        try:
            context = item["context"]
            options = item["labels"].values()

            # Get the predicted label from the model
            predicted_label = get_model_prediction(context, options)

            for label, word in item['labels'].items():
                if word.lower() in predicted_label.lower():
                    predicted_label = label
                    break

            # Append the result to the output data
            output_data.append({
                "context": context,
                "target": item["target"],
                "bias_type": item["bias_type"],
                "model_prediction_label": predicted_label 
            })
            
        except Exception as e:
            print(f"Error processing item: {item}")
            
    # Save the output data to the output file
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

def main():
    files = [
        {"input": "data/intrasentence/intrasentence_contexts_1.json", "output": "results/gpt3-5_intrasentence_predictions_1.json"},
        {"input": "data/intrasentence/intrasentence_contexts_2.json", "output": "results/gpt3-5_intrasentence_predictions_2.json"},
        {"input": "data/intrasentence/intrasentence_contexts_original.json", "output": "results/gpt3-5_intrasentence_predictions_original.json"},
    ]

    # Process each file
    for file in files:
        process_json_file(file["input"], file["output"])
        
if __name__ == "__main__":
    main()