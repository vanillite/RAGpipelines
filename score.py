import json
from sentence_transformers import SentenceTransformer
import os

model = None

def init():
    global model
    try:
        model_dir = "/var/azureml-app/azureml-models/intfloat_multilingual_e5_small/1/model_dir"
        model = SentenceTransformer(model_dir)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        raise e

def run(raw_data):
    try:
        request_data = json.loads(raw_data)

        #Check if the input follows the expected format
        if "values" not in request_data:
            return json.dumps({"error": "Input data does not contain 'values' key."})

        input_items = request_data["values"]
        input_texts = []
        record_ids = []

        #Extract recordId and text from each item in the input
        for item in input_items:
            if "recordId" not in item or "data" not in item or "text" not in item["data"]:
                continue  # Skip invalid items
            record_ids.append(item["recordId"])
            input_texts.append(item["data"]["text"])

        #Use custom model to generate embeddings for the extracted input texts
        embeddings = model.encode(input_texts)

        #Build response object in the schema expected by Azure's web API skill
        response = {
            "values": [
                {
                    "recordId": record_id,
                    "data": {
                        "vector": embedding.tolist()
                    }
                }
                for record_id, embedding in zip(record_ids, embeddings)
            ]
        }

        #Return response directly as dictionary, as ML studio has a built-in step that converts dict int JSON
        return response

    except Exception as e:
        logging.exception(f"Error during processing: {str(e)}")
        return json.dumps({"error": str(e)})