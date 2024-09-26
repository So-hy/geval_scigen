import json

# Paths for the uploaded files
test_cl_path = './data/scigen_result/test-CL.json'
gold_descriptions_path = './data/scigen_result/GOLD_descriptions.txt'
large_predictions_path = './data/scigen_result/BART-large-few-shot_predictions.txt'

# Read the JSON data from test-CL.json
with open(test_cl_path, 'r',  encoding="utf-8") as test_cl_file:
    test_cl_data = json.load(test_cl_file)

# Read the lines from GOLD_descriptions.txt
with open(gold_descriptions_path, 'r', encoding="utf-8") as gold_file:
    gold_lines = gold_file.readlines()

# Read the lines from T5-large-large_predictions.txt
with open(large_predictions_path, 'r', encoding="utf-8") as predictions_file:
    prediction_lines = predictions_file.readlines()

# Create a new output list with the specified format
output_data_reformatted = []

# Iterate over the test-CL data and match with descriptions and predictions
for idx, (key, value) in enumerate(test_cl_data.items()):
    # Match each test item with a line from GOLD_descriptions and predictions
    gold_description = gold_lines[idx].strip() if idx < len(gold_lines) else ""
    generated_description = prediction_lines[idx].strip() if idx < len(prediction_lines) else ""

    # Reformatted entry
    reformatted_entry = {
        "table_id": key,
        "table_info": {
            "table_caption": value.get("table_caption"),
            "table_column_names": value.get("table_column_names", []),
            "table_content_values": value.get("table_content_values", [])
        },
        "gold_description": gold_description,
        "generated_description": generated_description
    }

    # Add to reformatted output list
    output_data_reformatted.append(reformatted_entry)

# Write the reformatted output data to a new JSON file
output_file_path_reformatted = './data/bart-few.json'

# Write the reformatted data to a new JSON file
with open(output_file_path_reformatted, 'w', encoding="utf-8") as output_file_reformatted:
    json.dump(output_data_reformatted, output_file_reformatted, ensure_ascii=False, indent=4)


print(f"Reformatted JSON saved at: {output_file_path_reformatted}")