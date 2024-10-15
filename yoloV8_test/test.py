import torch
import json
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# print(torch.__version__)
with open('rekognition_results.json', 'r', encoding='utf-8') as f:
    rekognition_results = json.load(f)


print(rekognition_results[0]['Similarity'])
