import nltk
import json
import pandas as pd
import transformers
from transformers import pipeline
from nltk.tokenize import sent_tokenize


file_path = '../discharge_with_social_final.csv'
df = pd.read_csv(file_path)

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli",
                      device_map= "auto")

bart_list = []
for i in range(50):
  print(i)
  entry = df.loc[i]['text']
  sentences = sent_tokenize(entry)
  for sentence in sentences:
    sequence_to_classify = f"NOTE {i} " + sentence
    candidate_labels = ['relative needing care', 'no relative needing care', 'not specified']
    bart_list.append(classifier(sequence_to_classify, candidate_labels))

output_file_path = 'C:/Users/Ajay_XPS_9380/desktop/bart_relative_care.json'

with open(output_file_path, 'w') as fout:
  json.dump(bart_list, fout)

