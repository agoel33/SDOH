import nltk
import json
import pandas as pd
import transformers
from transformers import pipeline
from nltk.tokenize import sent_tokenize


file_path = '../discharge_with_social_final.csv'
df = pd.read_csv(file_path)

models = ['MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli', 'sileod/deberta-v3-base-tasksource-nli', "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
          "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "joeddav/xlm-roberta-large-xnli "
          "optimum/distilbert-base-uncased-mnli"]
for mdl in model:
  print("mdl")
  classifier = pipeline("zero-shot-classification",
                        model= mdl,
                        device_map= "auto")
  model_classification = []
  for i in range(5):
    print(i)
    entry = df.loc[i]['text']
    sentences = sent_tokenize(entry)
    for sentence in sentences:
      sequence_to_classify = f"NOTE {i} " + sentence
      candidate_labels = ['relative needing care', 'no relative needing care', 'not specified']
      model_classification.append(classifier(sequence_to_classify, candidate_labels))

  output_file_path = f'../SDOH/{mdl}_relative_care.json'
  with open(output_file_path, 'w') as fout:
    json.dump(model_classification, fout)
  print(f"{mdl} is done")

