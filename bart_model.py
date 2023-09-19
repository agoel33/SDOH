import nltk
import json
import pandas as pd
import transformers
from transformers import pipeline
from nltk.tokenize import sent_tokenize


file_path = 'discharge_with_social_final.csv'
df = pd.read_csv(file_path)

does_not_support_gpu = ['MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',  'sileod/deberta-v3-base-tasksource-nli', "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
          "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "NDugar/v2xl-again-mnli", "NDugar/1epochv3"]
permission_error = [ "joeddav/xlm-roberta-large-xnli"]
token_too_long = ["DAMO-NLP-SG/zero-shot-classify-SSTuning-base"]

models_used = ["valhalla/distilbart-mnli-12-1", "valhalla/distilbart-mnli-12-9", "HiTZ/A2T_RoBERTa_SMFA_WikiEvents-arg_ACE-arg"]

possible_labels = [['relative needing care', 'no relative needing care', 'not specified'], ["employed", "not specified", "unemployed"], 
["inprisonment or other incarceration", "not specified", "no impisonment or other incarceration"]["homeless", "not specified", "not homeless"], 
["food insecure", "not specified", "not food insecure"], ["low income", "not specified", "not low income"], ["marital estrangement", "not specified", "no marital estrangement"]]

models = ["eleldar/theme-classification", "Narsil/bart-large-mnli-opti", "morit/english_xlm_xnli", "BSC-LT/sciroshot"]

for label in possible_labels:
  print(f"{label}")
  for mdl in models:
    print(f"{mdl}")
    classifier = pipeline("zero-shot-classification",
                          model= mdl,
                          device_map= "auto")
    model_classification = []
    for i in range(50):
      print(i)
      entry = df.loc[i]['text']
      sentences = sent_tokenize(entry)
      for sentence in sentences:
        sequence_to_classify = f"NOTE {i} " + sentence
        candidate_labels = label
        model_classification.append(classifier(sequence_to_classify, candidate_labels))
    
    parts = mdl.split("/")
    new_string = "".join(parts)
    output_file_path = f'{new_string}_{label[0]}.json'
    with open(output_file_path, 'w') as fout:
      json.dump(model_classification, fout)
    print(f"{mdl} is done")

