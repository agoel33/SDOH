import nltk
import json
import pandas as pd
import transformers
from transformers import pipeline
from nltk.tokenize import sent_tokenize


file_path = 'discharge_with_social_final.csv'
df = pd.read_csv(file_path)


"BAD MODELS"
does_not_support_gpu = ['MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',  'sileod/deberta-v3-base-tasksource-nli', "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
          "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "NDugar/v2xl-again-mnli", "NDugar/1epochv3", 
          "KheireddineDaouadi/ZeroAraElectra", "cointegrated/rubert-base-cased-nli-threeway", "cross-encoder/nli-deberta-v3-base", "seduerr/paiintent", 
          "typeform/distilbert-base-uncased-mnli", "MoritzLaurer/ernie-m-large-mnli-xnli", "mjwong/e5-large-mnli", "claritylab/zero-shot-vanilla-binary-bert", 
          "claritylab/zero-shot-vanilla-bi-encoder", "mjwong/contriever-msmarco-mnli", "mjwong/bge-large-en-mnli-anli"]
permission_error = [ "joeddav/xlm-roberta-large-xnli"]
other_problems = ["osanseviero/test_zero", "arnov/name-gender", "Xenova/distilbert-base-uncased-mnli", "Atharva192003/zero-shot-classfier", "KoboldAI/fairseq-dense-1.3B"]
needs_tensor_flow_weights = ["typeform/roberta-large-mnli"]
gigantic_model = ["AntoineBlanot/flan-t5-xxl-classif-3way"]
token_too_long = ["DAMO-NLP-SG/zero-shot-classify-SSTuning-base", "morit/english_xlm_xnli", "BSC-LT/sciroshot"]

"USED MODELS"
models_used = ["valhalla/distilbart-mnli-12-1", "valhalla/distilbart-mnli-12-9", "HiTZ/A2T_RoBERTa_SMFA_WikiEvents-arg_ACE-arg", "eleldar/theme-classification", "Narsil/bart-large-mnli-opti",
"ClaudeYang/awesome_fb_model", "cross-encoder/nli-MiniLM2-L6-H768", "cross-encoder/nli-distilroberta-base", "cross-encoder/nli-roberta-base", 
"joeddav/bart-large-mnli-yahoo-answers", "navteca/bart-large-mnli", "oigele/Fb_improved_zeroshot", "HiTZ/A2T_RoBERTa_SMFA_ACE-arg", "HiTZ/A2T_RoBERTa_SMFA_TACRED-re",
"MoritzLaurer/xlm-v-base-mnli-xnli", "AyoubChLin/bart_large_mnli_finetune_cnn_news", "AyoubChLin/BART-mnli_cnn_256", "sjrhuschlee/flan-t5-base-mnli",  "HWERI/pythia-1.4b-deduped-sharegpt", "aisquared/chopt-1_3b", "breadlicker45/dough-instruct-base-001", 
"MBZUAI/LaMini-Neo-1.3B", "MayaPH/FinOPT-Franklin", "pszemraj/pythia-31m-KI_v1-2048-scratch", "BreadAi/PM_modelV2", "concedo/OPT-19M-ChatSalad"]

"LABELS"
possible_labels = [["food insecure", "not specified", "not food insecure"], ["low income", "not specified", "not low income"], ["marital estrangement", "not specified", "no marital estrangement"], 
["homeless", "not specified", "not homeless"], ['relative needing care', 'no relative needing care', 'not specified'], ["employed", "not specified", "unemployed"]]

temp_no_labels = [
["inprisonment or other incarceration", "not specified", "no inprisonment or other incarceration"]]


probelems = ["TheBloke/Project-Baize-v2-7B-GPTQ"]
"CURRENT MODELS"
models = ["HiTZ/A2T_RoBERTa_SMFA_TACRED-re"]

"TEST MODELS"



for label in temp_no_labels:
  print(f"{label}")
  for mdl in models:
    print(f"{mdl}")
    classifier = pipeline("zero-shot-classification", model= mdl,
                          device_map= "auto")
    model_classification = []
    for i in range(50):
      print(i)
      entry = df.loc[i]['text']
      sentences = sent_tokenize(entry)
      for sentence in sentences:
        sequence_to_classify = f"NOTE {i} " + sentence
        candidate_label = label
        model_classification.append(classifier(sequence_to_classify, candidate_label))
    parts = mdl.split("/")
    new_string = "".join(parts)
    new_string = new_string.replace("-", "_")
    
    output_file_path = f'{new_string}_{label[0]}.json'
    with open(output_file_path, 'w') as fout:
      json.dump(model_classification, fout)
    print(f"{mdl} is done")

