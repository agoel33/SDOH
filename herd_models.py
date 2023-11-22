import sys
# Add your folder path to sys.path
folder_path = '../Vornoi/QA/'
sys.path.append(folder_path)
from qa_utils import *
import nltk
import json
import pandas as pd
import transformers
from transformers import pipeline
from nltk.tokenize import sent_tokenize
saved_net = BertRegressor.load_from_checkpoint("/home/user/Vornoi/QA/vornoi/uh138i3k/checkpoints/epoch=14-step=5625.ckpt")
def run_router(input):
    #Router
    #saved net is a bert model. tokenize input and run it through the model
    input_ids = tokenizer.encode(input, return_tensors='pt').to(saved_net.device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(input_ids.device)
    output = saved_net(input_ids, attention_mask)
    output = output[:, 1:]
    print(output)
    best_model = model_names[output.argmax()]
    return best_model
tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-medium', use_fast = False)
model_names = [
    #XXS (1b order)
    # 'microsoft/phi-1_5',
    # 'Fredithefish/Guanaco-3B-Uncensored-v2',
    # 'EleutherAI/pythia-1b',
    # 'PY007/TinyLlama-1.1B-step-50K-105b',
    # 'cerebras/btlm-3b-8k-base',
    #XS (5b order)
    # 'TheBloke/Llama-2-7B-Chat-GGML', #some random error
    'TheBloke/Llama-2-7b-Chat-GPTQ', #1s per sentence
    # 'TheBloke/Airoboros-L2-7B-2.2-GPTQ', #some random error, ignoring for now
    'HyperbeeAI/Tulpar-7b-v0',
    # 'Open-Orca/Mistral-7B-OpenOrca',
    # 'mistralai/Mistral-7B-Instruct-v0.1',
    # 'mistralai/Mistral-7B-v0.1',
    # 'circulus/Llama-2-7b-orca-v1',
    # 'tiiuae/falcon-7b-instruct', #takes too long
    # 'meta-llama/Llama-2-7b-hf',
    # "stabilityai/StableBeluga-7B",
    # 'Lajonbot/tableBeluga-7B-instruct-pl-lora_unload',
    # 'THUDM/chatglm2-6b',
    'lmsys/vicuna-7b-v1.5',
    # 'lmsys/vicuna-7b-v1.3',
    # 'lmsys/vicuna-7b-v1.1',
    # 'TheBloke/Zarablend-L2-7B-GPTQ',
    #Small (10b order)
    'TheBloke/Spicyboros-13B-2.2-GPTQ',
    # 'TheBloke/openchat_v3.2_super-GPTQ', #also slow
    'TheBloke/Airoboros-L2-13B-2.2-GPTQ',
    # 'TheBloke/Pygmalion-2-13B-GPTQ', #Takes 7s per sentence
    # 'PygmalionAI/mythalion-13b',
    # 'lmsys/vicuna-13b-v1.5',
    # 'lmsys/vicuna-13b-v1.3',
    # 'lmsys/vicuna-13b-v1.1',
    # 'meta-llama/Llama-2-13b-hf',
    # 'AIDC-ai-business/Luban-13B',
    # 'uukuguy/speechless-llama2-luban-orca-platypus-13b',
    # 'yeontaek/llama-2-13B-ensemble-v5',
    # 'TFLai/OpenOrca-Platypus2-13B-QLoRA-0.80-epoch',
    # 'garage-bAInd/Stable-Platypus2-13B',
    # 'TheBloke/COTHuginn-4.5-19B-GPTQ', # 30 seconds per iteration
    'TheBloke/Unholy-v1-10l-13B-GPTQ', #1s per iteration
    'TheBloke/Nous-Hermes-13B-Code-GPTQ', #2s per iteration
    #Medium (30b order)
    # 'garage-bAInd/GPlatty-30B',
    # 'Writer/palmyra-20b-chat',
    # 'upstage/llama-30b-instruct-2048',
    # 'lmsys/vicuna-33b-v1.3',
    # 'tiiuae/falcon-40b',
    # 'garage-bAInd/SuperPlatty-30B',
    # 'CalderaAI/30B-Lazarus',
    'TheBloke/30B-Epsilon-GPTQ',
    # 'TheBloke/Airoboros-33B-2.1-GPTQ', #some random error
    #Large (70b order)
    # 'meta-llama/Llama-2-70b-chat-hf',
    # 'NousResearch/Nous-Hermes-Llama2-70b',
    # 'garage-bAInd/Platypus2-70B-instruct',
    # 'fangloveskari/Platypus_QLoRA_LLaMA_70b',
    # 'upstage/SOLAR-0-70b-16bit',
    # 'chargoddard/MelangeA-70b',
    'TheBloke/Airoboros-65B-GPT4-m2.0-GPTQ',
    'TheBloke/Llama-2-70B-Ensemble-v5-GPTQ', #3.5 seconds per iteration
    'TheBloke/Uni-TianYan-70B-GPTQ', #3s per iteration
    # # 'TheBloke/Synthia-70B-v1.2-GPTQ', #3s per example
    'TheBloke/ORCA_LLaMA_70B_QLoRA-GPTQ', #3s per example
    #XXL (150b order)
    # 'TheBloke/Falcon-180B-Chat-GPTQ', # 60s per sample
                    ]

prompt_1 = '''Below this prompt is a patient note. Does the note contain any evidence of homelessness? If it does contain evidence of homelessness return ABACRACADABRA 
along with direct evidence from the note, otherwise, return NONE with a justification of why there is no evidence of homelessness:

'''
test= '''
In the dimly lit corners of society, where shadows elongate and whispers of despair linger, there exists a stark reality that goes unnoticed by many—a reality that unfolds in the life of someone who is homeless. Picture a person, once firmly rooted in the comforting soil of stability, now adrift in the tumultuous sea of uncertainty. Each day, they navigate the harsh winds of life without the shelter of a permanent abode. The city streets, once bustling with purpose, now serve as both refuge and battleground for this individual.

Amidst the towering structures of concrete and steel, this homeless soul seeks solace beneath the city's indifferent skyline. A tattered sleeping bag becomes their makeshift fortress, shielding them from the cold grip of the night. Hunger pangs echo through silent alleyways as the city sleeps, and the search for sustenance becomes a relentless quest. The distant hum of traffic, once background noise, now symbolizes the perpetual motion that seems to have left them behind.

Yet, within the fragile shell of homelessness, resilience persists. There's a silent strength in the way this individual adapts to their ever-changing environment. Each possession, no matter how meager, becomes a cherished relic—a token of survival. A discarded cardboard box transforms into a humble abode, and the flickering glow of a streetlamp becomes a beacon of hope in the enveloping darkness.

The journey of the homeless is not just a physical one; it is a profound exploration of the human spirit. Faces weathered by hardship tell stories of shattered dreams and unforeseen circumstances. The stigma that often shadows the homeless fails to capture the complexity of their narratives—the missed opportunities, the fractured relationships, and the societal structures that let them slip through the cracks.

In the struggle for visibility, there exists an indomitable will to be seen, acknowledged, and understood. Despite the adversity, there is a shared humanity that transcends the labels imposed by circumstance. Every step taken on the unforgiving pavement is a testament to the endurance of the human spirit—a spirit that yearns for compassion, empathy, and the chance to rewrite the chapters of a life left unguarded.'''

def truncate_prompt(text, prompt, tokenizer, num_tokens):
    prompt_tokens = len(tokenizer.encode(prompt))
    length = num_tokens - prompt_tokens
    broken_text = []
    if len(tokenizer.encode(text)) <= length:
        broken_text.append(text)
        return broken_text
    
    i = 0
    tokenized_text= tokenizer.encode(text)
    while len(tokenized_text[i*length: -1]) > length:
        broken_text.append(tokenizer.decode(tokenized_text[i*length: (i+1)*length]))
        i += 1
    
    broken_text.append(tokenizer.decode(tokenized_text[i*length: -1]))

    broken_text[0] = broken_text[0][5:-1]
    return broken_text

file_path = 'discharge_with_social_final.csv'
df = pd.read_csv(file_path)

sequences = []
for i in range(50):
  print(i)
  entry = df.loc[i]['text']
  sentences = sent_tokenize(entry)
  for sentence in sentences:
    truncated_sentences = truncate_prompt(sentence, prompt_1, tokenizer, 508)
    for truncated_text in truncated_sentences:
      sequence_to_classify = f"NOTE {i} " + prompt_1 + truncated_text
      sequences.append(sequence_to_classify)

sequences_without_prompt = []
for i in range(50):
  print(i)
  entry = df.loc[i]['text']
  sentences = sent_tokenize(entry)
  for sentence in sentences:
    truncated_sentences = truncate_prompt(sentence, prompt_1, tokenizer, 508)
    for truncated_text in truncated_sentences:
      sequence_to_classify = f"NOTE {i}" + " " + truncated_text
      sequences_without_prompt.append(sequence_to_classify)


classifier = pipeline("zero-shot-classification", model = 'TheBloke/Llama-2-7b-Chat-GPTQ', device_map= "auto")
candidate_label = ["homeless", "not specified", "not homeless"]
output = classifier(sequences_without_prompt[:], candidate_label)
order = []
sequence_specific = []
for i in range(len(output)):
    sequence_specific.append(output[:][i]['sequence'])
    order.append(output[:][i]['labels'])
data = {'Sequence': sequence_specific, str(model_names[0]): order}
big_df = pd.DataFrame(data)
print(big_df)


for models in model_names[1:]:
    classifier = pipeline("zero-shot-classification", model = models, device_map= "auto")
    candidate_label = ["homeless", "not specified", "not homeless"]
    output = classifier(sequences_without_prompt[:], candidate_label)
    print("done")
    order = []
    sequence_specific = []
    for i in range(len(output)):
        sequence_specific.append(output[:][i]['sequence'])
        order.append(output[:][i]['labels'])
    data = {'Sequence': sequence_specific, str(models): order}
    new_df = pd.DataFrame(data)
    big_df = big_df.merge(new_df)
    big_df = big_df.drop_duplicates(subset = 'Sequence')
    print("dataframe is done")
    print(big_df)

big_df.to_csv('herd_models.csv', index= False)
big_df.to_json('herd_models.json', index = False)

