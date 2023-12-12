import nltk
import json
import pandas as pd
import transformers
from transformers import pipeline
from nltk.tokenize import sent_tokenize

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
    'TheBloke/ORCA_LLaMA_70B_QLoRA-GPTQ', #3s per exampledd
    #XXL (150b order)
    # 'TheBloke/Falcon-180B-Chat-GPTQ', # 60s per sample
                    ]

for models in model_names:
    print(models)
    classifier = pipeline("zero-shot-classification", model = models, device_map= "auto")