
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
import torch
from peft import PeftModel

from datasets import load_dataset

from peft import LoraConfig, get_peft_model

import evaluate

#train the model on the dataset using a huggingface trainer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import TextDataset
from transformers import AutoTokenizer

import pandas as pd
import numpy as np
import sklearn

base_model = "teknium/OpenHermes-2-Mistral-7B"
tokenizer_base = 'teknium/OpenHermes-2-Mistral-7B'
device_map = "auto"
load_in_4bit = True
load_in_8bit = False

df_train = pd.read_json('homelessnes_training_dataset.json')

num_train_samples = int(2e5)
num_val_samples = int(5e5)

def group_texts_mask_non_f1(examples, block_size = 128):
    
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    return result
#%%
quant_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

lora_r = int(2**5)
lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=2*lora_r,
            target_modules=["k_proj", "v_proj", 'q_proj', 'o_proj'],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["ln_f"],
        )

tokenizer = AutoTokenizer.from_pretrained(tokenizer_base)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=base_model,
    torch_dtype=torch.float16,
    device_map=device_map,
    quantization_config=quant_config,
    trust_remote_code=True,
)

model = get_peft_model(model, lora_config)

#%%
train_dataset = load_dataset('json', 
                                data_files='../data_spreadsheets/adapter_responses_train.json',
                                # data_files='../data_spreadsheets/ADAM_train.json',
                                # data_files='../data_spreadsheets/question_and_model_name_train.json',
                                split='train',
                                )

val_dataset = load_dataset('json',
                            data_files='../data_spreadsheets/adapter_responses_val.json',
                            # data_files='../data_spreadsheets/ADAM_val.json',
                            # data_files='../data_spreadsheets/question_and_model_name_val.json',
                            split='train',
                            )

def tokenize_function(example):
    
    tokenized_output = tokenizer(example["text"], truncation=True, max_length=512, padding=False)
    labels = [[-100] * find_last_index(tokenized_text, 28793) + tokenized_text[find_last_index(tokenized_text, 28793):] for tokenized_text in tokenized_output['input_ids']]
    # labels = [[-100] * find_second_last_index(tokenized_text, 28740) + tokenized_text[find_second_last_index(tokenized_text, 28740):] for tokenized_text in result['input_ids']]

    return {
        "input_ids": tokenized_output["input_ids"],
        "attention_mask": tokenized_output["attention_mask"],
        "labels" : labels
    }

# train_dataset = train_dataset.select(list(range(min(num_train_samples, len(train_dataset))))).map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"]).map(group_texts_mask_non_f1, batched=True, num_proc=20)
# val_dataset = val_dataset.select(list(range(min(num_val_samples, len(val_dataset))))).map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"]).map(group_texts_mask_non_f1, batched=True, num_proc=20)

train_dataset = train_dataset.select(list(range(min(num_train_samples, len(train_dataset))))).map(tokenize_function, batched=True, num_proc=20, remove_columns=["text", 'question_id', 'model_ix', 'label', 'dataset_ix']).map(group_texts_mask_non_f1, batched=True, num_proc=20)
val_dataset = val_dataset.select(list(range(min(num_val_samples, len(val_dataset))))).map(tokenize_function, batched=True, num_proc=20, remove_columns=["text", 'question_id', 'model_ix', 'label', 'dataset_ix']).map(group_texts_mask_non_f1, batched=True, num_proc=20)
# #%%

#%%
training_args = TrainingArguments(
    # output_dir="./r_64_question_and_model_name",
    # output_dir="./r_64",
    output_dir="./adapter_router",
    overwrite_output_dir=True,
    #
    num_train_epochs=50,
    #
    do_eval=True,
    #
    save_strategy="steps",
    #
    evaluation_strategy="steps",
    eval_steps=100,
    #
    per_device_train_batch_size=32,
    per_device_eval_batch_size = 32,
    auto_find_batch_size = True,
    #
    gradient_accumulation_steps=2,
    #
    save_steps=100,
    save_total_limit=2,
    #
    logging_steps = 100,
    #
    # prediction_loss_only=True,
    #
    fp16=True,
    #
    push_to_hub=True,
    #
    load_best_model_at_end = True,
    #
    # resume_from_checkpoint = '/home/user/Vornoi/QA/training_routers/r_64/checkpoint-69000',
    #
    label_names = ["labels"],
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#%%
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

#%%
trainer.train(
    # resume_from_checkpoint=True
    )

eval_metrics = trainer.evaluate()
print(eval_metrics)
# %%

'teknium/OpenHermes-2-Mistral-7B'