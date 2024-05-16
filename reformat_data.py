import pandas as pd

df = pd.read_json('homelessness_validation_dataset.json').reset_index()
df_2 = pd.read_json('homelessness_training_dataset.json').reset_index()

def format_text(row):
    prompt_parts = row['Prompt'].split(':', maxsplit=1)
    prompt_before_semicolon = prompt_parts[0]
    prompt_after_semicolon = prompt_parts[1].strip() if len(prompt_parts) > 1 else ''
    return f"[INST]<<SYS>>\n {prompt_before_semicolon} \n</SYS>> \n{prompt_after_semicolon}\n [/INST] {row['Model Response']}"

# Apply the formatting function to create the new 'text' column
df['text'] = df.apply(format_text, axis=1)
df_2['text'] = df.apply(format_text, axis=1)

print(df_2['text'][15])
print(len(df))
print(len(df_2))
df.to_json('homelessness_validation_final.json')
df_2.to_json('homelessness_training_final.json')