import pandas as pd

# Load the datasets
df = pd.read_json('homelessness_validation_dataset.json').reset_index()
df_2 = pd.read_json('homelessness_training_dataset.json').reset_index()

# Define the formatting function
def format_text(row):
    prompt_parts = row['Prompt'].split(':', maxsplit=1)
    prompt_before_semicolon = prompt_parts[0]
    prompt_after_semicolon = prompt_parts[1].strip() if len(prompt_parts) > 1 else ''
    return f"[INST]<<SYS>>\n {prompt_before_semicolon} \n</SYS>> \n{prompt_after_semicolon}\n [/INST] {row['Model Response']}"

# Apply the formatting function to create the new 'text' column
df['text'] = df.apply(format_text, axis=1)
df_2['text'] = df_2.apply(format_text, axis=1)  # Corrected this line

# Check for None values in the 'text' column and print the corresponding row indices
none_indices = df[df['text'].isna()].index
if len(none_indices) > 0:
    print("Rows with None values in 'text' column (validation dataset):", none_indices)

none_indices_2 = df_2[df_2['text'].isna()].index
if len(none_indices_2) > 0:
    print("Rows with None values in 'text' column (training dataset):", none_indices_2)

# Print some information about the datasets
print("Length of validation dataset:", len(df))
print("Length of training dataset:", len(df_2))

df.to_json('homelessness_validation_dataset_final_1.json')
df_2.to_json('homelessness_training_dataset_final_1.json')
