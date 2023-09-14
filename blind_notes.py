import pandas as pd

file_path = 'C:/Users/Ajay_XPS_9380/desktop/discharge_with_social_final.csv'

df = pd.read_csv(file_path)

output_file_path = 'C:/Users/Ajay_XPS_9380/desktop/blind_notes_2.txt'

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for i in range(100, 500):
        print(i)
        note = df.loc[i]['text']
        output_file.write(f"NOTE {i}\n {note}\n\n\n")
print("File blind_notes_2 correctly uploaded")