import re
import csv 

file_path = 'C:/Users/Ajay_XPS_9380/desktop/prompt_1.txt'
output_file= 'C:/Users/Ajay_XPS_9380/desktop/2output_prompt_1.csv'

with open(file_path, encoding="utf-8") as f:
    prompt1_string = f.read()



def extract_sections_and_codes(text):
    sections = re.findall(r"Prompt:(.*?)NONE\)\s*::(.*?)Response:(.*?)(?=\n\nPrompt:|$)", text, re.DOTALL)
    return [(section.strip(), code.strip()) for section, _, code in sections]

def write_to_csv(sections):
    with open(output_file, 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['section', 'code'])
        writer.writerows(sections)

if __name__ == "__main__":
    # Replace 'your_text_here' with the actual provided text
    sections_and_codes = extract_sections_and_codes(prompt1_string)
    write_to_csv(sections_and_codes)
    print("done")

