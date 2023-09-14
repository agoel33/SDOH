import openai
import pandas as pd
import re

# Set your OpenAI API key
openai.api_key = "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"

# File path and prompt template
file_path = 'C:/Users/Ajay_XPS_9380/desktop/discharge_with_social_final.csv'

# Read the CSV file and extract the first ten entries
df = pd.read_csv(file_path)


output_file_path = 'C:/Users/Ajay_XPS_9380/desktop/Z59.4_code_version.txt'
section_pattern = r'\b[A-Z][a-zA-Z]*(?: [A-Z][a-zA-Z]*)*:'
# Open the output file in write mode
with open(output_file_path, 'a', encoding='utf-8') as output_file:
    output_file.write("Z59.00 Specific Prompt\n")
    for i in range(8,10):
        output_file.write(f"\n Note {i}\n\n")
        print(i)
        note = df.loc[i]['text']
        section_headers = [(match.group(), match.start(), match.end()) for match in re.finditer(section_pattern, note)]
        # Extract the content between each section header
        sections = []
        # Find all occurrences of the section headers in the note along with their start and end positions
        section_headers = [(match.group(), match.start(), match.end()) for match in re.finditer(section_pattern, note)]

        # Function to check if a section contains only special characters (no letters or numbers)
        def contains_only_special_characters(section_content):
            return re.fullmatch(r'[^a-zA-Z0-9]*\s*', section_content) is not None

        # Extract the content between each section header and filter out empty and sections with no letters/numbers
        sections = []
        for j in range(8, len(section_headers)):  # Skip the first section header
            start_pos = section_headers[j][2]
            end_pos = section_headers[j + 1][1] if j < len(section_headers) - 1 else None
            section_content = note[start_pos:end_pos].strip()
            if section_content and not contains_only_special_characters(section_content):
                sections.append((section_headers[j][0], section_content))

        # Display the sections along with their headers
        for header, section in sections:
            curr_section_header = header
            curr_section = section
            prompt_template = f"""NOTE {i} Should the following note be annotated with the specific CDC Z Code Z59.4: Food insecurity? 
Please start your answer with YES if so, or NONE if not, providing direct evidence from the note if YES: 
                            {curr_section_header} {curr_section}"""
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages= [{ "role": "system", "content": "each of the following prompts is unique. start over." },
                        {"role": "user", "content": f"{prompt_template}"}],
                max_tokens=400,  # Adjust the response length as needed
                temperature=0.0,  # Set temperature to 0.0 for deterministic responses
            )
            output_file.write(f"Prompt: {prompt_template}\n")
            output_file.write(f"Response: NOTE {i} {response['choices'][0]['message']['content'].strip()}\n")
        output_file.write("\n\n")

print("Responses saved to Z59.4_code_version.txt")
