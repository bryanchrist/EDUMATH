import os 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch 
import numpy as np 
import pandas as pd
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.3-70B-Instruct')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.3-70B-Instruct', device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True, load_in_4bit=True)
df = pd.read_csv('data/ASDIV_Clean.csv')

pipe = pipeline(
    "text-generation", 
    model=model, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    tokenizer = tokenizer, 
    max_new_tokens = 250,
    do_sample = False
)
annotations = []
binary_labels = []
for i in range(0, len(df)):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an experienced elementary school teacher. You are tasked with reading an educational standard(s) and its substandards and then assessing a word problem and its solution to determine whether the problem meets the standard(s) and substandards. When responding, say "Yes." or "No." to indicate whether the problem meets the specified standard(s). Please put detailed reasoning after "Yes." or "No." You only need to say "Yes." or "No." once to indicate whether the problem meets the standard(s) and substandards as a whole. Only say "Yes." if the problem exactly matches the operations and constraints mentioned in the standard(s) and substandards. For example, if the standard mentions multiplication within a certain number range, then the problem will only meet that standard if it requires multiplication within that range. 
<|start_header_id|>user<|end_header_id|>
Standard(s): 
{df.iloc[i]['standard_VA']}

Substandards:
{df.iloc[i]['substandard_VA']}

Word Problem:
{df.iloc[i]['question']}

Solution: 
{df.iloc[i]['solution']}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    text = pipe(prompt)
    text = text[0]['generated_text']
    text = text.split("<|start_header_id|>assistant<|end_header_id|>")[1]
    if "Yes." in text:
        binary_label = 1
    if "No." in text:
        binary_label = 0
    binary_labels.append(binary_label)
    annotations.append(text)
    temp_df = df[:i+1]
    temp_df['annotations'] = annotations
    temp_df['binary_labels'] = binary_labels
    temp_df.to_csv("data/ASDIV_Clean_Llama70B_Annotations.csv")
    