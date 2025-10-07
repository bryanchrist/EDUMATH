import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', help="Huggingface model to annotate data, entered as string", type = str, default = "google/gemma-3-27b-it")
parser.add_argument('--tokenizer', nargs='+', help="Huggingface tokenizer for specified model, entered as string", type = str, default = "google/gemma-3-27b-it")
parser.add_argument('--input_file', help="Input CSV file with samples to annotate", type = str)
parser.add_argument('--output_file', help="Output CSV file for annotated samples", type = str)
parser.add_argument('--do_sample', help="Whether do_sample is enabled for annotation model", action="store_true")
args = parser.parse_args()
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch 
import numpy as np 
import pandas as pd
import json
torch._dynamo.config.disable = True
df = pd.read_csv("data/annotations_copy.csv")
if "gemma" in args.model or "sft" in args.model:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    #model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation='eager')
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)

if "gemma" not in args.model and "sft" not in args.model:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)

if args.do_sample:
    pipe = pipeline(
        "text-generation", 
        model=model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        tokenizer = tokenizer, 
        max_new_tokens = 500, #Llama
        do_sample = True)
if not args.do_sample:
    pipe = pipeline(
        "text-generation", 
        model=model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        tokenizer = tokenizer, 
        max_new_tokens = 500, #Llama
        do_sample = False)

df = pd.read_csv(args.input_file)
if "math_topic" not in df.columns.tolist():
    standards = pd.read_csv("data/matched_standards_summarized.csv")
    standards = standards[['grade', 'standard', 'substandard', 'math_topic']]
    df = pd.merge(df, standards, how = "left", on = ['grade', 'standard', 'substandard'])
import ast

def unwrap_list(val):
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        try:
            parsed = ast.literal_eval(val)  # safely parse string to Python object
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]
        except Exception:
            return val  # fallback if parsing fails
    return val

df["math_topic"] = df["math_topic"].apply(unwrap_list)
if args.llm_annotate:
    labels = []
    annotations = []
    for i in range(0, len(df)):
        query = f"""Topic: {df.iloc[i]['topic']}
        
Question: {df.iloc[i]['question']}
    
Does the question effectively incorporate the specified topic?"""
        
        prompt = f"""<bos><start_of_turn>user
Your goal is to read a math word problem and determine whether it effectively includes a pre-specified topic. If the question does not effectively include the topic, write "No." followed by your reasoning. If the question does effectively include the topic, write "Yes." followed by your reasoning.
    
Now evaluate whether this problem includes the specified topic and remember to exactly answer "Yes." or "No." followed by your reasoning.
{query} <end_of_turn>
<start_of_turn>model"""
    
        inputs = tokenizer.encode(prompt, return_tensors="pt", padding = "longest", pad_to_multiple_of=8).to(model.device)
        outputs = model.generate(inputs, max_new_tokens = 500, do_sample = False)
        text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        text = text.split("<start_of_turn>model")[1]
        yes_index = text.find("Yes.")
        no_index = text.find("No.")
    
        if yes_index != -1 and (no_index == -1 or yes_index < no_index):
            label = 1
            labels.append(label)
        elif no_index != -1:
            label = 0
            labels.append(label)
            
        annotations.append(text)
        temp = df[:i+1]
        temp['model_labels'] = labels
        temp['model_reasoning']= annotations
        temp.to_csv(args.output_file, index = False)