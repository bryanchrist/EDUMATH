import torch
import os
import pandas as pd
import numpy as np 
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# Load GPT-2
model_path = "openai-community/gpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load data
df = pd.read_csv("data/all_model_samples.csv")
stem = pd.read_csv("data/stem.csv")
asdiv = pd.read_csv("data/asdiv_corrected.csv")
# This function calculates the perplexity for question/answer pairs in a pandas dataframe in the format outputted by EDUMATH. Note that you will need a DF with a separate question
# and solution column for this code to run. 
def perplexity(df):
    # Create list to store perplexities
    ppls = []

    # Loop over all question/answer pairs in df, calculating ppl for each
    for i in range(0, len(df)):
        text = "Question: " + str(df.iloc[i]['question']) + "\n" + "Solution:\n" + str(df.iloc[i]['solution'])
        inputs = tokenizer(text, return_tensors = "pt").to(model.device)
        loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
        ppl = torch.exp(loss)
        ppl = ppl.cpu().detach().numpy()
        ppls.append(ppl)
    return ppls

# Example usage
ppl = perplexity(df)
df['ppl'] = ppl
df.to_csv("data/all_model_samples_with_ppl.csv", index = False)
ppl = perplexity(stem)
stem['ppl'] = ppl
stem.to_csv("data/stem_with_ppl.csv", index = False)
ppl = perplexity(asdiv)
asdiv['ppl'] = ppl
asdiv.to_csv("data/asdiv_with_ppl.csv", index = False)