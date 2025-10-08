from transformers import AutoTokenizer 
import pandas as pd
import torch
import numpy as np

stem = pd.read_csv('data/stem.csv') #Load desired dataset
model_path = "meta-llama/Llama-2-70b-hf"   # Specify the path to the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# This function calculates the length for questions in a pandas dataframe based on the Llama-2 tokenizer. Note you need to specify the column name for the question text. Can also be used to calculate length of solutions by providing the solution column name instead. 
def check_length(df, varname):
    # Create list to store lengths
    question_lengths = []

    # Loop over all questions, calculating the length for each
    for i in range(0, len(df)):
        output = df.iloc[i][varname]
        try: 
            inputs = tokenizer.encode(output, return_tensors="pt")
        except:
            pass
        length = inputs.shape[1]
        question_lengths.append(length)
    return question_lengths

# Example usage 
stem_len = check_length(stem, 'question')
stem['length'] = stem_len #Turn length into a column if desired
#Print results
print(f'Average overall question length in tokens for STEM: {np.mean(stem_len)} Standard Deviation: {np.std(stem_len)}')