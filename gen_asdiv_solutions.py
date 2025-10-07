import os 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch 
import numpy as np 
import pandas as pd
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.3-70B-Instruct')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.3-70B-Instruct', device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True, load_in_4bit=True)
df = pd.read_excel('data/ASDIV_Standards.xlsx')

pipe = pipeline(
    "text-generation", 
    model=model, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    tokenizer = tokenizer, 
    max_new_tokens = 1500,
    do_sample = True, 
    temperature = .7
)

prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an experienced elementary school teacher. You are tasked with developing step-by-step solutions to math word problems for your students. The solutions should outline all the necessary steps, show complete work, and be written in a way a grade school student would understand. Make sure you separate your solution by writing "Solution:\n" and then the solution and end your solution with saying "The final answer is _" where "_" is filled in with the final answer. Here are some examples:

Example 1:
Question: Marcus and his friends are starting a recycling project to help the school raise some money for charity. They were tasked to collect different materials and create useful things from those. If Marcus was able to gather 25 milk bottles, and John was able to gather 20 milk bottles, how many milk bottles do they have available for recycling?

Solution: 
Marcus gathered 25 milk bottles and John gathered 20. 
To find how many milk bottles they have available for recycling, we have to add the number of milk bottles they collected together. 
25 + 20 = 45. 
The final answer is 45.

Example 2: 
Question: A school buys 1,093 chairs. If it wants to distribute them equally into 35 classrooms, how many more chairs should be purchased?

Solution:
The school has 1,093 chairs and wants to distribute them equally into 35 classrooms.
To solve this problem, we first need to find how many chairs we will have left over if we divide the current number of chairs evenly.
1,093 / 35 = 31.23
This means we can currently give each classroom 31 chairs. 
To find how many we have left over, we need to multiply the number of chairs for each classroom (31) by the number of classrooms (35) and subtract the total from the 1,093 chairs we started with.
35 x 31 = 1,085
1,093 - 1,085 = 8 
This means we will have 8 chairs left over after giving each classroom 31 chairs. 
To find how many more chairs we need to purchase to give every classroom the same number, we have to subtract the number of remaining chairs (8) from the number of classrooms (35).
35 - 8 = 27
This means we will have to buy 27 more chairs. 
The final answer is 27. 

Example 3:
Question: Mrs. Hilt bought 15 boxes of citrus fruits from a fundraiser. She paid $12 for each box. If 6% sales tax was added to the total cost, how much was her total bill?

Solution:
To solve this problem, we first need to determine how much money Mrs. Hilt spent before the 6% tax was added to the total cost. 
To find the amount she spent before tax, we need to multiply the number of boxes of citrus she bought (15) by the cost for each box ($12).
15 x 12 = 180
To find the total amount she spent after tax, we need to multiply the total amount she spent before tax (180) by the tax rate (6%) and add the taxed amount to the total. 
180 x .06 = 10.8
180 + 10.8 = 190.8
After tax, Mrs. Hilt spent $190.80 in total. 
The final answer is 190.8

<|start_header_id|>user<|end_header_id|>

"""

solutions = []
for i in range(0, len(df)):
    answer = df.iloc[i]['answer']
    try: 
        answer = answer.split(" ")[0]
        answer = float(answer)
    except: 
        answer = answer 
        
    model_answer = ""
    count = 0 
    while model_answer!=answer and count<20:
        count+=1
        final_prompt = f"""Write a solution for this word problem, remembering to exactly follow the format of the examples: 

Question: {df.iloc[i]['question']}

Solution: 
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
        
        final_prompt = prompt + final_prompt
        text = pipe(final_prompt)
        text = text[0]['generated_text'].split(final_prompt)[1]
        try:
            model_answer = text.split("The final answer is ")[-1]
            text = text.split("Solution:")[-1]
            try: 
                model_answer = float(model_answer)
                if model_answer == answer:
                    solutions.append(text)
                    temp_df = df[:i+1]
                    temp_df['solutions'] = solutions
                    temp_df.to_csv("data/ASDIV_Standards_Solutions_Llama70B.csv")
                elif count == 20:
                    solutions.append("NOT SOLVED")
                    temp_df = df[:i+1]
                    temp_df['solutions'] = solutions
                    temp_df.to_csv("data/ASDIV_Standards_Solutions_Llama70B.csv")
            except:
                if model_answer == answer:
                    solutions.append(text)
                    temp_df = df[:i+1]
                    temp_df['solutions'] = solutions
                    temp_df.to_csv("data/ASDIV_Standards_Solutions_Llama70B.csv")
                elif count == 20:
                    solutions.append("NOT SOLVED")
                    temp_df = df[:i+1]
                    temp_df['solutions'] = solutions
                    temp_df.to_csv("data/ASDIV_Standards_Solutions_Llama70B.csv")
        except:
            pass
        
df['solutions'] = solutions
df.to_csv("data/ASDIV_Standards_Solutions_Llama70B.csv")