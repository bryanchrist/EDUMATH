import os 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch 
import numpy as np 
import pandas as pd
import json
model = "google/gemma-3-27b-it"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=torch.bfloat16)

annotations = pd.read_csv("data/all_model_samples.csv")
annotations = annotations[annotations['model_labels']==0]
df = pd.read_csv("data/annotations_copy.csv")
pipe = pipeline(
        "text-generation", 
        model=model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        tokenizer = tokenizer, 
        max_new_tokens = 500, #Llama
        do_sample = False)

formatted_examples = f"""Example 1:

Grade Level: {df[df['solvability']==0].iloc[0]['grade']}

Math Topic(s): 
{df[df['solvability']==0].iloc[0]['math_topic']}

Question: {df[df['solvability']==0].iloc[0]['question']}

Solution: 
{df[df['solvability']==0].iloc[0]['solution']}

Is this question high quality? 
No. The question is not high quality because it is not solvable because it does not define how many books we are starting with so it is impossible to determine how many will be left over. 

Error Type: solvability

Specific Error: It cannot be solved with the information present

Example 2:

Grade Level: {df[df['accuracy']==0].iloc[1]['grade']}

Math Topic(s): 
{df[df['accuracy']==0].iloc[1]['math_topic']}

Question: {df[df['accuracy']==0].iloc[1]['question']}

Solution: 
{df[df['accuracy']==0].iloc[1]['solution']}

Is this question high quality? 
No. This question is not high quality because the solution is incorrect. To solve this problem, we need to divide 9 by 3/4 instead of multiply 9 by 3/4. Therefore, the final answer should be 12 instead of 6 (9 / 3/4 = 12).

Error Type: accuracy

Specific Error: The final answer is wrong

Example 3:

Grade Level: 5

Math Topic(s): 
1. Solving a problem that requires writing an equation with a single uknown variable using addition, subtraction, multiplication, and/or division

Question: Danny bought a box of pencils and divided them evenly among 7 friends. Each friend got 9 pencils. How many pencils were in the box?

Solution:
To solve this problem, we can write an equation with a single variable representing the unknown quantity.

Let the unknown quantity (the total number of pencils in the box) be represented by the variable ""p.""

We know Danny evenly divided the pencils among 7 friends, and each friend got 9 pencils. Thus, we can write the equation:

p / 7 = 9

To solve for p, we multiply both sides of the equation by 7:

p = 9 x 7  
p = 63

Therefore, the box had 63 pencils.

The final answer is 63.

Is this question high quality?
No. The question is not high quality because the solution is too confusing for a student to read. While correct, the solution includes math notation that would be unfamiliar for a young learner. 

Error Type: accuracy

Specific Error: The solution is too complicated for a student/teacher to read

Example 4:

Grade Level: 3

Math Topic(s):
1. Single-step multiplication and/or division without remainders of whole numbers through 10 x 10

Question: There are 6 cats at the park. Each cat has 7 toes. How many toes do the cats have in total?

Solution:
To solve this problem, we need to multiply the number of cats (6) by the number of toes each cat has (7)
6 x 7 = 42
The cats have 42 toes in total. 
The final answer is 42. 

Is this question high quality?
No. The question is not high quality because it is not educationally inappropriate. Specifically, it contains misinformation (that cats have 7 toes). 

Error Type: educational appropriateness

Specific Error: It is strange, confusing, contains conflicting information, and/or is not based in reality (e.g., contains misinformation)

Example 5:

Grade Level: {df[df['standards_alignment']==0].iloc[5]['grade']}

Math Topic(s): 
{df[df['standards_alignment']==0].iloc[5]['math_topic']}

Question: {df[df['standards_alignment']==0].iloc[5]['question']}

Solution: 
{df[df['standards_alignment']==0].iloc[5]['solution']}

Is this question high quality? 
No. This question is not high quality because it does not incorporate important parts of the specified numbered topic(s). Specifically, it does not incorporate division with decimals. 

Error Type: standards alignment

Specific Error: It does not address some important parts of the numbered topic(s) or one or more of the numbered topic(s)

Example 6:

Grade Level: 5

Math Topic(s): 
1. Single-step and multistep addition, subtraction, multiplication, and/or division with whole numbers where:
   i) sums, differences, and products do not exceed five digits;
   ii) factors do not exceed two digits by three digits;
   iii) divisors do not exceed two digits; or
   iv) dividends do not exceed four digits
2. Division with remainders
3. Single-step and multistep addition, subtraction, and/or multiplication of decimals
4. Single-step division with decimals

Question: A Reptile has 18 boxes of food to pack into crates that hold 3 boxes each. If the Reptile packs the boxes into crates, how many crates will the Reptile have, and how many boxes will be left over?

Solution: 
To find out how many crates the Reptile will have and how many boxes will be left over, we need to divide the total number of boxes (18) by the number of boxes each crate can hold (3).
18 / 3 = 6 with a remainder of 0.
This means the Reptile will have 6 crates and there will be no boxes left over.
The final answer is 6.

Is this question high quality? 
No. This question is not high quality because while it is solvable, has an accurate solution, and is educationally appropriate, it does not meaningfully incorporate each of the numbered topics. Specifically, it does not require any decimal operations (Topics 3 and 4). 

Error Type: standards alignment

Specific Error: It does not address some important parts of the numbered topic(s) or one or more of the numbered topic(s)

Example 7:

Grade Level: 5

Math Topic(s): 
1. Single-step and multistep addition, subtraction, multiplication, and/or division with whole numbers where:
   i) sums, differences, and products do not exceed five digits;
   ii) factors do not exceed two digits by three digits;
   iii) divisors do not exceed two digits; or
   iv) dividends do not exceed four digits
2. Division with remainders
3. Single-step and multistep addition, subtraction, and/or multiplication of decimals
4. Single-step division with decimals

Question: A Virtual Reality (VR) company has 48 headsets to package and ship to customers. They want to put 4 headsets in each box. How many boxes will they need to buy, and how many headsets will be left over?

Solution: 
To find out how many boxes they need, we need to divide the total number of headsets (48) by the number of headsets per box (4).
48 / 4 = 12
This means they can fill 12 boxes with 4 headsets each.
To find the remainder, we can multiply the number of boxes (12) by the number of headsets per box (4) and subtract from the total number of headsets.
12 x 4 = 48
48 - 48 = 0
Since 48 divided by 4 has no remainder, they will not have any headsets left over.
The final answer is 12.

Is this question high quality? 
No. This question is not high quality because while it is solvable, has an accurate solution, and is educationally appropriate, it does not meaningfully incorporate each of the numbered topics. Specifically, it does not require any decimal operations (Topics 3 and 4).

Error Type: standards alignment

Specific Error: It does not address some important parts of the numbered topic(s) or one or more of the numbered topic(s)

Example 8:

Grade Level: 4

Math Topic(s): 
1. Single-step multiplication of a whole number 12 or less with a unit fraction with denominators 2, 3, 4, 5, 6, 8, 10, and 12

Question: A Power Ranger is making packs of energy bars for a hike. She needs to put 1/4 of a box of energy bars into each pack. If she has 8 boxes of energy bars, how many packs can she make?

Solution: 
To find out how many packs the Power Ranger can make, we first need to multiply the number of boxes she has by the fraction of a box that each pack needs.
The Power Ranger has 8 boxes and each pack needs 1/4 of a box.
8 × 1/4 = 8 ÷ 4 = 2
This means that 8 boxes can make 2 packs, since 8 × 1/4 is the same as 8 ÷ 4. 
So, the Power Ranger can make 2 packs with 8 boxes of energy bars and each pack having 1/4 of a box.
However, to get the correct answer we have to do 8 x 4 = 32 and 1/4 x 32 = 8. 
The final answer is 32.

Is this question high quality?
No. This question is not high quality because the solution is inaccurate. Specifically, while the final answer is correct, it contains intermediate reasoning steps that are incorrect and would be inappropriate to present to a young learner.

Error Type: accuracy

Specific Error: The final answer is correct but the intermediate reasoning is wrong"""

error_types = []
specific_errors = []
for i in range(0, len(annotations)):
    query = f"""Grade Level: {annotations.iloc[i]['grade']}

Math Topic(s): 
{annotations.iloc[i]['math_topic']}

Question: {annotations.iloc[i]['question']}

Solution:
{annotations.iloc[i]['solution']}

Evaluation: {annotations.iloc[i]['model_reasoning'].split("<end_of_turn>")[0].strip()}

Error Type: """
    prompt = f"""<bos><start_of_turn>user
You are an experienced elementary school teacher tasked with reading an evaluation of a word problem and solution for 3rd-5th grade students written by a less experienced teacher and identifying the main error in the problem and/or solution that they noticed. You will have acess to the grade level, math topic, question, solution, and the less experienced teacher's evaluation of the word problem. Your job is to read the evaluation and identify which of the four criteria below it failed to incorporate as well as which specific error it made for the criteria it failed to incorporate. If the word problem violates more than 1 criteria according to the evaluation, your job is to identify the most important criteria the word problem failed to incorporate. The criteria and associated errors are:
    
Solvability:
A solvable question means that it can be solved with the information present and does not contain a mathematical scenario that is impossible (e.g., giving away more money or items than you have). A question is not solvable for one of two reasons:
1. It cannot be solved with the information present
2. It contains a mathematical scenario that is impossible
    
Accuracy:
An accurate solution is one where the final answer and intermediate reasoning are both correct. A solution is inaccurate if:
1. The final answer is wrong
2. The final answer is correct but the intermediate reasoning is wrong, does not make sense, is unnecessarily repetitive, and/or is too complicated for a student/teacher to read
    
Educational Appropriateness:
An educationally appropriate question is one you would feel comfortable giving to a student in a 3rd-5th grade school setting. Educationally appropriate questions contain content and context appropriate for students in a school setting. There are four main reasons why a question would be educationally inappropriate:
1. It contains material inappropriate for a school setting (e.g., language about harming someone)
2. It is strange, confusing, contains conflicting information, and/or is not based in reality (e.g., contains misinformation)
3. It requires no mathematical operations to solve because it gives the answer away
4. It is inappropriate for a different reason
    
Standards alignment:
A standards aligned question is one that adequately addresses important elements from EACH pre-specified numbered math topic. There are four main reasons why a question would not be standards aligned:
1. It is too hard for the given topic(s)
2. It does not address some important parts of the numbered topic(s) or one or more of the numbered topic(s)
3. It does not address the numbered topic(s) at all
4. It requires additional math topics or operations that are not listed in the specified math topic(s) 

Here are some examples of successful evaluations:
{formatted_examples}
    
Now evaluate this word problem and remember to both identify which criteria the problem failed to incorporate using the "Error Type:" heading and the specific error it made for that criteria using the "Specific Error:" heading. The text in the "Specific Error:" heading should exactly match one of the error types for the criteria defined above. Do not include any information other than the specific error type in the "Specific Error:" heading. Make sure to include BOTH the error type and the specific error in your response.
{query} <end_of_turn>
<start_of_turn>model"""
    text = pipe(prompt)
    #print(f"Query: {query}\n\n{text[0]['generated_text'].split('<start_of_turn>model')[1]}")
    text = text[0]['generated_text'].split('<start_of_turn>model')[1]
    if "Error Type:" in text:
        error_type = text.split("Error Type: ")[1].split("Specific Error: ")[0].strip()
    else: 
        error_type = text.split("Specific Error: ")[0].strip()
    error_types.append(error_type)
    specific_error= text.split("Specific Error: ")[1].strip()
    specific_errors.append(specific_error)
    temp = annotations[:i+1]
    temp['error_type'] = error_types
    temp['specific_error']= specific_errors
    temp.to_csv("data/model_samples_with_error_annotations.csv", index = False)