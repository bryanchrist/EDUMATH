import os 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch 
import numpy as np 
import pandas as pd
import json
df = pd.read_csv("data/asdiv_inaccurate_solutions.csv")
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-27b-it')
model = AutoModelForCausalLM.from_pretrained('google/gemma-3-27b-it', device_map="auto", torch_dtype=torch.bfloat16, attn_implementation='eager')

pipe = pipeline(
    "text-generation", 
    model=model, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    tokenizer = tokenizer, 
    max_new_tokens = 1000, #Llama
    do_sample = False,
)

formatted_examples = f"""Example 1:

Grade Level: 5

Math Topic(s): 
1. Single-step multiplication of a whole number 12 or less with a proper fraction; the denominator of the fraction should be a factor of the whole number

Question: A Native American tribe is making traditional beadwork jewelry to sell at their annual festival. They have 9 spools of beads and each spool needs to be divided into 3/4 to make the correct pattern. How many sets of beads can they make in total?

Solution: 
To find the total number of sets of beads the tribe can make, we need to multiply the number of spools of beads (9) by the fraction of beads per set (3/4).
9 × 3/4 =?
To multiply a whole number by a fraction, we multiply the whole number by the numerator (3) and divide by the denominator (4).
9 × 3 = 27
27 ÷ 4 = 6.75
Since we are making sets of beads, the number of sets needs to be a whole number. 
The final answer is 6.75, but since the question asks for the total number of sets, we need to convert the decimal to a mixed number or just the whole number part, because you can't have a fraction of a set, so we can make 6 sets with 3/4 of beads left over.
The final answer is 6.

Reasoning for why the solution is inaccurate:
The solution is incorrect. To solve this problem, we need to divide 9 by 3/4 instead of multiply 9 by 3/4. Therefore, the final answer should be 12 instead of 6 (9 / 3/4 = 12).

Corrected Solution:
To find the total number of sets of beads the tribe can make, we need to divide the number of spools of beads (9) by the fraction of beads per set (3/4)
9 / 3/4 = 9 x 4/3 = 36/3 = 12
Therefore, the tribe can make 12 sets.
The final answer is 12. 

Example 2:

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

Reasoning for why the solution is inaccurate:
The solution is too confusing for a student to read. While correct, the solution includes math notation that would be unfamiliar for a young learner. 

Corrected Solution:
To solve this problem, we can write an equation that represents the total number of pencils in the box with the variable p.
We know Danny evenly divided the pencils among 7 friends, and each friend got 9 pencils. This makes our equation: 
p / 7 = 9

To solve for p, we multiply both sides of the equation by 7:
p = 9 x 7  
p = 63

Therefore, the box had 63 pencils.
The final answer is 63.

Example 3:

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

Reasoning for why the solution is inaccurate:
The solution is inaccurate. Specifically, while the final answer is correct, it contains intermediate reasoning steps that are incorrect and would be inappropriate to present to a young learner.

Corrected Solution:
To find out how many packs the Power Ranger can make, we need to divide the number of boxes she has by the fraction of a box that each pack needs.
The Power Ranger has 8 boxes and each pack needs 1/4 of a box.
8 / 1/4 = 8 x 4/1 = 32
The Power Ranger can make 32 packs
The final answer is 32.

Example 4:

Grade Level: 5

Math Topic(s): 
1. Single-step and multistep addition, subtraction, multiplication, and/or division with whole numbers where:
   i) sums, differences, and products do not exceed five digits;
   ii) factors do not exceed two digits by three digits;
   iii) divisors do not exceed two digits; or
   iv) dividends do not exceed four digits
2. Single-step and multistep addition, subtraction, and/or multiplication of decimals

Question: A bookshelf has 432 books to be packed into boxes. Each box can hold 12 books. How many boxes are needed to pack all the books, and how many books will be left unpacked?

Solution: 
To find the number of boxes needed to pack all the books, we first need to divide the total number of books (432) by the number of books each box can hold (12).
432 / 12 = 36
This means we can fill 36 boxes completely with 12 books each.
To find the total number of books used to fill these 36 boxes, we multiply the number of boxes (36) by the number of books per box (12).
36 x 12 = 432
Now, we need to check if there are any books left over after filling 36 boxes.
Since 36 x 12 = 432, we have used all the books and there are no books left unpacked.
However, if we had a remainder, we would need to report it as the number of unpacked books.
In this case, since we have no remainder, all the books were packed and there are no books left unpacked.
The final answer is 0.

Reasoning for why the solution is inaccurate:
The solution is inaccurate. Specifically, while the final answer is correct, the solution rambles on about remainders when the problem requires no remainder.

Corrected Solution: 
To find the number of boxes needed to pack all the books, we first need to divide the total number of books (432) by the number of books each box can hold (12).
432 / 12 = 36
This means we can fill 36 boxes completely with 12 books each without having any left over.
The final answer is 36 filled boxes with 0 books left over.

Example 5:

Grade Level: 4

Math Topic(s): 
1. Single-step division with remainders for a one-digit divisor and a two- or three-digit dividend
2. Single-step or multistep addition and/or subtraction with whole numbers less than 1,000,000

Question: A bakery has 480 cookies to package in boxes that hold 12 cookies each. How many boxes can they fill, and how many cookies will be left over after filling the boxes?

Solution: 
To find out how many boxes the bakery can fill and how many cookies will be left over, we first need to divide the total number of cookies (480) by the number of cookies each box can hold (12).
480 / 12 = 40
This means the bakery can fill 40 boxes. 
To find the remainder, we need to multiply the number of boxes (40) by the number of cookies each box can hold (12) and subtract this from the total number of cookies (480).
40 x 12 = 480
480 - 480 = 0
The final answer is 40 boxes with no cookies left over.

Reasoning for why the solution is inaccurate:
The solution is inaccurate. Specifically, while the final answer is correct, the solution includes an unneccesary calculation of a remainder after already finding that there would be no remainder by dividing 480 / 12 and obtaining a whole number (40). The solution should have just said there was no remainder after finding a whole number quotient rather than conducting additional calculations.

Corrected Solution:
To find out how many boxes the bakery can fill and how many cookies will be left over, we first need to divide the total number of cookies (480) by the number of cookies each box can hold (12).
480 / 12 = 40
This means the bakery can fill 40 boxes with 0 cookies left over.
The final answer is 40 boxes with no cookies left over.""" 

solutions = []
for i in range(0, len(df)):
    query = f"""Grade Level: {df.iloc[i]['grade']}

Math Topic(s): 
{df.iloc[i]['math_topic']}

Question: {df.iloc[i]['question']}

Solution:
{df.iloc[i]['solution']}

Reasoning for why the solution is inaccurate:
{df.iloc[i]['gemma_reasoning']}

Corrected Solution:"""
    
    prompt = f"""<bos><start_of_turn>user
You are an experienced elementary school teacher tasked with rewriting solutions for word problems for 3rd-5th grade students written by a less experienced teacher. The word problem and original solution will be based on a grade level and math topic(s) and will contain reasoning identifying what is wrong with the original solution. Your job is to use the reasoning to rewrite the solution to ensure it is accurate, high quality, and appropriate/readable for young learners. 

An accurate solution is one where the final answer and intermediate reasoning are both correct, there is not unnecessary repetition, and the wording is not too complicated for a student/teacher to read.

Here are some examples of successful solution rewrites:
{formatted_examples}

Now rewrite this word problem's solution:
{query} <end_of_turn>
<start_of_turn>model"""

    text = pipe(prompt)
    text = text[0]['generated_text']
    print(text.split("<start_of_turn>model")[1])
    solution = text.split("<start_of_turn>model")[1]
    solutions.append(solution)
    temp = df[:i+1]
    temp['rewritten_solution'] = solutions
    temp.to_csv("data/asdiv_rewrites_gemma.csv", index = False)
    