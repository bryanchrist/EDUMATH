import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', help="Huggingface model to generate samples from, entered as string", type = str)
parser.add_argument('--tokenizer', help="Huggingface tokenizer for specified model, entered as string", type = str)
parser.add_argument('--output_file', help="Desired name of output CSV file with generated data", type = str)
parser.add_argument('--num_samples', help="desired number of samples to generate", type = int, default = 1000)
parser.add_argument('--temperature', help="Temperature for generation", type = float, default = 1.0)
args = parser.parse_args()
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch 
import numpy as np 
import pandas as pd
import json
from peft import PeftModel
df = pd.read_csv("data/stem.csv")
stem = pd.read_csv("data/stem.csv")
if "gemma" in args.model or "sft" in args.model or "kto" in args.model or "dpo" in args.model or "EDUMATH" in args.model:
    if "qwen" not in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation='eager')
        
if "gemma" not in args.model and "sft" not in args.model and "kto" not in args.model and "dpo" not in args.model and "235" not in args.model and "EDUMATH" not in args.model:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)

if "qwen30b" in args.model:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token 
    model.config.pad_token_id = tokenizer.pad_token_id
    
if "235" in args.model:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
    )
    
if "235" not in args.model:
    pipe = pipeline(
        "text-generation", 
        model=model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        tokenizer = tokenizer, 
        max_new_tokens = 1000, #Llama
        do_sample = True, 
        temperature = args.temperature
    )
    
if "235" in args.model:
    pipe = pipeline(
        "text-generation", 
        model=model, 
        torch_dtype="auto", 
        device_map="auto", 
        tokenizer = tokenizer, 
        max_new_tokens = 10000, #Llama
        do_sample = True, 
        temperature = args.temperature
    )
topics = ['Superman', "Batman", "Wonder Woman", "Barbie", "Power Rangers", "basketball", "soccer", "football", "volleyball", 'field hockey',
'Fortnite', 'Spiderman', "Iron Man", "Captain America", "Captain Marvel", "Thor, the God of Thunder", "Black Panther", "Taylor Swift", "swimming",
"Pokémon", "Super Mario", "Naruto", "unicorns", "Hello Kitty", "Minecraft", "lacrosse", "cheer leading", "LeBron James", "Steph Curry", "Patrick Mahomes",
"Serena Williams", "dogs", "cats", "dinosaurs", "Harry Potter", "cars", "planes", "trains", "pizza", "cookies", "ice cream", 'candy', "Frozen (Elsa and Anna)",
    "Minecraft",
    "Star Wars",
    "Paw Patrol",
    "My Little Pony",
    "Minions",
    "Jurassic Park",
    "SpongeBob SquarePants",
    "Disney Princesses",
    "Toy Story",
    "The Incredibles",
    "Scooby-Doo",
    "Peppa Pig",
    "Dora the Explorer",
    "Pikachu",
    "Thomas the Tank Engine",
    "Sonic the Hedgehog",
    "Transformers",
    "Minions",
    "Cinderella",
    "Moana",
    "Shrek",
    "Winnie the Pooh",
    "Tom and Jerry",
    "Sesame Street",
    "The Lion King",
    "Alice in Wonderland",
    "The Little Mermaid",
    "Peter Pan",
    "Aladdin",
    "The Jungle Book",
    "Pocahontas",
    "Beauty and the Beast",
    "Frozen",
    "Ratatouille",
    "Finding Nemo",
    "Cars",
    "Up",
    "The Simpsons",
    "Looney Tunes",
    "Teenage Mutant Ninja Turtles",
    "Scooby-Doo",
    "Mythical Creatures (dragons, unicorns)",
    "Dinosaurs",
    "Space and Astronauts",
    "Robots",
    "Aliens",
    "Exploring the Ocean",
    "Underwater Creatures",
    "Pirates",
    "Fairies",
    "Wizards",
    "Magic Tricks",
    "Time Travel",
    "Detectives and Mystery",
    "Inventions",
    "The Avengers",
    "The Justice League",
    "Dance and Ballet",
    "Music Instruments",
    "Art and Drawing",
    "Science Experiments",
    "Cooking and Baking",
    "DIY Crafts",
    "Board Games",
    "Puzzles",
    "Riddles",
    "Pets (cats, dogs, hamsters)",
    "Farm Animals",
    "Zoo Animals",
    "Wildlife Conservation",
    "Plants and Gardening",
    "Hiking and Nature",
    "Weather and Meteorology",
    "The Solar System",
    "Camping",
    "National Parks",
    "Trains and Railroads",
    "Planes and Aviation",
    "Cars and Racing",
    "Construction Vehicles",
    "Firefighters",
    "Police Officers",
    "Doctors and Nurses",
    "Astronauts and Space Exploration",
    "Animals and Wildlife",
    "Space and Astronomy",
    "Robots and Technology",
    "Underwater Life",
    "Fairy Tales and Folklore",
    "Science Experiments",
    "Outer Space",
    "Weather and Meteorology",
    "Art and Drawing",
    "Music and Instruments",
    "Cooking and Baking",
    "Insects and Bugs",
    "Historical Figures",
    "Countries and Cultures",
    "Mythical Creatures",
    "Magic and Wizards",
    "Friendship and Relationships",
    "Ocean Life",
    "Cars and Vehicles",
    "Famous Inventors",
    "Famous Artists",
    "Ancient Civilizations",
    "Space Exploration",
    "DIY Crafts",
    "Gardening",
    "Environmental Conservation",
    "Time Travel",
    "Pirates and Treasure",
    "Famous Scientists",
    "Computer Programming",
    "Unexplained Mysteries",
    "Planets and the Solar System",
    "Cartoons and Animated Shows",
    "Photography",
    "National Parks",
    "Dance and Ballet",
    "Board Games",
    "Books and Reading",
    "Volcanoes",
    "Mythology",
    "Ancient Egypt",
    "Reptiles and Amphibians",
    "Recycling",
    "Fairy Gardens",
    "Indoor Games",
    "Marine Biology",
    "Virtual Reality",
    "Natural Disasters",
    "Construction and Building",
    "Inventions",
    "the Circus and Performing Arts",
    "Science Fiction",
    "Pottery and Ceramics",
    "Famous Explorers",
    "Birds and Bird Watching",
    "Famous Landmarks",
    "Health and Nutrition",
    "Myths and Legends",
    "Fashion and Clothing",
    "DIY Science Projects",
    "Cultural Festivals",
    "Construction Vehicles",
    "Forests and Trees",
    "Mummies",
    "Famous Composers",
    "Circus Animals",
    "Geology",
    "Farm Life",
    "Travel and Adventure",
    "Ballet and Dance",
    "Whales and Dolphins",
    "Mystery Stories",
    "Hiking and Camping",
    "Games and Puzzles",
    "Space Aliens and UFOs"
]

# Remove duplicates while preserving order
updated_topics = list(dict.fromkeys(topics))

interests = updated_topics
import random
substandard_list = []
for substandard in df['substandard']:
    if substandard not in substandard_list:
        substandard_list.append(substandard)

import random
questions = []
grades = []
standards = []
solutions = []
substandards = []
temp_df = []
topics = []
while len(questions)<args.num_samples:
    for substandard in substandard_list:
        topic = random.choice(interests)
        substandard_df = df[df['substandard']==substandard]
        grade = substandard_df.iloc[0]['grade']
        standard = substandard_df.iloc[0]['standard']
        query = f"""Grade: {substandard_df.iloc[0]['grade']}
    
Math Topic(s): 
{substandard_df.iloc[0]['math_topic']}
    
Question: """
        
        num_examples = min(8, len(substandard_df))
        
        examples = []
        if num_examples<8:
            temp_df = df[df['substandard']!=substandard]
            temp_df = temp_df[temp_df['grade']==grade]
            for i in range(8-num_examples):
                row = temp_df.iloc[random.randint(0, len(temp_df)-1)]
                example = f"""Grade: {row['grade']}
    
Math Topic(s): 
{row['math_topic']}

Question: {row['question']}
    
Solution:
{row['solution']}"""
                if example not in examples:
                    examples.append(example)
                    
        for i in range(num_examples):
            row = substandard_df.iloc[random.randint(0, len(substandard_df)-1)]
            example = f"""Grade: {row['grade']}
    
Math Topic(s): 
{row['math_topic']}

Question: {row['question']}
    
Solution:
{row['solution']}"""
            if example not in examples:
                examples.append(example)
        formatted_examples = "\n\n".join(examples)
        prompt = f"""<bos><start_of_turn>user
You are an experienced teacher tasked with writing word problems and solutions for 3rd-5th grade students. The question you write will be based on a grade level and math topic(s). The question's content should exactly match and incorporate ALL of the mathematical topics and constraints listed in the math topic(s). The question and answer pair you write should be solvable with the information presented in the question, contain an accurate solution, and contain language and context appropriate for a 3rd-5th grade student in a school setting (i.e., no harmful language and topics should be appropriate for school settings).
    
Here are some examples: 
{formatted_examples}
 
Write a new question about {topic} based on the grade level and math topic(s) below. Make sure to incorporate all of the information in the math topic(s) into your question; for example, if the math topic(s) mention remainders, the problem you write should require a remainder. Also make sure to write "Question:" followed by your question and "Solution:\n" followed by your solution. 
{query} <end_of_turn>
<start_of_turn>model"""
        if "wen" not in args.model:
            text = pipe(prompt)
            text = text[0]['generated_text']
            print(text.split("<start_of_turn>model")[1])
            text = text.split("<start_of_turn>model")[1]
            text = text.split("<end_of_turn>")[0]
        if "wen" in args.model:
            prompt = f"""You are an experienced teacher tasked with writing word problems and solutions for 3rd-5th grade students. The question you write will be based on a grade level and math topic(s). The question's content should exactly match and incorporate ALL of the mathematical topics and constraints listed in the math topic(s). The question and answer pair you write should be solvable with the information presented in the question, contain an accurate solution, and contain language and context appropriate for a 3rd-5th grade student in a school setting (i.e., no harmful language and topics should be appropriate for school settings).
    
Here are some examples: 
{formatted_examples}
 
Write a new question about {topic} based on the grade level and math topic(s) below. Make sure to incorporate all of the information in the math topic(s) into your question; for example, if the math topic(s) mention remainders, the problem you write should require a remainder. Also make sure to write "Question:" followed by your question and "Solution:\n" followed by your solution. 
{query}"""
            messages = [
                {"role": "user", "content": prompt}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            text = pipe(prompt)
            text = text[0]['generated_text']
            text = text.split(prompt)[1]
            if "<|im_end|>" in text:
                text = text.split("<|im_end|>")[0]
        if "Question: " in text:
            question = text.split("Question: ")[1]
            question = question.split("Solution:")[0]
            question = question.rstrip() 
        if "Question: " not in text:
            question = ""
            pass
        if "Solution:\n" in text:
            solution = text.split("Solution:\n")[1]
            solution = solution.rstrip() 
        if "Solution:\n" not in text: 
            try:
                solution = text.split("Solution:")[1]
                solution = solution.rstrip() 
            except:
                solution = ""
                pass
        if question!="" and question not in questions and "The final answer is" in solution and question not in df['question']:
            if question not in stem['question']:
                questions.append(question)
                grades.append(grade)
                standards.append(standard)
                substandards.append(substandard)
                solutions.append(solution)
                topics.append(topic)
                temp_df = pd.DataFrame({"question": questions, "solution": solutions, "grade": grades, "standard": standards, "substandard": substandards, "topic": topics})
                temp_df.to_csv(f"data/{args.output_file}.csv", index = False)
            else:
                pass