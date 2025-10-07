import os 
# Set up Wandb
os.environ["WANDB_PROJECT"] = "EDUMATH"  # name your W&B project
working_dir = "" #SET YOUR WORKING DIRECTORY IF DESIRED

# Create the name of the directories to store the models.
output_directory = os.path.join(working_dir, "gemma3_12b_sft")

#Create the directories if they don't exist.
if not os.path.exists(working_dir):
    os.mkdir(working_dir)
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

os.environ["WANDB_LOG_MODEL"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
import torch
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-12b-it')
model = AutoModelForCausalLM.from_pretrained('google/gemma-3-12b-it', device_map="auto", torch_dtype=torch.bfloat16, attn_implementation='eager')

from datasets import Dataset
import pandas as pd

stem = pd.read_csv("data/stem.csv")
stem = stem.sample(frac = 1, random_state = 42) #set random seed for reproducibility
from sklearn.model_selection import train_test_split

# Split df into 85% train, 15% test with a fixed random seed
train, val = train_test_split(stem, test_size=0.15, random_state=42)
train = Dataset.from_pandas(train)
val = Dataset.from_pandas(val)
train = train.map(lambda samples: tokenizer(samples['instruct_summarized']), batched = True)
val = val.map(lambda samples: tokenizer(samples['instruct_summarized']), batched = True)

from transformers import TrainingArguments

def create_training_arguments(path, learning_rate=0.000001, epochs=5, save_steps = 1500, eval_steps=500):
    training_args = TrainingArguments(
        output_dir=path,  # Where the model predictions and checkpoints will be written
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=learning_rate, 
        num_train_epochs=epochs,
        logging_steps=eval_steps,
        eval_strategy="steps",  # Perform evaluation at regular intervals
        eval_steps=eval_steps,  # Evaluate every `eval_steps` steps
        save_steps=save_steps,
        load_best_model_at_end=True,
        report_to = "wandb",
        do_eval = True,
        warmup_ratio = .1,
    )
    return training_args

training_args = create_training_arguments(output_directory)

from transformers import Trainer, DataCollatorForLanguageModeling

def create_trainer(model, training_args, train_dataset, eval_dataset):
    trainer = Trainer(
        model=model,  
        args=training_args,  # The args for the training.
        train_dataset=train_dataset,  
        eval_dataset = eval_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )
    return trainer

trainer = create_trainer(model, training_args, train, val)

trainer.train()

trainer.model.save_pretrained(f"{output_directory}/best_model")