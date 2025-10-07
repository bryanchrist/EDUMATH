import os 
# Set up Wandb
os.environ["WANDB_PROJECT"] = "EDUMATH"  # name your W&B project
working_dir = "" #SET YOUR WORKING DIRECTORY IF DESIRED

output_directory = os.path.join(working_dir, "EDUMATH_12b")
#Create the directories if they don't exist.
if not os.path.exists(working_dir):
    os.mkdir(working_dir)
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
from datasets import load_dataset, Dataset
from trl import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
os.environ["WANDB_LOG_MODEL"] = "false"

df = pd.read_csv('data/kto_data.csv')
neg_label = round(df['label'].mean()+1, 2)
pos_label = round((1-df['label'].mean())+1, 2)
df = df.sample(frac = 1, random_state = 42) #set random seed for reproducibility
from sklearn.model_selection import train_test_split

# Split df into 85% train, 15% test with a fixed random seed
train, val = train_test_split(df, test_size=0.15, random_state=42)
train = Dataset.from_pandas(train)
val = Dataset.from_pandas(val)

tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-12b-it')
model = AutoModelForCausalLM.from_pretrained('bryanchrist/gemma3_12b_sft', device_map="auto", torch_dtype=torch.bfloat16, attn_implementation='eager')

training_args = KTOConfig(output_dir=output_directory, 
        desirable_weight=pos_label,
        undesirable_weight=neg_label,
        num_train_epochs=5,
        logging_steps=250,
        eval_strategy="steps",  # Perform evaluation at regular intervals
        eval_steps=250,  # Evaluate every `eval_steps` steps
        save_steps=500,
        load_best_model_at_end=True,
        report_to = "wandb",
        do_eval = True,
        warmup_ratio = .1,
        learning_rate = .000005,
    )
trainer = KTOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train, eval_dataset = val)
trainer.train()
trainer.model.save_pretrained(f"{output_directory}/best_model")