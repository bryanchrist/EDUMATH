import os
from datasets import load_dataset, DatasetDict
import evaluate
import pandas as pd
import numpy as np
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm
import logging
from torch import nn
# Load Dataset in pandas format to calculate class weights
df = pd.read_csv('data/classifier_data.csv')
# Find class weights for modified training objective based on inverse class balance
neg_weight = df['label'].mean()
pos_weight = 1 - neg_weight

# Load the dataset as HuggingFace Dataset
dataset = load_dataset('csv', data_files="data/classifier_data.csv")

# Do train/valid/test split
dataset_train_valid_test = dataset['train'].train_test_split(test_size = .2, seed = 42)
dataset_valid_test = dataset_train_valid_test['test'].train_test_split(test_size = .5, seed = 42)
# Create a DatasetDict to hold the splits
train_test_valid_dataset = DatasetDict({
    'train': dataset_train_valid_test['train'],
    'test': dataset_valid_test['train'],
    'valid': dataset_valid_test['test']
})

# Set up tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

# Preprocess and collate data
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=False)

tokenized_train_test_valid_dataset = train_test_valid_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# This is a function to load and compute the metrics for the classifier
def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    balanced_accuracy = evaluate.load('hyperml/balanced_accuracy')
    roc_auc_score= evaluate.load('roc_auc')
    f1 = evaluate.load('f1')
    precision = evaluate.load('precision')
    recall = evaluate.load('recall')
    predictions, labels = eval_pred
    predicted_labels = np.argmax(predictions, axis=1)
    
    #Convert NumPy array to PyTorch tensor
    predictions_tensor = torch.from_numpy(predictions)
    
    #Apply softmax to convert logits to probabilities
    probabilities = F.softmax(predictions_tensor, dim=1)
   
    #Convert probabilities back to NumPy array
    probabilities_np = probabilities.numpy()
    
    accuracy = accuracy.compute(predictions=predicted_labels, references=labels)
    balanced_accuracy = balanced_accuracy.compute(predictions=predicted_labels, references=labels)
    roc_auc = roc_auc_score.compute(references = labels, prediction_scores = probabilities_np[:, 1])
    f1 = f1.compute(predictions=predicted_labels, references=labels)
    precision = precision.compute(predictions=predicted_labels, references=labels)
    recall = recall.compute(predictions=predicted_labels, references=labels)
    return {'accuracy':  accuracy, 'balanced_accuracy' : balanced_accuracy, 'f1':f1, 'auc': roc_auc, 'precision': precision, 'recall': recall}

# Create labels for classifier
id2label = {0: "GOOD", 1: "NOT GOOD"}
label2id = {"GOOD": 0, "NOT GOOD": 1}

# Load ModernBERT
model = AutoModelForSequenceClassification.from_pretrained(
   "answerdotai/ModernBERT-large", num_labels=2, id2label=id2label, label2id=label2id, device_map = "auto", torch_dtype = torch.bfloat16)

# Create tensor for class weights
class_weights = torch.tensor([neg_weight, pos_weight], device = model.device)

# Create class for weighted trainer
class WeightedBertTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(torch.bfloat16)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels") 
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(logits, dim=1)

        # Assuming binary classification, so consider only the positive class (index 1)
        positive_class_probabilities = probabilities[:, 1]
        # Use weighted binary cross-entropy loss
        if self.class_weights is not None:
            # Ensure class_weights tensor matches the size of the labels tensor
            loss_fct = nn.CrossEntropyLoss(weight = self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        else:
            # If class weights are not provided, use regular binary cross-entropy loss
            loss = F.binary_cross_entropy(positive_class_probabilities, labels.float())
                
        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # Multi-GPU
        if self.args.n_gpu > 1:
            loss = loss.mean()

        loss.backward()

        return loss.detach()

        
# Set up training arguments
training_args = TrainingArguments(
    output_dir="EDUMATH_classifier",
    learning_rate=.00001,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    warmup_ratio = .1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_strategy = 'epoch',
    logging_steps = 1,
    report_to="wandb")

# Set up trainer
trainer = WeightedBertTrainer(class_weights = class_weights,
     model=model,
     args=training_args,
     train_dataset=tokenized_train_test_valid_dataset["train"],
     eval_dataset=tokenized_train_test_valid_dataset["valid"],
     tokenizer=tokenizer,
     data_collator=data_collator,
     compute_metrics=compute_metrics,
 )

# Train model
trainer.train()

# Evaluate performance on test data
trainer.evaluate(eval_dataset = tokenized_train_test_valid_dataset["test"])

#Save final model
trainer.model.save_pretrained("EDUMATH_classifier/best_model")