from evaluate import load
import pandas as pd
import numpy as np 
import itertools
import os
bertscore = load("bertscore")

math_instruct_gsm8k = pd.read_csv('data/gsm8k_questions.csv')
math_instruct_gsm8k = math_instruct_gsm8k.drop_duplicates(subset ='instruction')
math_instruct_gsm8k['question'] = math_instruct_gsm8k['instruction']
math_instruct_gsm8k = math_instruct_gsm8k.sample(frac = 1, random_state = 42)
gsm8k = pd.read_csv('data/gsm8k_original.csv')
gsm8k = gsm8k.sample(frac = 1, random_state = 42)
asdiv = pd.read_csv('data/ASDiv_clean-Copy1.csv')
asdiv = asdiv.sample(frac = 1, random_state = 42)
svamp = pd.read_json('data/SVAMP.json')
svamp['question'] = svamp['Body'] + " " + svamp['Question']
svamp = svamp.sample(frac = 1, random_state = 42)
gsm_hard = pd.read_json('data/gsmhard.json')
gsm_hard = gsm_hard.sample(frac = 1, random_state = 42)
egsm = pd.read_csv('data/egsm.csv')
egsm = egsm.sample(frac = 1, random_state = 42)
df = pd.read_csv("data/all_model_samples.csv")
stem = pd.read_csv("data/stem.csv")
stem = stem.sample(frac = 1, random_state = 42)
asdiv_subset = pd.read_csv("data/asdiv_corrected.csv")
asdiv_subset = asdiv_subset.sample(frac = 1, random_state = 42)
mathwizards = pd.read_csv("data/mathwizards_clean.csv")
mathwizards = mathwizards.sample(frac = 1, random_state = 42)
# Note this function expects two pandas dfs as input, column names for the questions in each df as strings, and optional arguments for whether 
# you are calculating within df bertscore and the limit for how many rows you want to compare. The function itself calculates bertscore for all pairwise comparisons
# within the specified limit. 
import numpy as np

def score(df1, df2, df1var, df2var, same_df=False, limit=1000):
    # Create lists to store metrics
    precision = []
    recall = []
    f1 = []
        
    # Dynamically adjust limits
    lim1 = min(limit, len(df1))
    lim2 = min(limit, len(df2))

    refs = []
    preds = []

    if not same_df:
        # Shuffle dataframes
        df1 = df1.sample(frac=1).reset_index(drop=True)
        df2 = df2.sample(frac=1).reset_index(drop=True)

        # Use a set to track compared pairs
        compared = set()

        for i in range(lim1):
            for j in range(lim2):
                # Skip if (i,j) or (j,i) was already compared
                if (i, j) in compared or (j, i) in compared:
                    continue  

                compared.add((i, j))

                ref = str(df1.iloc[i][df1var])
                pred = str(df2.iloc[j][df2var])
                preds.append(pred)
                refs.append(ref)

                if len(preds) == 128:
                    results = bertscore.compute(
                        predictions=preds,
                        references=refs,
                        model_type="distilbert-base-uncased",
                        lang="en",
                        batch_size=128
                    )
                    precision.append(np.mean(results['precision']))
                    recall.append(np.mean(results['recall']))
                    f1.append(np.mean(results['f1']))
                    refs, preds = [], []

    else:
        # Compare only unique pairs i < j for within-df comparison
        for i in range(lim1):
            for j in range(i + 1, lim2):
                ref = str(df1.iloc[i][df1var])
                pred = str(df2.iloc[j][df2var])
                preds.append(pred)
                refs.append(ref)

                if len(preds) == 128:
                    results = bertscore.compute(
                        predictions=preds,
                        references=refs,
                        model_type="distilbert-base-uncased",
                        lang="en",
                        batch_size=128
                    )
                    precision.append(np.mean(results['precision']))
                    recall.append(np.mean(results['recall']))
                    f1.append(np.mean(results['f1']))
                    refs, preds = [], []

    return precision, recall, f1
#loop over each dataset and calculate overall within-dataset bertscore using same_df = True and save to text file. For df, the bertscores should be grouped by model
#then loop over df by model and compare it to asdiv_subset
def save_result(dataset_name, model_name, grade1, grade2, comp_type, precision, recall, f1, filename="bertscore_results.csv"):
    """
    Save a single result to CSV immediately.
    """
    row = {
        "dataset": dataset_name,
        "model": model_name,
        "grade1": grade1,
        "grade2": grade2,
        "comparison_type": comp_type,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    # If file exists, append; else create
    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)
        df_existing = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
        df_existing.to_csv(filename, index=False)
    else:
        pd.DataFrame([row]).to_csv(filename, index=False)


# ---------------------------------------------------
# 1. Within-dataset (same_df=True)
# ---------------------------------------------------
all_datasets = {
    "math_instruct_gsm8k": math_instruct_gsm8k,
    "gsm8k": gsm8k,
    "asdiv": asdiv,
    "svamp": svamp,
    "gsm_hard": gsm_hard,
    "egsm": egsm,
    "stem": stem, 
    "asdiv_subset": asdiv_subset,
    "mathwizards": mathwizards,
}

for name, dataset in all_datasets.items():
    scores = score(dataset, dataset, 'question', 'question', same_df=True)
    save_result(name, None, None, None, "within_dataset",
                np.mean(scores[0]), np.mean(scores[1]), np.mean(scores[2]))


# ---------------------------------------------------
# 2. df grouped by model → within-dataset
# ---------------------------------------------------
for model, subset in df.groupby("model"):
    scores = score(subset, subset, 'question', 'question', same_df=True)
    save_result("df", model, None, None, "within_dataset_model",
                np.mean(scores[0]), np.mean(scores[1]), np.mean(scores[2]))


# ---------------------------------------------------
# 3. df vs asdiv_subset (overall + by grade)
# ---------------------------------------------------
# Overall per model
for model, subset in df.groupby("model"):
    scores = score(subset, asdiv_subset, 'question', 'question')
    save_result("df_vs_asdiv", model, None, None, "overall_model",
                np.mean(scores[0]), np.mean(scores[1]), np.mean(scores[2]))

