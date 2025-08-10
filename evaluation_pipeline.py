from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from data_loading_util import load_trace_data, load_pairs_data, load_next_activity_data, load_discovery_data
from prompt_builder_util import build_few_shot_prompt_for_trace_anomaly, build_few_shot_prompt_for_activity_anomaly, build_few_shot_prompt_for_next_activity, build_few_shot_prompt_for_dfg, build_few_shot_prompt_for_process_tree
from functools import partial
import torch
import os
import pandas as pd
from datasets import concatenate_datasets
import datetime
from evaluation_util import compute_footprint_fitness, compute_footprint_matrix_pairs, generate_traces_from_tree, compute_footprint_matrix
from statistics import mean
import ast
import gc

tqdm.pandas()


LLAMA_MODEL = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
MISTRAL_MODEL = "unsloth/Mistral-Large-Instruct-2407-bnb-4bit"

# ---------------- Configuration ----------------
task_name = "activity_anomaly"  # Excluded task (i.e., the one we validate/evaluate on)
mode = "evaluation"  # "validation" (second split) or "evaluation" (third split)
model_type = "final" # "checkpoint" (only for development, recommended not to use) or "final" (normal Lora model name from hugging-face or locally); for "evaluation"-mode always "final"

# base models
model_name = LLAMA_MODEL
# model_name = MISTRAL_MODEL

# instruction-tuned models
# model_name = "pyrihtm/lora_Llama-3.3-70B-Instruct_it4pm_anomaly"
# model_name = "pyrihtm/lora_Llama-3.3-70B-Instruct_it4pm_prediction"
# model_name = "pyrihtm/lora_Llama-3.3-70B-Instruct_it4pm_discovery"
# model_name = "pyrihtm/lora_Mistral-Large-Instruct-2407_it4pm_anomaly"
# model_name = "pyrihtm/lora_Mistral-Large-Instruct-2407_it4pm_prediction"
# model_name = "pyrihtm/lora_Mistral-Large-Instruct-2407_it4pm_discovery"

model_name_label = model_name.replace("/", "_") # replace "/" for file saving

num_samples = "all" # Limit dataset size or "all"; make sure to use an even number for anomaly tasks, otherwise will be turned to an even number dynamically
num_shots = 0 # make sure to use an even number for anomaly tasks (pairs of negative and positive samples)

max_seq_length = 1024
dtype = torch.bfloat16
load_in_4bit = True

#------------------ Development Configuration (can be ignored) ------------------
steps_range = [1000, 2000, 3000, 4000] if mode == "validation" and model_type == "checkpoint" else [0]

task_types = ["activity_anomaly", "trace_anomaly", "next_activity", "dfg_generation", "pt_generation"]
if task_name not in task_types:
    raise ValueError(f"Invalid task_name '{task_name}'. Must be one of {task_types}.")

modes = ["validation", "evaluation"]
if mode not in modes:
    raise ValueError(f"Invalid mode '{mode}'. Must be one of {modes}.")

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ---------------- Load Model & Tokenizer ----------------
def load_model_and_tokenizer(checkpoint_dir):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_dir,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left" # LLMs continue generation of the prompt, therefore we're adding padding tokens before the prompt

    if model_name == MISTRAL_MODEL or "Mistral" in model_name :
        tokenizer.pad_token = tokenizer.eos_token # llama-3.3 has its token
        print("Model = Mistral. Changing pad_token to eos_token.")

    return model, tokenizer

# ---------------- Dataset Selection (Validation vs. Evaluation) ----------------
if task_name == "activity_anomaly":
    train_dataset, val_dataset, test_dataset = load_pairs_data()
elif task_name == "trace_anomaly":
    train_dataset, val_dataset, test_dataset = load_trace_data()
elif task_name == "next_activity":
    train_dataset, val_dataset, test_dataset = load_next_activity_data()
elif task_name == "dfg_generation":
    train_dataset, val_dataset, test_dataset = load_discovery_data()
elif task_name == "pt_generation":
    train_dataset, val_dataset, test_dataset = load_discovery_data()

# Load datasets
dataset = val_dataset if mode == "validation" else test_dataset

# ---------------- Balance True/False Labels for Anomaly Tasks ----------------
if num_samples == "all" or num_samples > len(dataset):
    num_samples = len(dataset)

if task_name in ["activity_anomaly", "trace_anomaly"]:
    label_column = "ds_labels"
    positive_samples_full = dataset.filter(lambda x: x[label_column] == "True")
    negative_samples_full = dataset.filter(lambda x: x[label_column] == "False")
    max_samples_per_class = 2 * min(positive_samples_full.num_rows, negative_samples_full.num_rows)

    if num_samples > max_samples_per_class:
        num_samples = max_samples_per_class

    if num_samples % 2 != 0:
        num_samples -= 1

    samples_per_class = num_samples // 2

    # Separate positive and negative samples
    positive_samples = positive_samples_full.shuffle(seed=4).select(range(samples_per_class))
    negative_samples = negative_samples_full.shuffle(seed=4).select(range(samples_per_class))
    dataset = concatenate_datasets([positive_samples, negative_samples])

dataset = dataset.shuffle(seed=3407).select(range(num_samples))

# ---------------- Select Samples for Few-Shot Inference ----------------
def get_few_shot_examples(task_name, train_dataset, num_shots, seed=3407):
    """
    Returns a list (or small Dataset) of `num_shots` examples from the train_dataset
    for the given task. For anomaly tasks, ensures an even split between True/False.
    If num_shots=0, returns an empty list.
    """

    if num_shots == 0:
        return []

    if task_name in ["activity_anomaly", "trace_anomaly"]:
        # half True, half False
        assert num_shots % 2 == 0, "num_shots must be even for anomaly tasks"
        half_shots = num_shots // 2

        positives = train_dataset.filter(lambda x: x["ds_labels"] == "True")
        negatives = train_dataset.filter(lambda x: x["ds_labels"] == "False")

        positives = positives.shuffle(seed=seed).select(range(min(half_shots, len(positives))))
        negatives = negatives.shuffle(seed=seed).select(range(min(half_shots, len(negatives))))

        few_shot_ds = concatenate_datasets([positives, negatives]).shuffle(seed=seed)
        # Convert to list of dict
        return [few_shot_ds[i] for i in range(len(few_shot_ds))]

    else:
        # For next_activity, dfg, process_tree, just pick num_shots
        ds_shuffled = train_dataset.shuffle(seed=seed)
        few_shot_ds = ds_shuffled.select(range(min(num_shots, len(ds_shuffled))))
        return [few_shot_ds[i] for i in range(len(few_shot_ds))]


# ---------------- Generation Functions ----------------
def generate_binary_output(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=5,
        return_dict_in_generate=True,
    )
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    decoded_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    processed_text = decoded_text.strip().lower()
    if "true" in processed_text:
        return decoded_text, "True"
    elif "false" in processed_text:
        return decoded_text, "False"
    print(decoded_text, "neither true nor false")
    return decoded_text, "False"

def generate_activity_output(model, tokenizer, prompt, activities):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=25,
        return_dict_in_generate=True,
    )
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    decoded_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    processed_text = decoded_text.strip().lower()
    for activity in activities:
        if activity.strip().lower() in processed_text:
            return decoded_text, activity
        
    print('\n"', processed_text, '" - does not contain any activity')
    return decoded_text, "None"

def generate_dfg_output(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=200,
        return_dict_in_generate=True,
    )
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    decoded_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    # parse list of pars like 'A' -> 'B'\n 'C' -> 'D' into a list of tuples
    parsed = decoded_text.split("[END]")[0]
    parsed = parsed.split("\n")
    parsed = [x.split(" -> ") for x in parsed if " -> " in x]
    parsed = [(x[0].strip(), x[1].strip()) for x in parsed]
    return decoded_text, parsed

def generate_pt_output(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=150,
        return_dict_in_generate=True,
    )
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    decoded_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    parsed = decoded_text.split("[END]")[0].rstrip()
    return decoded_text, parsed

# ---------------- Compute Predictions ----------------
def compute_y(example, model, tokenizer, task, few_shot_examples):
    # Choose prompt based on task
    if task == "activity_anomaly":
        example["true_label"] = example["ds_labels"]
        prompt = build_few_shot_prompt_for_activity_anomaly(few_shot_examples, example)
        decoded_text, prediction = generate_binary_output(model, tokenizer, prompt)

    elif task == "trace_anomaly":
        example["true_label"] = example["ds_labels"]
        prompt = build_few_shot_prompt_for_trace_anomaly(few_shot_examples, example)
        decoded_text, prediction = generate_binary_output(model, tokenizer, prompt)

    elif task == "next_activity":
        example["true_label"] = example["next"]
        prompt = build_few_shot_prompt_for_next_activity(few_shot_examples, example)
        decoded_text, prediction = generate_activity_output(model, tokenizer, prompt, example["unique_activities"])

    elif task == "dfg_generation":
        example["true_label"] = example["dfg"]
        prompt = build_few_shot_prompt_for_dfg(few_shot_examples, example)
        decoded_text, prediction = generate_dfg_output(model, tokenizer, prompt)

    elif task == "pt_generation":
        example["true_label"] = example["pt"]
        prompt = build_few_shot_prompt_for_process_tree(few_shot_examples, example)
        decoded_text, prediction = generate_pt_output(model, tokenizer, prompt)

    example["y"] = prediction
    example["decoded_text"] = decoded_text
    example["prompt"] = prompt
    return example

# ---------------- Prediction Loop ----------------
def run_predictions_loop(val_ds, model, tokenizer, task):
    few_shot_examples = get_few_shot_examples(task, train_dataset, num_shots)
    compute_partial = partial(compute_y, model=model, tokenizer=tokenizer, task=task, few_shot_examples=few_shot_examples)
    return val_ds.map(compute_partial, desc=f"Generating outputs for {task}")


# ---------------- Evaluation Function ----------------
def evaluate(val_ds):
    predicted_labels = val_ds["y"]
    true_labels = val_ds["true_label"]

    if task_name == "next_activity":
        def compute_valid_labels(example):
            valid_labels = {
                row["next"] for row in val_ds
                if row["model_id"] == example["model_id"] and row["prefix"] == example["prefix"]
            }
            # Determine if prediction is valid under multiple-choice conditions
            example["all_true_labels"] = list(valid_labels)  # Convert to list for compatibility
            example["is_correct"] = example["y"] in valid_labels
            example["is_multichoice"] = len(valid_labels) > 1
            return example
        val_ds = val_ds.map(compute_valid_labels)

        adjusted_true_labels = []
        for example in val_ds:
            if example["is_correct"]:
                adjusted_true_labels.append(example["y"])
            else:
                adjusted_true_labels.append(example["true_label"])

        precision_micro = precision_score(adjusted_true_labels, predicted_labels, average='micro', zero_division=0)
        recall_micro = recall_score(adjusted_true_labels, predicted_labels, average='micro', zero_division=0)
        f1_micro = f1_score(adjusted_true_labels, predicted_labels, average='micro', zero_division=0)
        precision_macro = precision_score(adjusted_true_labels, predicted_labels, average='macro', zero_division=0)
        recall_macro = recall_score(adjusted_true_labels, predicted_labels, average='macro', zero_division=0)
        f1_macro = f1_score(adjusted_true_labels, predicted_labels, average='macro', zero_division=0)

        return val_ds, {
            "precision mic": precision_micro,
            "recall mic": recall_micro,
            "f1 mic": f1_micro,
            "precision mac": precision_macro,
            "recall mac": recall_macro,
            "f1 mac": f1_macro,
        }     

    elif task_name in ["activity_anomaly", "trace_anomaly"]:
        precision_micro = precision_score(true_labels, predicted_labels, average='micro', zero_division=0)
        recall_micro = recall_score(true_labels, predicted_labels, average='micro', zero_division=0)
        f1_micro = f1_score(true_labels, predicted_labels, average='micro', zero_division=0)
        precision_macro = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
        f1_macro = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

        return val_ds, {
            "precision mic": precision_micro,
            "recall mic": recall_micro,
            "f1 mic": f1_micro,
            "precision mac": precision_macro,
            "recall mac": recall_macro,
            "f1 mac": f1_macro,
        }
    
    elif task_name == "dfg_generation":
        def compute_fitness(example):

            try:
                true_matrix = compute_footprint_matrix_pairs(ast.literal_eval(example["dfg"]), example["unique_activities"])
            except Exception as e:
                print(f"Exception while computing true_matrix for DFG: {example['dfg']}")
                print(f"Error: {e}")
                true_matrix = None

            try:
                pred_matrix = compute_footprint_matrix_pairs(example["y"], example["unique_activities"])
            except Exception as e:
                print(f"Exception while computing pred_matrix for predicted DFG: {example['y']}")
                print(f"Error: {e}")
                pred_matrix = None
            
            if true_matrix is None or pred_matrix is None:
                example["fitness"] = 0
            else:
                example["fitness"] = compute_footprint_fitness(true_matrix, pred_matrix)

            return example
            
        val_ds = val_ds.map(compute_fitness)
        avg_fitness = mean(val_ds["fitness"])

        return val_ds, {
            "avg_fitness": avg_fitness,
        }
    
    elif task_name == "pt_generation":
        def compute_fitness(example):
            if "tau" in example["pt"]:
                print("tau detected in ground truth, skipping evaluation for this sample.")
                example["fitness"] = None
                return example

            true_tree = example["pt"].replace("\n", " ")
            pred_tree = example["y"].replace("\n", " ")

            try:
                true_str_traces = generate_traces_from_tree(true_tree, example["unique_activities"])
                true_matrix = compute_footprint_matrix(true_str_traces, example["unique_activities"])
                pred_str_traces = generate_traces_from_tree(pred_tree, example["unique_activities"])
                pred_matrix = compute_footprint_matrix(pred_str_traces, example["unique_activities"])

                example["fitness"] = compute_footprint_fitness(true_matrix, pred_matrix)
                return example

            except Exception as e:
                print(f"Trace generation or fitness computation failed: {e}, true_tree: {true_tree}, pred_tree: {pred_tree}")
                example["fitness"] = None
                return example

        val_ds = val_ds.map(compute_fitness)
        valid_fitness_scores = [f for f in val_ds["fitness"] if f is not None]
        avg_fitness = mean(valid_fitness_scores) if valid_fitness_scores else 0

        return val_ds, {
            "avg_fitness": avg_fitness,
        }

# ---------------- Bulk Evaluation ----------------
def bulk_evaluate():
    all_results = []

    for step in steps_range:
        all_inference_logs = []
        if mode == "validation" and model_type == "checkpoint":
            checkpoint = f"{model_name}-{step}"
        else:
            checkpoint = f"{model_name}"  # Final model in evaluation

        model, tokenizer = load_model_and_tokenizer(checkpoint)
        val_ds = run_predictions_loop(dataset, model, tokenizer, task_name)

        output_dir_logs = f"eval/{mode}/{task_name}_{model_name_label}/"
        os.makedirs(output_dir_logs, exist_ok=True)

        val_ds, results = evaluate(val_ds)
        results["step"] = step
        print(results)
        
        all_results.append(results)        

        # Collect inference logs
        for example in val_ds:
            row = {
                "id": example["id"],  
                "prompt": example["prompt"],
                "decoded_text": example["decoded_text"],  # Raw model output
                "prediction": example["y"],  # Model's final prediction after output processing
                "true_label": example["true_label"],  # Correct label
                "unique_activities": example["unique_activities"]
            }

            # Append task-specific fields
            if task_name == "activity_anomaly":
                row["eventually_follows"] = example["eventually_follows"]
            elif task_name == "trace_anomaly":
                row["trace"] = example["trace"]
            elif task_name == "next_activity":
                row["trace"] = example["trace"]
                row["prefix"] = example["prefix"]
                row["is_multichoice"] = example["is_multichoice"]
                row["is_correct"] = example["is_correct"]
                row["all_true_labels"] = example["all_true_labels"]
            elif task_name in ["pt_generation", "dfg_generation"]:
                row["fitness"] = example["fitness"]

            all_inference_logs.append(row)

        inference_df = pd.DataFrame(all_inference_logs)
        inference_df.to_csv(f"{output_dir_logs}logs_{timestamp}_samples-{num_samples}_shots-{num_shots}_step-{step}.csv", index=False, escapechar='\\')

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{output_dir_logs}results_{timestamp}_samples-{num_samples}_shots-{num_shots}.csv", index=False)

    print("Evaluation completed!")
    print(results_df)

bulk_evaluate()
