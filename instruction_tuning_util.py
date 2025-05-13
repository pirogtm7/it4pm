from datasets import Dataset, concatenate_datasets
import torch
import random
import re
import pandas as pd
import datetime
import importlib
import os
import ast

#--------------------------combine the datasets--------------
def keep_only_columns(ds, keep_cols):
    cols_to_remove = [c for c in ds.column_names if c not in keep_cols]
    return ds.remove_columns(cols_to_remove)

def prepare_trace_ds(ds, task_name="trace_anomaly", skip=False):
    if skip:
        return Dataset.from_dict({})
    ds = ds.rename_column("ds_labels", "gold_answer")
    ds = ds.map(lambda x: {"task_name": task_name})
    keep_cols = {
        "model_id", "unique_activities", 
        "trace",              
        "gold_answer", 
        "task_name"
    }
    ds = keep_only_columns(ds, keep_cols)
    return ds

def prepare_pairs_ds(ds, task_name="activity_anomaly", skip=False):
    if skip:
        return Dataset.from_dict({})
    ds = ds.rename_column("ds_labels", "gold_answer")
    ds = ds.map(lambda x: {"task_name": task_name})
    keep_cols = {
        "model_id", "unique_activities", 
        "eventually_follows",
        "gold_answer", 
        "task_name"
    }
    ds = keep_only_columns(ds, keep_cols)
    return ds

def prepare_na_ds(ds, task_name="next_activity", skip=False):
    if skip:
        return Dataset.from_dict({})
    ds = ds.rename_column("next", "gold_answer")
    ds = ds.map(lambda x: {"task_name": task_name})
    keep_cols = {
        "model_id", "unique_activities", 
        "prefix",
        "gold_answer", 
        "task_name"
    }
    ds = keep_only_columns(ds, keep_cols)
    return ds

def prepare_discovery_ds(ds, skip_dfg=False, skip_pt=False):
    """
    For discovery dataset, we have two tasks: dfg and process_tree.
    We'll create two subsets:
      - one with task_name="dfg"   and gold_answer= ds["dfg"]
      - one with task_name="process_tree" and gold_answer= ds["pt"]
    We can skip one or both tasks if needed.
    """
    # (A) DFG subset
    if not skip_dfg:
        dfg_sub = ds.rename_column("dfg", "gold_answer")
        dfg_sub = dfg_sub.map(lambda x: {"task_name": "dfg"})
        keep_cols_dfg = {
            "model_id", "unique_activities", 
            "gold_answer",
            "task_name"
        }
        dfg_sub = keep_only_columns(dfg_sub, keep_cols_dfg)
    else:
        dfg_sub = Dataset.from_dict({})

    # We have to restore the original column name for the process tree subset
    # we can just reload from the original ds param
    ds2 = ds
    if not skip_pt:
        pt_sub = ds2.rename_column("pt", "gold_answer")
        pt_sub = pt_sub.map(lambda x: {"task_name": "process_tree"})
        keep_cols_pt = {
            "model_id", "unique_activities", 
            "gold_answer",
            "task_name"
        }
        pt_sub = keep_only_columns(pt_sub, keep_cols_pt)
    else:
        pt_sub = Dataset.from_dict({})

    # Concatenate them
    if len(dfg_sub) == 0 and len(pt_sub) == 0:
        discovery_combined = Dataset.from_dict({})  # empty
    elif len(dfg_sub) == 0:
        discovery_combined = pt_sub
    elif len(pt_sub) == 0:
        discovery_combined = dfg_sub
    else:
        discovery_combined = concatenate_datasets([dfg_sub, pt_sub])

    return discovery_combined

def build_combined_train_dataset(
    trace_train_ds, 
    pairs_train_ds, 
    na_train_ds, 
    discovery_train_ds,
    skip_trace=False,
    skip_pairs=False,
    skip_na=False,
    skip_dfg=False,
    skip_pt=False
):
    """
    Combine the four train datasets into one, 
    while:
      - potentially skipping certain tasks (leave-one-out)
      - keeping relevant columns (trace/eventually_follows/prefix and model_id/unique_activities).
      - adding task_name + gold_answer columns.
    """
    trace_prepped = prepare_trace_ds(trace_train_ds, skip=skip_trace)
    pairs_prepped = prepare_pairs_ds(pairs_train_ds, skip=skip_pairs)
    na_prepped = prepare_na_ds(na_train_ds, skip=skip_na)
    disc_prepped = prepare_discovery_ds(discovery_train_ds, skip_dfg=skip_dfg, skip_pt=skip_pt)

    # Concatenate all
    subsets = []
    if len(trace_prepped) > 0:
        subsets.append(trace_prepped)
    if len(pairs_prepped) > 0:
        subsets.append(pairs_prepped)
    if len(na_prepped) > 0:
        subsets.append(na_prepped)
    if len(disc_prepped) > 0:
        subsets.append(disc_prepped)

    if len(subsets) == 0:
        return Dataset.from_dict({})  # return empty if everything is skipped

    combined = concatenate_datasets(subsets)
    return combined


#-------------------------- Tokenization ---------------------------------
def get_formatted_prompt(example, template_text):
    """
    Replace placeholder values in the prompt with real data.
    """
    # here we either extract the real key, or if the key is not relevant for the current task, substitute it with empty value
    eventually_follows = example.get("eventually_follows", ["", ""]) # if no key exists - return two empty elements to substitute a pair of activities
    eventually_follows = eventually_follows if isinstance(eventually_follows, (list, tuple)) else ["", ""]
    placeholder_values = {
        "activities": str(set(example["unique_activities"])), # unique_activities is always relevant
        "act1": eventually_follows[0], 
        "act2": eventually_follows[1],
        "trace": str(example.get("trace", [])),
        "prefix": str(example.get("prefix", [])),
        "next_act": example.get("gold_answer", ""), #  in S-NAP, gold answer is the next activity
        "wrong_next_act": example.get("wrong_next_act", ""),
        "reduced_prefix": str(example.get("reduced_prefix", "")),
        "extended_prefix": str(example.get("extended_prefix", "")),
        "reduced_dfg": str(example.get("reduced_dfg", "")),
        "extended_dfg": str(example.get("extended_dfg", "")),
    }

    placeholders_in_template = re.findall(r"{(.*?)}", template_text)
    filtered_format_dict = { key: placeholder_values.get(key, "") for key in placeholders_in_template }
    
    try:
        return template_text.format(**filtered_format_dict)
    except KeyError as e:
        raise ValueError(f"Missing placeholder variable in format: {e}")


def get_formatted_answer(example, template_text):
    """
    Replace placeholder values in the answer with real data.
    """
    eventually_follows = example.get("eventually_follows", ["", ""]) # if no key exists - return two empty elements to substitute a pair of activities
    eventually_follows = eventually_follows if isinstance(eventually_follows, (list, tuple)) else ["", ""]
    placeholder_values = {
        "gold_answer": str(example.get("gold_answer", "")),
        "prefix": str(example.get("prefix", [])),
        "eventually_follows": str(example.get("eventually_follows", "")),
        "trace": str(example.get("trace", [])),
        "act_from_reduced_prefix": str(example.get("act_from_reduced_prefix", "")),
        "act_from_extended_prefix": str(example.get("act_from_extended_prefix", "")),
        "pair_from_reduced_dfg": str(example.get("pair_from_reduced_dfg", "")),
        "pair_from_extended_dfg": str(example.get("pair_from_extended_dfg", "")),
        "act1": eventually_follows[0], 
        "act2": eventually_follows[1],
    }
    
    placeholders_in_template = re.findall(r"{(.*?)}", template_text)
    filtered_format_dict = { key: placeholder_values.get(key, "") for key in placeholders_in_template }
    
    try:
        return template_text.format(**filtered_format_dict)
    except KeyError as e:
        raise ValueError(f"Missing placeholder variable in format: {e}")
    

def generate_inverted_answers(example, task_name):
    unique_activities = list(example["unique_activities"])

    if task_name == "next_activity":
        core_prefix = example["prefix"]  # activities before next
        next_activity = example["gold_answer"]

        # ---------------- Reduced Prefix ----------------
        # Remove one activity from core_prefix (never remove next_activity)
        idx = random.randrange(len(core_prefix))
        removed_act = core_prefix[idx]
        reduced_core = core_prefix[:idx] + core_prefix[idx+1:]

        reduced_prefix = reduced_core + [next_activity]
        example["reduced_prefix"] = reduced_prefix
        example["act_from_reduced_prefix"] = removed_act

        # ---------------- Extended Prefix ----------------
        # Add one activity into core_prefix at random position
        added_act = random.choice(unique_activities)
        insert_idx = random.randint(0, len(core_prefix))
        extended_core = core_prefix[:insert_idx] + [added_act] + core_prefix[insert_idx:]
        extended_prefix = extended_core + [next_activity]

        example["extended_prefix"] = extended_prefix
        example["act_from_extended_prefix"] = added_act

        # ---------------- Wrong Next Activity ----------------
        possible_wrong = [act for act in unique_activities if act != next_activity]
        example["wrong_next_act"] = random.choice(possible_wrong)

    elif task_name == "dfg":
        # reduced_dfg: randomly remove one tuple from the list "dfg"
        dfg = ast.literal_eval(example["gold_answer"])
        idx = random.randrange(len(dfg))
        removed_pair = dfg[idx]
        reduced_dfg = dfg[:idx] + dfg[idx+1:]
        example["reduced_dfg"] = reduced_dfg
        example["pair_from_reduced_dfg"] = removed_pair
    
        # extended_dfg: randomly add one tuple with two activities from unique_activities to dfg,
        # but the same pair must not exist already.
        possible_pairs = []
        random_idx = random.randint(0, len(dfg))
        for a in unique_activities:
            for b in unique_activities:
                pair = (a, b)
                if pair not in dfg:
                    possible_pairs.append(pair)
        if possible_pairs:
            added_pair = random.choice(possible_pairs)
            extended_dfg = dfg[:random_idx] + [added_pair] + dfg[random_idx:]
            example["extended_dfg"] = extended_dfg
            example["pair_from_extended_dfg"] = added_pair
        else:
            dummy_pair = ("dummy1","dummy2")
            example["extended_dfg"] = dfg[:random_idx] + [dummy_pair] + dfg[random_idx:]
            example["pair_from_extended_dfg"] = dummy_pair
            print("All pairs exist")
    
    return example


def build_instruction_encoding(example, tokenizer, excluded_task, max_length=1024):
    """
    Build an instruction (prompt) + gold answer for a single example,
    then return {input_ids, attention_mask, labels} for causal LM training.
    
    Required fields in `example`:
      - task_name         : one of ["trace_anomaly", "activity_anomaly", "next_activity", "dfg", "process_tree"]
      - unique_activities : list or set of strings
      - other fields needed by that specific task, such as "trace", "eventually_follows", "prefix"
      - gold_answer       : the ground-truth string to generate (e.g. "True", "False", "myActivity", "[(a,b),(b,c)] [END]", etc.)
    
    Returns a dict with "input_ids", "attention_mask", "labels".
    """
    prompt_module_name = f"prompts_per_cluster.prompts_{excluded_task}_excluded"
    prompts = importlib.import_module(prompt_module_name)

    task_name = example["task_name"]
    gold_answer = example["gold_answer"]   

    possible_templates = prompts.TASK_PROMPTS_VARIANTS[task_name]
    is_inverted_template = False
    
    # randomly choose a prompt variation based on probabilities as described in the thesis
    if task_name == "trace_anomaly" or task_name == "activity_anomaly":
        if gold_answer == "False":
            inverted_templates = prompts.TASK_PROMPTS_INVERTED_NEGATIVE.get(task_name)
            if inverted_templates and random.random() < 0.2: # handling cases for validation sets
                possible_templates = inverted_templates
                is_inverted_template = True
        elif gold_answer == "True":
            inverted_templates = prompts.TASK_PROMPTS_INVERTED_POSITIVE.get(task_name)
            if inverted_templates and random.random() < 0.2:
                possible_templates = inverted_templates
                is_inverted_template = True
    elif task_name == "next_activity":
        inverted_templates = prompts.TASK_PROMPTS_INVERTED_NEGATIVE.get(task_name) # only for anomaly-excluded
        if inverted_templates and random.random() < 0.2:
            possible_templates = inverted_templates
            is_inverted_template = True
            example = generate_inverted_answers(example, task_name)
        else:
            inverted_templates = prompts.TASK_PROMPTS_INVERTED_POSITIVE.get(task_name)
            if inverted_templates and random.random() < 0.2:
                possible_templates = inverted_templates
                is_inverted_template = True
                example = generate_inverted_answers(example, task_name)
    elif task_name == "dfg": # only for anomaly-excluded
        inverted_templates = prompts.TASK_PROMPTS_INVERTED_NEGATIVE.get(task_name)
        if inverted_templates and random.random() < 0.2:
            possible_templates = inverted_templates
            is_inverted_template = True
            example = generate_inverted_answers(example, task_name)

    template_text = random.choice(possible_templates)

    # format the prompt with correct values dynamically based on placeholders
    prompt_text = get_formatted_prompt(example, template_text["template"])
    prompt_text = prompts.GENERAL_INTRO + "\n\n" + prompt_text

    # The answer we want the model to generate
    # e.g. "True", "False", "activity_name", "[(a,b), (b,c)] [END]", or the entire process tree, etc.
    answer_text = get_formatted_answer(example, template_text["answer"])
    answer_text = answer_text + (" [END]" if task_name in {"dfg", "process_tree"} and not is_inverted_template else "") + tokenizer.eos_token

    # print('Prompt: "', prompt_text, '"')
    # print('\nAnswer: "', answer_text, '"')
    example["prompt_text"] = prompt_text
    example["answer_text"] = answer_text

    # Tokenize the prompt and the answer
    enc_prompt = tokenizer(prompt_text, add_special_tokens=False)
    enc_answer = tokenizer(answer_text, add_special_tokens=False)

    # Combine the prompt and the answer
    input_ids = enc_prompt["input_ids"] + enc_answer["input_ids"]
    attention_mask = enc_prompt["attention_mask"] + enc_answer["attention_mask"]

    # Build labels with prompt being masked (repeated instructions are ignored for model's loss calculation as it's not expected to generate instructions - for faster convergence):
    #   - Prompt => -100
    #   - Answer => actual token IDs
    labels = [-100] * len(enc_prompt["input_ids"]) + enc_answer["input_ids"]
    # labels = enc_prompt["input_ids"] + enc_answer["input_ids"] # for experiments with no prompt masking

    # Truncate if needed
    truncated = len(input_ids) > max_length
    if truncated:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "truncated": truncated,
        "prompt_text": prompt_text,
        "answer_text": answer_text
    }

def instruction_map_fn(examples, tokenizer, excluded_task):
    # Create lists for logging inputs
    prompts_list = []
    answers_list = []
    task_names_list = []

    output_dir_inputs = f"inputs_log/{excluded_task}/"
    os.makedirs(output_dir_inputs, exist_ok=True)

    results = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "truncated": []
    }
    for i in range(len(examples["task_name"])):
        example = {
            col_name: examples[col_name][i] 
            for col_name in examples
        }

        out = build_instruction_encoding(example, tokenizer, excluded_task)

        prompts_list.append(out["prompt_text"])
        answers_list.append(out["answer_text"])
        task_names_list.append(example["task_name"])

        results["input_ids"].append(out["input_ids"])
        results["attention_mask"].append(out["attention_mask"])
        results["labels"].append(out["labels"])
        results["truncated"].append(out["truncated"])

    # Save final training data to csv files in batches
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Current timestamp
    df = pd.DataFrame({"task_name": task_names_list, "prompt": prompts_list, "answer": answers_list})
    df.to_csv(f"{output_dir_inputs}prompts_and_answers_{timestamp}.csv", index=False)
    return results

#---------------- data collator---------------------------------------
def causal_data_collator(features, tokenizer):
    """
    Pads input_ids, attention_mask, and labels to the same length across the batch.
    We pad prompt tokens with -100 for labels so they're ignored in the loss.
    """
    # Separate out each field
    batch_input_ids = [f["input_ids"] for f in features]
    batch_attention = [f["attention_mask"] for f in features]
    batch_labels    = [f["labels"] for f in features]  # each is a list of ints
    
    # Let HF pad the inputs & attention mask
    padded = tokenizer.pad(
        {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention
        },
        padding=True,  # True => pad to the longest sequence in this batch
        return_tensors="pt",
    )
    
    # Now pad labels to the same sequence length
    max_length = padded["input_ids"].size(1)
    padded_labels = []
    for lbl in batch_labels:
        # pad up to max_length
        num_to_pad = max_length - len(lbl)
        padded_labels.append(lbl + [-100] * num_to_pad)
    
    padded["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    
    return padded