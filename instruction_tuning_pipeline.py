from unsloth import FastLanguageModel
from functools import partial
from trl import SFTTrainer
from transformers import TrainingArguments
from data_loading_util import load_trace_data, load_pairs_data, load_next_activity_data, load_discovery_data
import datetime
from instruction_tuning_util import causal_data_collator, build_combined_train_dataset, instruction_map_fn, prepare_trace_ds, prepare_pairs_ds, prepare_na_ds, prepare_discovery_ds
import os
from transformers import TrainerCallback
from datasets import concatenate_datasets
import torch

# ------------------ Configuration ------------------------
max_seq_length = 1024
dtype = torch.bfloat16
load_in_4bit = True

# base models
model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
# model_name = "unsloth/Mistral-Large-Instruct-2407-bnb-4bit"

label_model_name = model_name.replace("/", "_") # replace "/" for file saving

# Task selection: Choose one of ["prediction", "anomaly", "discovery"]
excluded_task = "anomaly"

# training details
train_cap = 30000
valid_cap = 500
epochs = 3
learning_rate = 1e-5
train_batch_size = 8
train_accum_steps = 4
eval_batch_size = 16
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_steps = 20
eval_save_steps = 250

save_dir = (f"{label_model_name}_{excluded_task}_samples-{train_cap}_epochs-{epochs}_"
            f"lr-{learning_rate}_batch-{train_batch_size}x{train_accum_steps}_time-{timestamp}")

# --------------- Custom Checkpoint Callback ---------------------
class CustomCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")        
        custom_name = f"{args.output_dir}/{save_dir}_step-{state.global_step}"
        
        if os.path.exists(checkpoint_dir):
            os.rename(checkpoint_dir, custom_name)

# ---------------- Load Model & Tokenizer --------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ------------------ Load datasets ----------------------
trace_train_ds, trace_val_ds, _ = load_trace_data()
pairs_train_ds, pairs_val_ds, _ = load_pairs_data()
na_train_ds, na_val_ds, _ = load_next_activity_data()
discovery_train_ds, discovery_val_ds, discovery_test_ds = load_discovery_data()

# Shuffle training datasets
trace_train_ds = trace_train_ds.shuffle(seed=3407)
pairs_train_ds = pairs_train_ds.shuffle(seed=3407)
na_train_ds = na_train_ds.shuffle(seed=3407) 
discovery_train_ds = concatenate_datasets([discovery_train_ds, discovery_val_ds, discovery_test_ds]).shuffle(seed=3407) # we concatenate splits to use all data in this cluster as it's smaller than the cap

# Cut the training sets
trace_train_ds = trace_train_ds.select(range(min(train_cap, len(trace_train_ds))))
pairs_train_ds = pairs_train_ds.select(range(min(train_cap, len(pairs_train_ds))))

if excluded_task == "discovery":
    na_train_ds = na_train_ds.select(range(min(train_cap * 2, len(na_train_ds)))) # multiplied by 2 as there's only one task type in prediction cluster; in discovery-excluded training twice the cap amount must be used for sample balancing
else:
    na_train_ds = na_train_ds.select(range(min(train_cap, len(na_train_ds))))

discovery_train_ds = discovery_train_ds.select(range(min(train_cap, len(discovery_train_ds))))

# Select training and valid data based on excluded task
if excluded_task == "prediction":  # Exclude prediction (next_activity is used for validation)    
    combined_train_ds = build_combined_train_dataset(
        trace_train_ds=trace_train_ds,
        pairs_train_ds=pairs_train_ds,
        na_train_ds=None,
        discovery_train_ds=discovery_train_ds,
        skip_na=True
    ) 
    na_val_ds = na_val_ds.shuffle(seed=3407)   
    val_ds = prepare_na_ds(na_val_ds.select(range(valid_cap)))  # Validate on next_activity

elif excluded_task == "anomaly":  # Exclude anomaly (trace & pairs are used for validation)    
    combined_train_ds = build_combined_train_dataset(
        trace_train_ds=None,
        pairs_train_ds=None,
        na_train_ds=na_train_ds,
        discovery_train_ds=discovery_train_ds,
        skip_trace=True,
        skip_pairs=True
    )
    trace_val_ds = trace_val_ds.shuffle(seed=3407)
    pairs_val_ds = pairs_val_ds.shuffle(seed=3407)
    val_ds = concatenate_datasets([
        prepare_trace_ds(trace_val_ds.select(range(valid_cap // 2))),
        prepare_pairs_ds(pairs_val_ds.select(range(valid_cap // 2)))
    ])

elif excluded_task == "discovery":  # Exclude discovery (discovery is used for validation)    
    combined_train_ds = build_combined_train_dataset(
        trace_train_ds=trace_train_ds,
        pairs_train_ds=pairs_train_ds,
        na_train_ds=na_train_ds,
        discovery_train_ds=None,
        skip_dfg=True,
        skip_pt=True 
    )
    discovery_val_ds = discovery_val_ds.shuffle(seed=3407)
    val_ds = prepare_discovery_ds(discovery_val_ds.select(range(valid_cap)))  # Validate on discovery

# Shuffle final datasets
combined_train_ds = combined_train_ds.shuffle(seed=3407)
val_ds = val_ds.shuffle(seed=3407)

# ------------------ Tokenization -------------------------
map_func = partial(instruction_map_fn, tokenizer=tokenizer, excluded_task=excluded_task)
combined_train_ds = combined_train_ds.map(map_func, batched=True)
val_ds = val_ds.map(map_func, batched=True)

# Filter truncated rows
combined_train_ds = combined_train_ds.filter(lambda example: not example["truncated"])

# Data Collation
def my_collator(features):
    return causal_data_collator(features, tokenizer)

# ---------------- Training Arguments --------------------------
training_args = TrainingArguments(
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps=train_accum_steps,
    num_train_epochs=epochs,
    learning_rate=learning_rate,
    bf16=True,
    fp16=False,
    logging_steps=log_steps,
    optim="adamw_8bit",
    weight_decay=0.01,
    seed=3407,
    output_dir="outputs",
    report_to="none",
    lr_scheduler_type="linear",
    # per_device_eval_batch_size=eval_batch_size,
    # eval_strategy="steps",
    # eval_steps=eval_save_steps,
    save_strategy="steps",
    save_steps=eval_save_steps,
)

# ------------------- Trainer Setup and Initialization -----------------------
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=combined_train_ds,
    # eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=my_collator,
)

trainer.add_callback(CustomCheckpointCallback())

trainer.train()
# trainer.train(resume_from_checkpoint='') # in case training must be resumed from a different checkpoint