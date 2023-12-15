# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

from trl import SFTTrainer
from trl.import_utils import is_xpu_available
from trl.trainer import ConstantLengthDataset

os.environ["WANDB_PROJECT"] = "CS546" # name your W&B project 

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="mistralai/Mistral-7B-v0.1", metadata={"help": "the model name"})

    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    data_files: Optional[str] = field(default="datasets/train_prm800k.jsonl", metadata={"help": "the data files"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "whether to use gradient checkpointing"})
    max_steps: Optional[int] = field(default=5000, metadata={"help": "max number of training steps"})
    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "optimizer learning rate"})
    warmup_steps: Optional[int] = field(default=500, metadata={"help": "the number of warmup steps"})
    

    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    )




script_args = tyro.cli(ScriptArguments)

if script_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")

# `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
# `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
if script_args.gradient_checkpointing:
    raise ValueError("gradient_checkpointing not supported")


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Given a math word problem, solve the problem while showing the steps.\n### Question: {example['question'].strip()}\n ### Solution: {example['chosen'].strip()}"
    return text

def prepare_classification_prompt(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Given a math word problem and a solution, determine whether the solution is correct or wrong.\n### Question: {example['question']}\n ### Solution: {example['solution']}\n ### Answer: {example['answer']}"
    return text


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        "json",
        data_files=args.data_files,
        split = 'train',
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map={"": Accelerator().local_process_index},
    trust_remote_code=True,
    token = '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',
    cache_dir = '/shared/data2/tanayd2/hf'
)
base_model.config.use_cache = False

peft_config = script_args.peft_config

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# evaluate 10 times in all so dervie save_steps and eval_steps from max_steps
save_steps = script_args.max_steps // 15
eval_steps = script_args.max_steps // 15

training_args = TrainingArguments(
            output_dir= script_args.output_dir,
            max_steps=script_args.max_steps,
            logging_steps=5,
            save_steps=save_steps,
            evaluation_strategy="steps", # Evaluate the model every logging step
            eval_steps=eval_steps,   
            save_total_limit=2,     # Only keep one checkpoint
            per_device_train_batch_size=script_args.per_device_train_batch_size,
            per_device_eval_batch_size=script_args.per_device_eval_batch_size,
            gradient_accumulation_steps=2, # set in accelerate config
            gradient_checkpointing=script_args.gradient_checkpointing,
            group_by_length=script_args.group_by_length,
            learning_rate=script_args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=script_args.warmup_steps,
            weight_decay=0.05,
            max_grad_norm = 2, # set in accelerate config
            optim="paged_adamw_32bit",
            bf16=True,
            remove_unused_columns=False,
            report_to="wandb",
            run_name="sft_llama2",
        )

train_dataset, eval_dataset = create_datasets(tokenizer, script_args)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=script_args.packing,
    max_seq_length=None, # since passing constant length dataset
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
trainer.save_model(script_args.output_dir)

output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
if is_xpu_available():
    torch.xpu.empty_cache()
else:
    torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)