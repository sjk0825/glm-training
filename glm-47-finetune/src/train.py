import os
import sys
import yaml
import torch
import wandb
from dataclasses import dataclass, field
from typing import Optional

from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, load_dataset
from transformers import TrainingArguments


@dataclass
class ModelConfig:
    model_name: str = "THUDM/GLM-4-7b-chat"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: Optional[torch.dtype] = None


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list = field(default_factory=lambda: [
        "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainConfig:
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    max_grad_norm: float = 1.0
    seed: int = 42
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = False
    label_names: list = field(default_factory=lambda: ["labels"])


@dataclass
class WandbConfig:
    project: str = "glm-4.7-finetune"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: list = field(default_factory=list)
    notes: str = ""


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def format_conversation(prompt: str, response: str) -> str:
    return f"""<|user|>
{prompt}
<|assistant|>
{response}"""


def create_dataset_from_texts(texts: list, batch_size: int = 10) -> Dataset:
    formatted_texts = []
    for i in range(0, len(texts), 2):
        if i + 1 < len(texts):
            formatted = format_conversation(texts[i], texts[i + 1])
            formatted_texts.append(formatted)
    
    return Dataset.from_dict({"text": formatted_texts})


def prepare_sample_data() -> Dataset:
    sample_data = [
        "Write a Python function to calculate factorial.",
        """def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result""",
        
        "How do I sort a list in Python?",
        """# Using sorted() - returns new sorted list
sorted_list = sorted([3, 1, 4, 1, 5, 9, 2, 6])
# Using sort() - sorts in place
my_list = [3, 1, 4, 1, 5, 9, 2, 6]
my_list.sort()

# With custom key
my_list.sort(key=lambda x: -x)  # descending order""",
        
        "Explain what is a closure in programming.",
        """A closure is a function that remembers variables from its outer scope even after the outer function has finished executing.

Example in Python:
def outer(x):
    def inner():
        print(x)  # x is captured from outer scope
    return inner

closure = outer(10)
closure()  # Prints: 10""",
        
        "Write a function to reverse a string.",
        """def reverse_string(s):
    return s[::-1]

# Alternative:
def reverse_string_manual(s):
    result = ""
    for char in s:
        result = char + result
    return result""",
        
        "What is the difference between list and tuple?",
        """- List: mutable (can be modified), slower, uses more memory
  my_list = [1, 2, 3]
  my_list.append(4)  # OK

- Tuple: immutable (cannot be modified), faster, less memory
  my_tuple = (1, 2, 3)
  my_tuple.append(4)  # Error!""",
        
        "How do I read a file in Python?",
        """# Read entire file
with open('file.txt', 'r') as f:
    content = f.read()

# Read line by line
with open('file.txt', 'r') as f:
    for line in f:
        print(line.strip())

# Write to file
with open('output.txt', 'w') as f:
    f.write('Hello, World!')""",
        
        "Write a Python class for a Bank Account.",
        """class BankAccount:
    def __init__(self, initial_balance=0):
        self.balance = initial_balance
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            return True
        return False
    
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            return True
        return False
    
    def get_balance(self):
        return self.balance""",
        
        "What is list comprehension?",
        """# List comprehension creates lists in a concise way

# Basic
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# Nested
matrix = [[i*3+j for j in range(3)] for i in range(3)]""",
        
        "How do I handle exceptions in Python?",
        """try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"Error: {e}")
else:
    print("Success!")
finally:
    print("Always executes")""",
    ]
    
    return create_dataset_from_texts(sample_data)


def load_training_data(config: dict) -> Dataset:
    if "dataset_name" in config and config["dataset_name"]:
        dataset = load_dataset(
            config["dataset_name"],
            name=config.get("dataset_config", None),
            split=config.get("dataset_split", "train")
        )
        if "text_field" in config:
            dataset = dataset.map(
                lambda x: {"text": x[config["text_field"]]},
                remove_columns=dataset.column_names
            )
        return dataset
    else:
        print("Using sample dataset...")
        return prepare_sample_data()


def setup_model(model_config: ModelConfig, lora_config: LoRAConfig):
    print(f"Loading model: {model_config.model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.model_name,
        max_seq_length=model_config.max_seq_length,
        load_in_4bit=model_config.load_in_4bit,
        dtype=model_config.dtype,
        trust_remote_code=True,
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Setting up LoRA...")
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def setup_wandb(wandb_config: WandbConfig, config: dict):
    wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=wandb_config.name,
        tags=wandb_config.tags,
        notes=wandb_config.notes,
        config=config
    )


def train(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    train_config: TrainConfig,
    wandb_config: WandbConfig,
    config: dict
):
    setup_wandb(wandb_config, config)
    
    model, tokenizer = setup_model(model_config, lora_config)
    dataset = load_training_data(config.get("data", {}))
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Sample: {dataset[0]['text'][:200]}...")
    
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_train_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        warmup_ratio=train_config.warmup_ratio,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        max_grad_norm=train_config.max_grad_norm,
        seed=train_config.seed,
        dataloader_num_workers=train_config.dataloader_num_workers,
        remove_unused_columns=train_config.remove_unused_columns,
        label_names=train_config.label_names,
        report_to=["wandb"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=model_config.max_seq_length,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print("\nSaving model...")
    trainer.save_model(f"{train_config.output_dir}/final")
    tokenizer.save_pretrained(f"{train_config.output_dir}/final")
    
    wandb.finish()
    print("Training complete!")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/train.yaml"
    
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        print(f"Config file not found: {config_path}")
        print("Using default configuration...")
        config = {
            "model": {
                "model_name": "THUDM/GLM-4-7b-chat",
                "max_seq_length": 2048,
                "load_in_4bit": True
            },
            "lora": {
                "r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.0
            },
            "train": {
                "output_dir": "./outputs",
                "num_train_epochs": 3,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2e-4
            },
            "wandb": {
                "project": "glm-4.7-finetune"
            }
        }
    
    model_config = ModelConfig(**config.get("model", {}))
    lora_config = LoRAConfig(**config.get("lora", {}))
    train_config = TrainConfig(**config.get("train", {}))
    wandb_config = WandbConfig(**config.get("wandb", {}))
    
    print("=" * 50)
    print("GLM-4.7 Fine-tuning Configuration")
    print("=" * 50)
    print(f"Model: {model_config.model_name}")
    print(f"Max Seq Length: {model_config.max_seq_length}")
    print(f"Load in 4-bit: {model_config.load_in_4bit}")
    print(f"LoRA Rank: {lora_config.r}")
    print(f"Batch Size: {train_config.per_device_train_batch_size}")
    print(f"Learning Rate: {train_config.learning_rate}")
    print(f"W&B Project: {wandb_config.project}")
    print("=" * 50)
    
    train(model_config, lora_config, train_config, wandb_config, config)


if __name__ == "__main__":
    main()
