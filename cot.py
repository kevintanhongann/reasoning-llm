import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from typing import List, Dict

class ChainOfThoughtTrainer:
    def __init__(self, model_name: str = "gpt2-medium", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def prepare_dataset(self, examples: List[Dict[str, str]]) -> Dataset:
        """Prepare the dataset for training."""
        def tokenize_function(example):
            prompt = f"Question: {example['question']}\nLet's solve this step by step:\n{example['reasoning']}\nFinal answer: {example['answer']}"
            return self.tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

        dataset = Dataset.from_list(examples)
        tokenized_dataset = dataset.map(tokenize_function)
        tokenized_dataset = tokenized_dataset.remove_columns(['question', 'reasoning', 'answer'])
        tokenized_dataset = tokenized_dataset.rename_column("input_ids", "labels")
        return tokenized_dataset

    def train(self, dataset: Dataset, output_dir: str, num_train_epochs: int = 3, per_device_train_batch_size: int = 8):
        """Train the model using the prepared dataset."""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    # Example usage
    cot_trainer = ChainOfThoughtTrainer()

    # Prepare example data
    examples = [
        {
            "question": "What is 12345 + 54321?",
            "reasoning": "Step 1: Align the numbers vertically.\n12345\n54321\nStep 2: Add the digits in each column, starting from the right.\n5 + 1 = 6\n4 + 2 = 6\n3 + 3 = 6\n2 + 4 = 6\n1 + 5 = 6\nStep 3: Combine the results.\n",
            "answer": "66666"
        },
        # Add more examples here
    ]

    # Prepare and train the model
    dataset = cot_trainer.prepare_dataset(examples)
    cot_trainer.train(dataset, output_dir="./cot_model")

    print("Training complete. Model saved in ./cot_model")
