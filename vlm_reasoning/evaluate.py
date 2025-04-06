import csv
import os
import re
from PIL import Image
from model_wrapper import ReasonVLM, apply_prompt
from transformers import LlavaNextForConditionalGeneration, AutoProcessor, PreTrainedModel, GenerationMixin, LlavaNextForConditionalGeneration, AutoConfig
from datasets import load_dataset
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import argparse
import os
import wandb
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
import itertools


def preprocess_hri():
    # Path to your CSV file and frames folder
    csv_path = 'data.csv'
    frames_dir = 'frames'

    # List to store the processed rows
    processed_rows = []

    # Open and read the CSV file
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            clip_uid = row['clip_uid']

            # Build the image file path (only take the image with suffix "_0.png")
            image_path = os.path.join(frames_dir, f"{clip_uid}_0.png")

            # Load the image using PIL
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image '{image_path}': {e}")
                continue

            # Process the summary text:
            summary = row['summary']
            # Remove the leading "#Summary" if it exists and strip extra spaces.
            if summary.startswith("#Summary"):
                summary = summary[len("#Summary"):].strip()

            # Replace standalone "C" with "human" using regex word boundaries
            summary = re.sub(r'\bC\b', 'human', summary)

            # Create the row with image, question, and answers list (containing the processed summary)
            processed_row = {
                "image": image,
                "question": "What are the human's intentions?",
                "answers": [summary]
            }
            processed_rows.append(processed_row)

    # processed_rows now contains your preprocessed dataset.
    print(f"Processed {len(processed_rows)} rows.")
    return processed_rows


def main():
    checkpoint_path = "<visual_reasoner_checkpoint_path>"
    pretrained_path = "<pretrained_model_path>"
    processed_dataset = preprocess_hri()
    model = ReasonVLM.from_pretrained("llava-hf/llama3-llava-next-8b-hf", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf", trust_remote_code=True, padding_side="left")
    model.visual_reasoner.load_state_dict(torch.load(checkpoint_path))
    if os.path.exists(pretrained_path):
        print("Loading pretrained model from:", pretrained_path)
        is_lora = True
        base_lm = model.model.language_model

        # 2. Inject LoRA layers into base_lm.
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        base_lm = get_peft_model(base_lm, lora_config)
        model.model.language_model = base_lm
        base_lm.load_adapter(pretrained_path, adapter_name="dd")
        base_lm.set_adapter("dd")
    accelerator = Accelerator()
    model.visual_reasoner.to(accelerator.device)
    model.model.language_model.to(accelerator.device)
    def collate_fn(batch):
        texts = [apply_prompt(item) for item in batch]
        images = [item["image"] for item in batch]

        # this is for the first forward pass, asks the model to reason on the image
        prompts = processor.apply_chat_template(texts, add_generation_prompt=True)
        inputs = processor(text=prompts, images=images, return_tensors="pt", do_pad=True, padding=True)
        # this is for the second forward pass, asks the model to generate the answer
        inputs_answers = [apply_prompt(item, wants_answer=True) for item in batch]
        inputs_answers = processor.apply_chat_template(inputs_answers)
        inputs_answers = processor(text=inputs_answers, images=images, return_tensors="pt", do_pad=True, padding=True)
        inputs_answers = inputs_answers["input_ids"]
        labels = [apply_prompt(item, wants_answer=True, is_label=True) for item in batch]
        labels = processor.apply_chat_template(labels)
        labels = processor(text=labels, images=images, return_tensors="pt", do_pad=True, padding=True)
        labels = labels["input_ids"]

        # Mask the input part in labels using inputs_answers
        # For each sequence, set label tokens to -100 where they match the input
        for i in range(len(labels)):
            # Find the actual length by counting non-padding tokens
            input_len = (inputs_answers[i] != processor.tokenizer.pad_token_id).sum().item()
            # Only keep labels after the input sequence by setting -100 for input tokens
            labels[i][:input_len] = -100
        # now pad inputs_answers to the same length as labels
        max_length = max([len(label) for label in labels])
        for i in range(len(inputs_answers)):
            inputs_answers = nn.functional.pad(inputs_answers, (0, max_length - inputs_answers.shape[1]), value=processor.tokenizer.pad_token_id)

        inputs["labels"] = labels
        inputs["input_ids_answers"] = inputs_answers
        return inputs

    batch_size = 1  # Adjust batch size as needed
    dataloader = DataLoader(processed_dataset, batch_size=batch_size, collate_fn=collate_fn)
    collected_loss = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            model.reset_image_reasoning()
            # Move all tensor inputs to the target device.
            image_sizes = batch.pop("image_sizes", None)
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            labels = batch.pop("labels", None)
            input_ids_answers = batch.pop("input_ids_answers", None)
            with torch.no_grad():
                output = model.forward(**batch, image_sizes=image_sizes, output_hidden_states=True)
                # Get the logits of the last token, then pass it as image reasoning hint.
                image_reasoning = output.hidden_states[-1][:, -1, :]
            model.set_image_reasoning(image_reasoning)

            if is_lora:
                model.model.language_model.disable_adapter_layers()
            loss = model.forward(input_ids=input_ids_answers, labels=labels, pixel_values=batch["pixel_values"], image_sizes=image_sizes).loss
            if is_lora:
                model.model.language_model.enable_adapter_layers()
            collected_loss.append(loss.item())
            # print(f"Loss: {loss.item()}")
    print(f"Average Loss: {sum(collected_loss) / len(collected_loss)}")


if __name__ == "__main__":
    main()
