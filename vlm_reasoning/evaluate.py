import csv
from functools import partial
import os
import re
from PIL import Image
from model_wrapper import ReasonVLM
from transformers import AutoProcessor

from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

from model_wrapper import collate_fn as collate_fn_base


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
    is_lora = False
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
        base_lm.load_adapter(pretrained_path, adapter_name="default")
    accelerator = Accelerator()

    collate_fn = partial(collate_fn_base, processor=processor, tokenizer=processor.tokenizer)  # TODO: remove tokenizer from next commit

    batch_size = 1  # Adjust batch size as needed
    dataloader = DataLoader(processed_dataset, batch_size=batch_size, collate_fn=collate_fn)
    model, dataloader = accelerator.prepare(model, dataloader)
    collected_loss = []
    unwrapped_model = accelerator.unwrap_model(model)
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            unwrapped_model.reset_image_reasoning()
            image_sizes = batch.pop("image_sizes", None)
            input_ids_answers = batch.pop("input_ids_answers", None)

            # ---------- First Forward Pass ----------
            output = model.forward(**batch, image_sizes=image_sizes, output_hidden_states=True)
            image_reasoning = output.hidden_states[-1][:, -1, :]
            unwrapped_model.set_image_reasoning(image_reasoning)

            if is_lora:
                model.model.language_model.disable_adapter_layers()
            output2 = model.forward(
                **batch,
                input_ids_answers=input_ids_answers,
                image_sizes=image_sizes,
            )
            loss = output2.loss
            if is_lora:
                model.model.language_model.enable_adapter_layers()
            collected_loss.append(loss.item())
    print(f"Average Loss: {sum(collected_loss) / len(collected_loss)}")


if __name__ == "__main__":
    main()
