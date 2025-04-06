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
from peft import LoraConfig, get_peft_model
import itertools


def apply_prompt(example, is_label=False, wants_answer=False):
    assert "image" in example, "The example must have an 'image' field"
    assert not isinstance(example["image"], list), "Cannot give more than 1 image"
    assert "question" in example, "The example must have a 'question' field"
    if not wants_answer:
        base_prompt = "Analyse the image and reason on the user's question. In particular, reflect on how the image you are fed could be improved to facilitate answering the question. Your goal is NOT to reply to the question but to help a separate system do that. Begin by describing the image, then think where a person would look and give attention to. Finally imagine improving the image for your purposes. QUESTION:\n"
    else:
        base_prompt = ""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": base_prompt + example["question"]},
            ],
        },
    ]
    if is_label:
        answers = example["answers"]
        # answers is a list of 10 possible answers, get the most common one
        answer = max(set(answers), key=answers.count)
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            }
        )
    return messages


class VisualReasoner(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(VisualReasoner, self).__init__()
        hidden_dim = in_dim * 2  # Wider hidden dimension for better representation
        self.visual_reasoner = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),  # GELU activation is commonly used in modern architectures
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1)  # Add dropout for regularization
        )
        # Gating mechanism
        self.gate = nn.Linear(in_dim, out_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = x.to(self.visual_reasoner[0].weight.device)
        features = self.visual_reasoner(x)
        gate_values = torch.sigmoid(self.gate(x))
        return gate_values * self.proj(features)


class ReasonVLM():
    def __init__(self, name):
        # FIXME: code below is super slow. Why?
        self.model = LlavaNextForConditionalGeneration.from_pretrained(name, device_map="auto", torch_dtype=torch.float16)

        # visual reasoning hint layer
        in_feat = self.model.language_model.model.embed_tokens.embedding_dim
        out_feat = self.model.config.vision_config.hidden_size

        self.visual_reasoner = VisualReasoner(in_feat, out_feat).to(dtype=torch.float16)
        self.model_forward = self.model.forward
        self.base_class_embedding = self.model.vision_tower.vision_model.embeddings.class_embedding.clone()

    def set_image_reasoning(self, image_reasoning):
        # add visual reasoning hint
        self.base_class_embedding = self.model.vision_tower.vision_model.embeddings.class_embedding.clone()
        image_reasoning = self.visual_reasoner(image_reasoning)
        new_class_embedding = self.base_class_embedding + image_reasoning
        device = self.model.vision_tower.vision_model.embeddings.class_embedding.device
        new_class_embedding = torch.nn.Parameter(new_class_embedding.to(device=device, dtype=torch.float16))
        self.model.vision_tower.vision_model.embeddings.class_embedding = new_class_embedding

    def reset_image_reasoning(self):
        # reset the class embedding to the original one
        self.model.vision_tower.vision_model.embeddings.class_embedding = torch.nn.Parameter(self.base_class_embedding)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # config = kwargs.get("config", None)
        # if config is None:
        #     config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        return cls(pretrained_model_name_or_path)

    def __getattr__(self, name):
        # Only called if the attribute wasn't found the usual way
        return getattr(self.model, name)

    def prepare4training(self):
        # freeze model weights
        for param in self.model.parameters():
            param.requires_grad = False
        model.eval()
        # train the visual reasoning hint layer
        for param in self.visual_reasoner.parameters():
            param.requires_grad = True
        self.visual_reasoner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name or path of the input dataset (must include 'image' and 'question' fields)",
        default="facebook/textvqa",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="llava-hf/llama3-llava-next-8b-hf",
        help="Name or path of the model to use for generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory where the model will be saved",
        required=True,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for the optimizer (default: 1e-5)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Number of steps between saving checkpoints (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint",
    )
    args = parser.parse_args()

    run_name = os.environ.get("RUN_NAME", "default_run")
    # Create the output directory if it doesn't exist.
    os.makedirs(os.path.join(args.output_dir, run_name), exist_ok=True)

    # Initialize wandb logging.
    wandb.init(project="VLM-Reasoning", config=vars(args), name=run_name)

    model = ReasonVLM.from_pretrained(args.model_name, trust_remote_code=True)
    if args.resume:
        # load the model from the final checkpoint if it exists
        # otherwise load the model from the last checkpoint
        checkpoint_path = os.path.join(args.output_dir, run_name, "final_checkpoint.pth")
        if not os.path.exists(checkpoint_path):
            checkpoints = [f for f in os.listdir(os.path.join(args.output_dir, run_name)) if f.startswith("checkpoint-") and f.endswith(".pth")]
            checkpoints.sort(key=lambda x: int(x.split("-")[1].split(".")[0]), reverse=True)
            if checkpoints:
                checkpoint_path = os.path.join(args.output_dir, run_name, checkpoints[0])
            else:
                raise ValueError("No checkpoint found to resume from.")
        print(f"Loading checkpoint from {checkpoint_path}")
        model.visual_reasoner.load_state_dict(torch.load(checkpoint_path))
        # TODO: resume lora
    accelerator = Accelerator(mixed_precision="no", gradient_accumulation_steps=64)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split, trust_remote_code=True)
    dataset = dataset.filter(
        lambda x: x["image"] is not None and x["image"].format is not None,
        num_proc=12,
    )
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True, padding_side="left")
    wandb.watch(model.visual_reasoner, log="all")
    # Define a collate function to batch-process images and texts.
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

    # Create a DataLoader to iterate over the dataset in batches.
    batch_size = 1  # Adjust batch size as needed
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.visual_reasoner.parameters(), lr=args.lr)
    model.prepare4training()
    model.visual_reasoner = accelerator.prepare(model.visual_reasoner)
    dataloader = accelerator.prepare(dataloader)
    results = []
    global_step = 0

    for epoch in range(args.epochs):
        # Iterate over batches from the DataLoader.
        for batch in tqdm(dataloader, total=len(dataloader)):
            with accelerator.accumulate([model.visual_reasoner]):
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

                loss = model.forward(input_ids=input_ids_answers, labels=labels, pixel_values=batch["pixel_values"], image_sizes=image_sizes).loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                results.append(loss.item())
                global_step += 1

                # Local checkpointing.
                if global_step % args.checkpoint_interval == 0:
                    # Log loss to wandb.
                    wandb.log({"loss": loss.item(), "step": global_step, "epoch": epoch})
                    print(f"Epoch {epoch} Step {global_step}, Loss: {loss.item()}")

                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model.visual_reasoner)
                    checkpoint_path = os.path.join(args.output_dir, run_name, f"checkpoint-{global_step}.pth")
                    torch.save(unwrapped_model.state_dict(), checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")

                    # Keep only the 5 most recent checkpoints
                    checkpoint_dir = os.path.join(args.output_dir, run_name)
                    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint-") and f.endswith(".pth")]
                    checkpoints.sort(key=lambda x: int(x.split("-")[1].split(".")[0]), reverse=True)
                    for old_checkpoint in checkpoints[5:]:
                        old_checkpoint_path = os.path.join(checkpoint_dir, old_checkpoint)
                        os.remove(old_checkpoint_path)
                        print(f"Removed old checkpoint: {old_checkpoint_path}")

    # Save the final model checkpoint.
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model.visual_reasoner)
    final_checkpoint_path = os.path.join(args.output_dir, run_name, "final_checkpoint.pth")
    torch.save(unwrapped_model.state_dict(), final_checkpoint_path)
    print(f"Final model saved: {final_checkpoint_path}")

    # 1. Create your model (unprepared).
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

    # 3. Create trainable parameter iterator and optimizer (still unprepared).
    trainable_params = itertools.chain(
        model.visual_reasoner.parameters(),
        model.model.language_model.parameters()  # includes LoRA parameters
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # 4. Prepare the entire training setup in *one go*:
    model.model, model.visual_reasoner, optimizer, dataloader = accelerator.prepare(
        model.model,
        model.visual_reasoner,
        optimizer,
        dataloader
    )

    # 5. Now set all modules to .train() mode.
    model.model.language_model.train()
    model.visual_reasoner.train()

    global_step = 0
    for epoch in range(args.epochs):
        for batch in tqdm(dataloader, total=len(dataloader)):
            with accelerator.accumulate([model.model.language_model, model.visual_reasoner]):
                model.reset_image_reasoning()

                # Move to device
                image_sizes = batch.pop("image_sizes", None)
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                labels = batch.pop("labels", None)
                input_ids_answers = batch.pop("input_ids_answers", None)

                # -------------------------------------------------
                # FIRST FORWARD PASS: ENABLE LoRA ADAPTER
                # -------------------------------------------------
                output = model.forward(
                    **batch,
                    image_sizes=image_sizes,
                    output_hidden_states=True
                )
                # Collect the final hidden state for image reasoning
                image_reasoning = output.hidden_states[-1][:, -1, :]
                model.set_image_reasoning(image_reasoning)

                # -------------------------------------------------
                # SECOND FORWARD PASS: DISABLE LoRA ADAPTER
                # -------------------------------------------------
                model.model.language_model.disable_adapter_layers()  # Turn off LoRA
                output2 = model.forward(
                    input_ids=input_ids_answers,
                    image_sizes=image_sizes,
                    pixel_values=batch["pixel_values"],
                    labels=labels
                )
                loss = output2.loss
                model.model.language_model.enable_adapter_layers()  # Turn on LoRA

                # Backpropagate on the second pass's loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                # Local checkpointing
                if global_step % args.checkpoint_interval == 0:
                    wandb.log({"loss_couple": loss.item(), "step": global_step, "epoch": epoch})
                    print(f"Epoch {epoch} Step {global_step}, Loss: {loss.item()}")
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model.visual_reasoner)
                    checkpoint_path = os.path.join(args.output_dir, run_name, f"checkpoint-{global_step}.pth")
                    torch.save(unwrapped_model.state_dict(), checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")
                    model.model.language_model.save_pretrained(os.path.join(args.output_dir, run_name))

                    # Keep only the 5 most recent checkpoints
                    checkpoint_dir = os.path.join(args.output_dir, run_name)
                    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint-") and f.endswith(".pth")]
                    checkpoints.sort(key=lambda x: int(x.split("-")[1].split(".")[0]), reverse=True)
                    for old_checkpoint in checkpoints[5:]:
                        old_checkpoint_path = os.path.join(checkpoint_dir, old_checkpoint)
                        os.remove(old_checkpoint_path)
                        print(f"Removed old checkpoint: {old_checkpoint_path}")

    # Final checkpoint
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model.visual_reasoner)
    final_checkpoint_path = os.path.join(args.output_dir, run_name, "final_checkpoint.pth")
    torch.save(unwrapped_model.state_dict(), final_checkpoint_path)
    model.model.language_model.save_pretrained(os.path.join(args.output_dir, run_name))
    print(f"Final model saved: {final_checkpoint_path}")
