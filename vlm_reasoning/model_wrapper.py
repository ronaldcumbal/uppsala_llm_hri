import os
import argparse
import itertools
import logging
from types import MethodType

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import LlavaNextForConditionalGeneration, AutoProcessor, get_cosine_schedule_with_warmup
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
import wandb
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, load_peft_weights


# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_prompt(example):
    assert "image" in example, "The example must have an 'image' field"
    assert not isinstance(example["image"], list), "Cannot give more than 1 image"
    assert "question" in example, "The example must have a 'question' field"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": example["question"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me begin by analyzing the image. Based on the question, the image would benefit from being manipulated like this: [manipulation]"},
            ],
        },
    ]
    return messages


# ---- Checkpoint Helpers ----
def save_checkpoint(model: nn.Module, checkpoint_dir: str, run_name: str, global_step: int) -> None:
    os.makedirs(os.path.join(checkpoint_dir, run_name), exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, run_name, f"checkpoint-{global_step}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    cleanup_checkpoints(os.path.join(checkpoint_dir, run_name), keep=5)


def cleanup_checkpoints(checkpoint_dir: str, keep: int = 3) -> None:
    checkpoints = [f for f in os.listdir(checkpoint_dir)
                   if f.startswith("checkpoint-") and f.endswith(".pth")]
    checkpoints.sort(key=lambda x: int(x.split("-")[1].split(".")[0]), reverse=True)
    for old_ckpt in checkpoints[keep:]:
        path = os.path.join(checkpoint_dir, old_ckpt)
        os.remove(path)
        logger.info(f"Removed old checkpoint: {path}")


# ---- Model Definitions ----
class VisualReasoner(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """
        A simple feed-forward network with a gating mechanism.
        """
        super(VisualReasoner, self).__init__()
        hidden_dim = in_dim * 2
        self.visual_reasoner = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        self.gate = nn.Linear(in_dim, out_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.visual_reasoner[0].weight.device)
        features = self.visual_reasoner(x)
        gate_values = torch.sigmoid(self.gate(x))
        return gate_values * self.proj(features)


# used to get the position in a sequence of the image (first or last)
def get_contiguous_block_mask(input_ids, target_token_index, mode='first'):
    """
    Extracts a mask for the first or last contiguous block of `target_token_index` in each sequence.
    """
    bs, seq_len = input_ids.shape
    masks = torch.zeros_like(input_ids, dtype=torch.bool)

    for b in range(bs):
        positions = (input_ids[b] == target_token_index).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            continue

        # Find contiguous blocks (simple approach)
        diffs = positions[1:] - positions[:-1]
        breaks = (diffs != 1).nonzero(as_tuple=True)[0] + 1
        split_positions = torch.tensor_split(positions, breaks.tolist())

        if mode == 'first':
            block = split_positions[0]
        elif mode == 'last':
            block = split_positions[-1]
        else:
            raise ValueError("mode must be 'first' or 'last'")

        masks[b, block[0]:block[-1] + 1] = True

    return masks.unsqueeze(-1)


class ReasonVLM(nn.Module):
    def __init__(self, model_name: str, torch_dtype="auto", deepspeed=False):
        """
        Load the pre-trained model and attach a visual reasoning head.
        """
        super().__init__()
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto" if not deepspeed else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        in_feat = self.model.language_model.model.embed_tokens.embedding_dim
        out_feat = self.model.config.vision_config.hidden_size
        # do not cast, this will be fp32 to improve stability
        self.visual_reasoner = VisualReasoner(in_feat, out_feat)#.to(self.model.dtype)

        # Keep a clone of the original vision embeddings forward method.
        self._original_embeddings_forward = None
        self.reasoning_hint = None
        self.base_class_embedding = self.model.vision_tower.vision_model.embeddings.class_embedding.clone()

    def second_forward(self, input_ids, input_ids_answers, attention_mask, pixel_values, image_sizes):
        # first, add the image token ids to the end of the sequence in the same number as the beginning
        image_token = self.model.config.image_token_index
        # count how many image tokens are in the input_ids
        image_token_count = (input_ids == image_token).sum(dim=1)
        # add the image token to the end of the input_ids
        image_token_sequence = torch.LongTensor([image_token] * image_token_count.max()).unsqueeze(0).expand(input_ids.size(0), -1).to(input_ids.device)
        input_ids = torch.cat([input_ids, image_token_sequence], dim=1)
        # FIXME: this is okay for bs=1 but needs to be padded for larger batches
        input_ids = torch.cat([input_ids, input_ids_answers], dim=1)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # get the labels by copying the input_ids and setting the first part that is not the answer to -100
        labels = torch.full_like(input_ids, -100)
        # Align the labels to the final input_ids_answers tokens
        labels[:, -input_ids_answers.size(1):] = input_ids_answers

        # now update the attention mask to include the image tokens and the answer tokens
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.size(0), image_token_count.max()))], dim=1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.size(0), input_ids_answers.size(1)))], dim=1)

        if pixel_values is not None and pixel_values.size(0) > 0:
            # Regular image embeddings first pass
            image_features = self.model.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=self.model.config.vision_feature_layer,
                vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
            )
            image_features, feature_lens = self.model.pack_image_features(
                image_features,
                image_sizes,
                vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
                image_newline=self.model.image_newline,
            )

            # Now process custom embeddings (second pass) by replacing the forward of the embeddings
            vision_embeddings = self.model.vision_tower.vision_model.embeddings
            if self._original_embeddings_forward is None:
                self._original_embeddings_forward = vision_embeddings.forward

            reasoning_hint = self.reasoning_hint
            if reasoning_hint is None:
                raise ValueError("Reasoning hint is not set. Call set_image_reasoning() before forward.")

            def patched_forward(self_embed, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
                batch_size, _, height, width = pixel_values.shape
                if not interpolate_pos_encoding and (height != self_embed.image_size or width != self_embed.image_size):
                    raise ValueError(
                        f"Input image size ({height}x{width}) doesn't match model requirement "
                        f"({self_embed.image_size}x{self_embed.image_size})."
                    )
                target_dtype = self_embed.patch_embedding.weight.dtype
                patch_embeds = self_embed.patch_embedding(pixel_values.to(dtype=target_dtype))
                patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
                # Inject the reasoning hint into the class embedding.
                class_embeds = self_embed.class_embedding + reasoning_hint.to(self_embed.class_embedding.device)
                class_embeds = class_embeds.expand(batch_size, 1, -1)
                embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
                if interpolate_pos_encoding:
                    embeddings = embeddings + self_embed.interpolate_pos_encoding(embeddings, height, width)
                else:
                    embeddings = embeddings + self_embed.position_embedding(self_embed.position_ids)
                return embeddings

            vision_embeddings.forward = MethodType(patched_forward, vision_embeddings)

            second_image_features = self.model.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=self.model.config.vision_feature_layer,
                vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
            )
            second_image_features, _ = self.model.pack_image_features(
                second_image_features,
                image_sizes,
                vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
                image_newline=self.model.image_newline,
            )
            # Integrate both embeddings in inputs_embeds
            special_image_mask = get_contiguous_block_mask(input_ids, self.model.config.image_token_index, mode='first')
            special_custom_image_mask = get_contiguous_block_mask(input_ids, self.model.config.image_token_index, mode='last')

            if special_image_mask.sum() != image_features.numel() // image_features.size(-1):
                raise ValueError("Mismatch in standard image tokens and embeddings.")

            if special_custom_image_mask.sum() != second_image_features.numel() // second_image_features.size(-1):
                raise ValueError("Mismatch in custom image tokens and embeddings.")

            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask.expand_as(inputs_embeds), image_features.to(inputs_embeds.dtype))
            inputs_embeds = inputs_embeds.masked_scatter(special_custom_image_mask.expand_as(inputs_embeds), second_image_features.to(inputs_embeds.dtype))
        return self.model.forward(inputs_embeds=inputs_embeds, pixel_values=None, labels=labels)

    def set_image_reasoning(self, image_reasoning: torch.Tensor) -> None:
        """
        Monkey-patch the vision embeddings' forward method to inject the visual reasoning hint.
        """
        # Compute reasoning hint.
        self.reasoning_hint = self.visual_reasoner(image_reasoning)

    def reset_image_reasoning(self) -> None:
        self.reasoning_hint = None
        if self._original_embeddings_forward is not None:
            self.model.vision_tower.vision_model.embeddings.forward = self._original_embeddings_forward
            self._original_embeddings_forward = None

    def forward(self, *args, **kwargs):
        if self.reasoning_hint is not None:
            # Use the second forward pass with custom embeddings.
            return self.second_forward(*args, **kwargs)
        # Use the original forward pass.
        return self.model(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "ReasonVLM":
        return cls(model_name, **kwargs)


# ---- Data Collation ----
def collate_fn(batch, processor, tokenizer):
    texts = [apply_prompt(item) for item in batch]
    answers = [item["answers"] for item in batch]
    answers = [max(set(ans), key=ans.count) for ans in answers]
    answers = [f" Based on the new image, the answer is: {ans}" for ans in answers]
    images = [item["image"] for item in batch]

    # First forward pass prompts.
    prompts = processor.apply_chat_template(texts, continue_final_message=True)
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)

    # Second forward pass prompts for answer generation.
    answers = processor(text=answers, return_tensors="pt", padding=False, add_special_tokens=False)

    inputs["input_ids_answers"] = answers["input_ids"]
    return inputs


# ---- Unified Training Loop ----
def train_loop(model: ReasonVLM, dataloader: DataLoader, optimizer, lr_scheduler, accelerator, args, run_name: str, stage: int):
    """
    A unified training loop that handles both training phases.
      - Stage 1 updates only the visual_reasoner.
      - Stage 2 (with LoRA) updates both the visual_reasoner and the LM's adapter layers.
    """
    logger.info(f"Starting training Stage {stage}")
    global_step = 0
    unwrapped_model = accelerator.unwrap_model(model)
    # Save initial parameters to check that updates are applied.
    old_params = {
        name: param.clone().detach()
        for name, param in unwrapped_model.visual_reasoner.named_parameters()
    }

    for epoch in range(args.epochs):
        for batch in tqdm(dataloader, total=len(dataloader), desc=f"Stage {stage} Epoch {epoch}"):
            with accelerator.accumulate(model):
                unwrapped_model.reset_image_reasoning()
                image_sizes = batch.pop("image_sizes", None)
                input_ids_answers = batch.pop("input_ids_answers", None)

                # ---------- First Forward Pass ----------
                output = model.forward(**batch, image_sizes=image_sizes, output_hidden_states=True)
                # FIXME: not sure why but even if model is bf16, the output is float32
                # this is good atm as visual reasoner is fp32
                image_reasoning = output.hidden_states[-1][:, -1, :]
                unwrapped_model.set_image_reasoning(image_reasoning)

                # ---------- Second Forward Pass ----------
                if stage == 2:
                    # disable LoRA layers temporarily.
                    unwrapped_model.model.language_model.disable_adapter_layers()
                with accelerator.autocast():
                    output2 = model.forward(
                        **batch,
                        input_ids_answers=input_ids_answers,
                        image_sizes=image_sizes,
                    )
                    loss = output2.loss
                if stage == 2:
                    unwrapped_model.model.language_model.enable_adapter_layers()

                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"Loss is {loss}, terminating training.")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ---------- Checkpointing ----------
                if global_step % args.checkpoint_interval == 0 and accelerator.is_main_process:
                    unw_vr = accelerator.unwrap_model(model).visual_reasoner
                    if old_params is not None:
                        # We only check params updating once
                        updated = any(
                            not torch.equal(param.data.cpu(), old_params[name].cpu())
                            for name, param in unw_vr.named_parameters()
                        )
                        if not updated:
                            # FIXME: right now, model is updated 2*bs*accum_steps this means this will raise
                            # if bs=1 and accum_steps > 50 with defualt checkpoint_interval=100
                            raise RuntimeError("[ERROR] visual_reasoner parameters not updating! Note: check this is not caused bycheckpoint_interval < accumulated batch size.")
                        old_params = None
                    wandb.log({f"loss_stage_{stage}": loss.item(), "step": global_step, "epoch": epoch})
                    logger.info(f"Stage {stage} Epoch {epoch} Step {global_step}, Loss: {loss.item()}")
                    checkpoint_dir = os.path.join(args.output_dir, f"stage{stage}")
                    save_checkpoint(unw_vr, checkpoint_dir, run_name, global_step)
                    if stage == 2:
                        accelerator.unwrap_model(model).model.language_model.save_pretrained(os.path.join(args.output_dir, run_name))
    else:
        accelerator.wait_for_everyone()
        final_path = os.path.join(args.output_dir, run_name, f"final_checkpoint_stage{stage}.pth")
        torch.save(accelerator.unwrap_model(model).visual_reasoner.state_dict(), final_path)
        if stage == 2:
            accelerator.unwrap_model(model).model.language_model.save_pretrained(os.path.join(args.output_dir, run_name))
        logger.info(f"Stage {stage} Final model saved: {final_path}")


# ---- Main Routine ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="facebook/textvqa", help="Dataset name/path")
    parser.add_argument("--training-stage", type=int, default=None,
                        help="Training stage (1 or 2, default: None = both stages)")
    parser.add_argument("--half-precision", action="store_true", help="Use half precision")
    parser.add_argument("--deepspeed", type=int, default=None, help="Use DeepSpeed with the specified stage")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--model-name", type=str, default="llava-hf/llama3-llava-next-8b-hf", help="Model name/path")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for saving checkpoints")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--checkpoint-interval", type=int, default=128, help="Steps between saving checkpoints")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    args = parser.parse_args()

    # if run_name is not set, we won't use wandb as it is likely to be a local debug run
    run_name = os.environ.get("RUN_NAME", None)
    output_path = os.path.join(args.output_dir, run_name if run_name else "debug")
    os.makedirs(output_path, exist_ok=True)

    mixed_precision = "bf16" if args.half_precision else "no"
    deepspeed_plugin = None
    if args.deepspeed is not None:
        deepspeed_plugin = DeepSpeedPlugin(
            "./deep.json",
            zero_stage=args.deepspeed,
        )
    # NOTE: this code does not work with grad accumulation
    accelerator = Accelerator(mixed_precision=mixed_precision, deepspeed_plugin=deepspeed_plugin, gradient_accumulation_steps=1)
    if accelerator.is_main_process:
        wandb.init(project="VLM-Reasoning", config=vars(args), name=run_name, mode="disabled" if run_name is None else None)

    # Load model and resume if needed.
    model = ReasonVLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16 if args.half_precision else torch.float32, deepspeed=args.deepspeed is not None)
    if args.resume:
        checkpoint_path = os.path.join(output_path, "final_checkpoint.pth")
        if not os.path.exists(checkpoint_path):
            ckpts = [f for f in os.listdir(output_path)
                     if f.startswith("checkpoint-") and f.endswith(".pth")]
            if ckpts:
                ckpts.sort(key=lambda x: int(x.split("-")[1].split(".")[0]), reverse=True)
                checkpoint_path = os.path.join(output_path, ckpts[0])
            else:
                raise ValueError("No checkpoint found to resume from.")
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model.visual_reasoner.load_state_dict(torch.load(checkpoint_path))

    with accelerator.main_process_first():
        # Prepare dataset.
        if args.dataset_name == "facebook/textvqa":
            dataset = load_dataset(args.dataset_name, split=args.dataset_split, trust_remote_code=True)
            dataset = dataset.filter(lambda x: x["image"] is not None and x["image"].format is not None, num_proc=12)
        else:
            # likely VisualCoT
            dataset = load_from_disk(args.dataset_name)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True, padding_side="left")
    data_loader = DataLoader(dataset, batch_size=1,
                             collate_fn=lambda batch: collate_fn(batch, processor, processor.tokenizer))

    if accelerator.is_main_process:
        wandb.watch(model.visual_reasoner, log="all")
    # Prepare model for training stage 1
    # Freeze all pre-trained parameters.
    for param in model.model.parameters():
        param.requires_grad = False
    model.model.eval()
    # Only update the visual_reasoner head during stage 1.
    for param in model.visual_reasoner.parameters():
        param.requires_grad = True
    model.visual_reasoner.train()
    optimizer_stage1 = torch.optim.AdamW(model.visual_reasoner.parameters(), lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer_stage1,
        num_warmup_steps=100,
        num_training_steps=args.epochs * len(data_loader),
    )

    # --- Stage 1 Training (or full training without LoRA) ---
    if args.training_stage is None or args.training_stage == 1:
        # Prepare modules with accelerator.
        model, optimizer_stage1, lr_scheduler, data_loader = accelerator.prepare(
            model, optimizer_stage1, lr_scheduler, data_loader
        )
        train_loop(model, data_loader, optimizer_stage1, lr_scheduler, accelerator, args, run_name, stage=1)
        model = accelerator.unwrap_model(model)

    # --- Stage 2 Training: Inject LoRA and continue training ---
    if args.training_stage is None or args.training_stage == 2:
        base_lm = model.model.language_model
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        base_lm = get_peft_model(base_lm, lora_config)
        if args.resume:
            # load LoRA checkpoint if resuming
            base_lm.load_adapter(checkpoint_path, adapter_name="default")
        model.model.language_model = base_lm

        # set trainable params
        trainable_params = itertools.chain(
            model.visual_reasoner.parameters(),
            model.model.language_model.parameters()
        )
        optimizer_stage2 = torch.optim.AdamW(trainable_params, lr=args.lr)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer_stage2,
            num_warmup_steps=100,
            num_training_steps=args.epochs * len(data_loader),
        )
        model, optimizer_stage2, lr_scheduler, data_loader = accelerator.prepare(
            model, optimizer_stage2, lr_scheduler, data_loader
        )
        train_loop(model, data_loader, optimizer_stage2, lr_scheduler, accelerator, args, run_name, stage=2)


if __name__ == "__main__":
    main()
