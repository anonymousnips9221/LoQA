import argparse
import torch
from auto_gptq.modeling.auto import AutoGPTQForCausalLM
from evaluate_hf import test_func
import os
import csv
from auto_gptq.utils.peft_utils import get_gptq_peft_model, GPTQLoraConfig
from transformers import AutoTokenizer, LlamaTokenizerFast
import transformers
from typing import Dict
from peft import PeftModel

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def load_loqa(load_path, model):
    weights = torch.load(load_path)
    loaded = 0
    for n, p in model.named_parameters():
        if any([x in n for x in ["lora"]]):
            p.data = weights[n]
            loaded += 1
    print(f"successfully loaded {loaded} trained parameter tensors")
    return model


def load_model_and_tokenizer(checkpoint_path):
    peft_config = GPTQLoraConfig.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path
    )
    model = AutoGPTQForCausalLM.from_quantized(
        peft_config.base_model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        inject_fused_attention=False,
        inject_fused_mlp=False,
        use_triton=True,
        warmup_triton=False,
        trainable=True,
        # torch_dtype=torch.float32
    )
    model.model.quantize_config = model.quantize_config
    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)
    model = get_gptq_peft_model(model, peft_config, checkpoint_path)
    model = load_loqa(os.path.join(checkpoint_path, "adapter_model"), model=model)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        peft_config.base_model_name_or_path,
        padding_side="left",
        truncation_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if isinstance(tokenizer, LlamaTokenizerFast):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        tokenizer.add_special_tokens(
            {
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != -1 else 0
                ),
            }
        )

    return model, tokenizer


def evaluate_checkpoint(checkpoint_path, data_dir):
    print(f"Evaluating checkpoint: {checkpoint_path}")

    model, tokenizer = load_model_and_tokenizer(checkpoint_path)

    checkpoint_results = []
    for ntrain in [0, 5]:
        print(f"Evaluating {checkpoint_path} with ntrain={ntrain}")
        results = test_func(
            model=model,
            tokenizer=tokenizer,
            ntrain=ntrain,
            data_dir=data_dir,
        )
        checkpoint_results.append(
            [
                checkpoint_path,
                ntrain,
                results["weighted_accuracy"],
                results["categories"]["STEM"],
                results["categories"]["humanities"],
                results["categories"]["social sciences"],
                results["categories"]["other (business, health, misc.)"],
            ]
        )

    return checkpoint_results


def evaluate_model(eval_directory, results_file, data_dir):
    checkpoint_path = eval_directory

    print(f"Starting evaluation for checkpoint {checkpoint_path}.")
    checkpoint_results = evaluate_checkpoint(checkpoint_path, data_dir)

    with open(
        os.path.join(checkpoint_path, results_file), mode="w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "checkpoint",
                "ntrain",
                "avg",
                "STEM",
                "humanities",
                "social sciences",
                "other",
            ]
        )
        for row in checkpoint_results:
            writer.writerow(row)
    print("Evaluation completed and results are saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on a single GPU")
    parser.add_argument(
        "--eval_directory",
        type=str,
        required=True,
        help="The directory where the evaluation model is stored (single checkpoint path)",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="The file name where the results will be saved",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The directory where the evaluation data is stored",
    )
    args = parser.parse_args()

    evaluate_model(args.eval_directory, args.results_file, args.data_dir)
