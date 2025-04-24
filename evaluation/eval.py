"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import json
import os
import time
import torch
import numpy as np
import shortuuid
import random

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm
from datasets import load_dataset

def load_binary_data(bin_file, max_samples=None):
    """Load data from binary file."""
    print(f"Loading binary data from {bin_file}")
    data = np.memmap(bin_file, dtype=np.uint16, mode='r')
    
    # If the data is 2D (shaped as samples x tokens), reshape it
    if os.path.exists(bin_file.replace('.bin', '_metadata.json')):
        # Load metadata for CNN/DailyMail
        with open(bin_file.replace('.bin', '_metadata.json'), 'r') as f:
            metadata = json.load(f)
        max_length = metadata['max_length']
        num_samples = len(metadata['lengths'])
        data = data.reshape(num_samples, max_length)
    else:
        # For OpenWebText, infer shape from max_samples and context_length
        context_length = 768  # Default OpenWebText context length
        num_samples = len(data) // context_length
        data = data.reshape(num_samples, context_length)
    
    if max_samples is not None:
        data = data[:max_samples]
    
    print(f"Loaded {len(data)} samples")
    return data

def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        dataset_type="spec_bench",
        max_samples=200,
        **kwargs,
):
    if dataset_type == "spec_bench" or dataset_type == "mt_bench":
        questions = load_questions(question_file, question_begin, question_end)
        get_answers = get_model_answers
    elif dataset_type == "cnn_dailymail":
        # Load pre-selected CNN/DailyMail samples from binary file
        questions = load_binary_data("data/cnn_dailymail/val_200.bin", max_samples)
        get_answers = get_model_answers_cnn
    elif dataset_type == "openwebtext":
        # Load pre-selected OpenWebText samples from binary file
        questions = load_binary_data("data/openwebtext/val_200.bin", max_samples)
        get_answers = get_model_answers_owt
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_answers
        ).remote
    else:
        get_answers_func = get_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        # Only pass max_new_tokens for spec_bench/mt_bench
        if dataset_type in ["spec_bench", "mt_bench"]:
            ans_handles.append(
                get_answers_func(
                    model,
                    tokenizer,
                    forward_func,
                    model_id,
                    questions[i: i + chunk_size],
                    answer_file,
                    max_new_tokens,
                    num_choices,
                    **kwargs,
                )
            )
        else:
            ans_handles.append(
                get_answers_func(
                    model,
                    tokenizer,
                    forward_func,
                    model_id,
                    questions[i: i + chunk_size],
                    answer_file,
                    num_choices,
                    **kwargs,
                )
            )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        num_choices,
        **kwargs,
):

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)
        conv = get_conversation_template("vicuna")
        turns = []
        steps = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            conv.stop_str = "</s>"
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            
            torch.cuda.synchronize()
            start_time = time.time()
            output_ids, new_token, step, accept_length_tree = forward_func(
                inputs,
                model,
                tokenizer,
                max_new_tokens,
                **kwargs,
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            # be consistent with the template's stop_token_ids
            if conv.stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in conv.stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            if conv.stop_str and output.find(conv.stop_str) > 0:
                output = output[: output.find(conv.stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            if conv.name == "xgen" and output.startswith("Assistant:"):
                output = output.replace("Assistant:", "", 1).strip()
            

            turns.append(output)
            steps.append(int(step))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
    print('Warmup done')

    accept_lengths_tree = []
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            cur_accept_lengths_tree = []
            torch.manual_seed(i)
            conv = get_conversation_template("vicuna")
            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                conv.stop_str = "</s>"
                prompt = conv.get_prompt()
                inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
                input_ids = inputs.input_ids
                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    output_ids, new_token, step, accept_length_tree = forward_func(
                        inputs,
                        model,
                        tokenizer,
                        max_new_tokens,
                        **kwargs,
                    )
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    accept_lengths_tree.extend(accept_length_tree)
                    output_ids = output_ids[0][len(input_ids[0]):]

                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)
                steps.append(int(step))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                cur_accept_lengths_tree.extend(accept_length_tree)
                conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                            "accept_lengths": cur_accept_lengths_tree})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))


@torch.inference_mode()
def get_model_answers_cnn(
        model,
        tokenizer,
        forward_func,
        model_id,
        token_data,
        answer_file,
        num_choices,
        **kwargs,
):
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    # Process each sample
    accept_lengths_tree = []
    for idx, tokens in enumerate(tqdm(token_data)):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            
            # Convert tokens to input format (similar to OWT approach)
            input_ids = torch.tensor([tokens.tolist()], dtype=torch.long).to("cuda")
            attention_mask = torch.ones_like(input_ids).to("cuda")
            
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, step, accept_length_tree = forward_func(
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                    model,
                    tokenizer,
                    256,
                    **kwargs,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                accept_lengths_tree.extend(accept_length_tree)
                
                # Get only the newly generated tokens
                generated_ids = output_ids[0][len(input_ids[0]):]

                # Decode the full article for reference
                article = tokenizer.decode(
                    input_ids[0],
                    spaces_between_special_tokens=False,
                )
                
                # Decode just the newly generated summary
                output = tokenizer.decode(
                    generated_ids,
                    spaces_between_special_tokens=False,
                )
                
                # Clean up any special tokens
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()
                
            except RuntimeError as e:
                print("ERROR article index: ", idx)
                print(e)
                output = "ERROR"
                total_time = 0
                step = 0
                new_token = 0
                accept_length_tree = []

            # Store results for this choice
            choices.append({
                "index": i, 
                "summary": output,  # Using 'summary' as the key for CNN data
                "decoding_steps": int(step), 
                "new_tokens": int(new_token), 
                "wall_time": total_time,
                "accept_lengths": accept_length_tree
            })

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "article_idx": idx,
                "article": article[:100] + "...",  # Store a snippet of the article
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
    
    if len(accept_lengths_tree) > 0:
        print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))


@torch.inference_mode()
def get_model_answers_owt(
        model,
        tokenizer,
        forward_func,
        model_id,
        token_data,
        answer_file,
        num_choices,
        **kwargs,
):
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    # Process each sample
    for idx, tokens in enumerate(tqdm(token_data)):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            
            # Convert tokens to input format
            input_ids = torch.tensor([tokens.tolist()], dtype=torch.long).to("cuda")
            attention_mask = torch.ones_like(input_ids).to("cuda")
            
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, step, accept_length_tree = forward_func(
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                    model,
                    tokenizer,
                    256,
                    **kwargs,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                
                # Get only the newly generated tokens
                generated_ids = output_ids[0][len(input_ids[0]):]

                # Decode the full completion
                context_str = tokenizer.decode(
                    input_ids[0],
                    spaces_between_special_tokens=False,
                )
                
                # Decode just the newly generated content
                output = tokenizer.decode(
                    generated_ids,
                    spaces_between_special_tokens=False,
                )
                
                # Clean up any special tokens
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()
                
            except RuntimeError as e:
                print("ERROR context index: ", idx)
                print(e)
                output = "ERROR"
                total_time = 0
                step = 0
                new_token = 0
                accept_length_tree = []

            # Store results for this choice
            choices.append({
                "index": i, 
                "completion": output, 
                "decoding_steps": int(step), 
                "new_tokens": int(new_token), 
                "wall_time": total_time,
                "accept_lengths": accept_length_tree  # Use the per-choice accept_length_tree
            })

        # Decode context for reference (truncated to save space)
        context_str = tokenizer.decode(
            input_ids[0],
            spaces_between_special_tokens=False,
        )
        
        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "context_idx": idx,
                "context": context_str[:100] + "...",  # Store a snippet of the context
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
    
    print("#Mean accepted tokens: ", np.mean(accept_length_tree))


def reorg_answer_file(answer_file):
    """Sort by ID and de-duplication across different dataset types"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            data = json.loads(l)
            # Handle different ID fields based on dataset type
            if "question_id" in data:
                # For spec_bench dataset
                id_key = data["question_id"]
            elif "article_idx" in data:
                # For CNN/DailyMail dataset
                id_key = f"article_{data['article_idx']}"
            elif "context_idx" in data:
                # For OpenWebText dataset
                id_key = f"context_{data['context_idx']}"
            else:
                # Fallback to answer_id if no other ID is available
                id_key = data["answer_id"]
            
            answers[id_key] = l

    # Sort IDs while keeping dataset types grouped
    sorted_ids = sorted(list(answers.keys()))
    
    with open(answer_file, "w") as fout:
        for id_key in sorted_ids:
            fout.write(answers[id_key])
    
    print(f"Reorganized {len(sorted_ids)} entries in {answer_file}")