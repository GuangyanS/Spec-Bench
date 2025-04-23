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

def load_openwebtext_data(bin_file, max_samples=200, context_length=768):
    """Load data from OpenWebText binary file."""
    # Load binary data
    print(f"Loading OpenWebText data from {bin_file}")
    data = np.memmap(bin_file, dtype=np.uint16, mode='r')
    
    # Find valid starting points for sequences
    samples = []
    sample_count = 0
    
    # Try to find samples with at least context_length tokens
    idx = 0
    while idx < len(data) - context_length and sample_count < max_samples:
        # Take a chunk of data and ensure it's at least context_length
        context = data[idx:idx+context_length].tolist()
        samples.append(context)
        sample_count += 1
        
        # Move to next potential sample (with some overlap possible)
        idx += context_length  # Could add randomness here if desired
    
    print(f"Loaded {len(samples)} OpenWebText samples")
    return samples

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
        max_cnndm_questions=200,
        max_owt_samples=200,
        owt_context_length=768,
        filter_by_length=True,
        max_length=1024,
        **kwargs,
):
    if dataset_type == "spec_bench" or dataset_type == "mt_bench":
        questions = load_questions(question_file, question_begin, question_end)
        get_answers = get_model_answers
    elif dataset_type == "cnn_dailymail":
        # Load CNN/DailyMail dataset with a maximum limit
        full_questions = load_dataset("cnn_dailymail", "3.0.0", split="validation", streaming=False)["article"][question_begin:question_end]
        
        # Apply maximum limit if needed
        if filter_by_length and hasattr(tokenizer, "encode"):
            print(f"Filtering CNN/DailyMail articles by token count (max: {max_length})")
            # Find articles under the token limit
            short_articles = []
            article_lengths = []
            max_token_threshold = 700  # Filter to articles under this token length
            
            for article in full_questions:
                tokens = tokenizer.encode(article, truncation=False, add_special_tokens=False)
                token_length = len(tokens)
                article_lengths.append(token_length)
                
                if token_length < max_token_threshold:
                    short_articles.append(article)
            
            print(f"Found {len(short_articles)} articles under {max_token_threshold} tokens")
            print(f"Range of token counts in dataset: {min(article_lengths)} - {max(article_lengths)}")
            
            # Randomly select articles if we have more than we need
            if len(short_articles) > max_cnndm_questions:
                # Set a seed for reproducibility
                random.seed(42)
                questions = random.sample(short_articles, max_cnndm_questions)
                print(f"Randomly selected {max_cnndm_questions} articles from {len(short_articles)} eligible articles")
            else:
                # If we don't have enough articles under the threshold, take all we have
                questions = short_articles
                print(f"Using all {len(short_articles)} articles under the token threshold (fewer than requested {max_cnndm_questions})")
        else:
            # Standard limit without token count filtering
            questions = full_questions[:max_cnndm_questions]
            
        print(f"Loaded {len(questions)} CNN/DailyMail articles (limited to max {max_cnndm_questions})")
        get_answers = get_model_answers_cnn
    elif dataset_type == "openwebtext":
        # For OpenWebText, question_file should be the path to the binary file
        questions = load_openwebtext_data(
            "data/openwebtext/val.bin", 
            max_samples=max_owt_samples, 
            context_length=owt_context_length
        )
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

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
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
        articles,
        answer_file,
        max_new_tokens,
        num_choices,
        max_length=1024,
        **kwargs,
):
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    # Set pad token id if needed
    if tokenizer.pad_token_id is None and hasattr(tokenizer, "eos_token_id"):
        print("Pad token ID not set. Using EOS token as pad token.")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    article = articles[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)
        conv = get_conversation_template("vicuna")
        summary = ""
        steps = 0
        new_tokens = 0
        wall_time = 0
        
        # For CNN dataset, we use the article directly instead of turns
        qs = f"Summarize the following article: {article}"
        
        # Check article token length and truncate if necessary
        article_tokens = tokenizer.encode(qs, truncation=False, add_special_tokens=False)
        if len(article_tokens) > max_length - 100:  # Leave room for prompt and generation
            print(f"Truncating warmup article from {len(article_tokens)} tokens to fit max_length={max_length}")
            # Decode just enough tokens to fit within max_length
            truncated_tokens = article_tokens[:max_length - 100]
            qs = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
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
            256,
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
    
    print('Warmup done')

    accept_lengths_tree = []
    for idx, article in enumerate(tqdm(articles)):
        choices = []
        for i in range(num_choices):
            cur_accept_lengths_tree = []
            torch.manual_seed(i)
            conv = get_conversation_template("vicuna")
            
            # For CNN dataset, process the article as a single input
            qs = f"Summarize the following article: {article}"
            
            # Check article token length and truncate if necessary
            article_tokens = tokenizer.encode(qs, truncation=False, add_special_tokens=False)
            if len(article_tokens) > max_length - 100:  # Leave room for prompt and generation
                # Decode just enough tokens to fit within max_length
                truncated_tokens = article_tokens[:max_length - 100]
                qs = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            conv.stop_str = "</s>"
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            
            # Create attention mask if not present
            if 'attention_mask' not in inputs:
                attention_mask = torch.ones_like(input_ids).to("cuda")
                inputs['attention_mask'] = attention_mask
            
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, step, accept_length_tree = forward_func(
                    inputs,
                    model,
                    tokenizer,
                    256,
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
                print("ERROR article index: ", idx)
                output = "ERROR"

            # Store results for this choice
            choices.append({
                "index": i, 
                "summary": output, 
                "decoding_steps": int(step), 
                "new_tokens": int(new_token), 
                "wall_time": total_time,
                "accept_lengths": cur_accept_lengths_tree
            })

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "article_idx": idx,
                "article": article[:100] + "...",  # Store a snippet of the article for reference
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
    
    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))


@torch.inference_mode()
def get_model_answers_owt(
        model,
        tokenizer,
        forward_func,
        model_id,
        contexts,
        answer_file,
        max_new_tokens,
        num_choices,
        **kwargs,
):
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    # Decode a sample context to check
    sample_context = tokenizer.decode([int(token) for token in contexts[0]])
    print(f"Sample context (truncated): {sample_context[:100]}...")

    # Set pad token id if needed
    if tokenizer.pad_token_id is None and hasattr(tokenizer, "eos_token_id"):
        print("Pad token ID not set. Using EOS token as pad token.")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # warmup
    for _ in range(3):
        torch.manual_seed(0)
        context_tokens = contexts[0]
        
        # Convert numpy tokens to input_ids format
        input_ids = torch.tensor([context_tokens], dtype=torch.long).to("cuda")
        
        # Create attention mask (1 for all tokens)
        attention_mask = torch.ones_like(input_ids).to("cuda")
        
        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, new_token, step, accept_length_tree = forward_func(
            {"input_ids": input_ids, "attention_mask": attention_mask},  # Include attention mask
            model,
            tokenizer,
            256,
            **kwargs,
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        # Get only the newly generated tokens
        output_ids = output_ids[0][len(input_ids[0]):]
        
        output = tokenizer.decode(
            output_ids,
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
    
    print('Warmup done')

    accept_lengths_tree = []
    for idx, context_tokens in enumerate(tqdm(contexts)):
        choices = []
        for i in range(num_choices):
            cur_accept_lengths_tree = []
            torch.manual_seed(i)
            
            # Convert numpy tokens to input_ids format
            input_ids = torch.tensor([context_tokens], dtype=torch.long).to("cuda")
            
            # Create attention mask (1 for all tokens)
            attention_mask = torch.ones_like(input_ids).to("cuda")
            
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, step, accept_length_tree = forward_func(
                    {"input_ids": input_ids, "attention_mask": attention_mask},  # Include attention mask
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

            # Store results for this choice
            choices.append({
                "index": i, 
                "completion": output, 
                "decoding_steps": int(step), 
                "new_tokens": int(new_token), 
                "wall_time": total_time,
                "accept_lengths": cur_accept_lengths_tree
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
    
    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))


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
    
    # Print dataset type statistics
    spec_bench_count = sum(1 for id_key in sorted_ids if not id_key.startswith(("article_", "context_")))
    cnn_count = sum(1 for id_key in sorted_ids if id_key.startswith("article_"))
    owt_count = sum(1 for id_key in sorted_ids if id_key.startswith("context_"))
    print(f"Dataset stats: Spec-Bench: {spec_bench_count}, CNN/DailyMail: {cnn_count}, OpenWebText: {owt_count}")

