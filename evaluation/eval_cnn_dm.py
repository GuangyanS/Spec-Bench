"""Generate answers with local models for CNN/DailyMail dataset.

Usage:
python3 eval_cnn_dm.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import argparse
import json
import os
import time
import torch
import numpy as np
import shortuuid

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from fastchat.utils import str_to_torch_dtype
from tqdm import tqdm
import lade
from datasets import load_dataset


def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        **kwargs,
):
    questions = load_dataset("cnn_dailymail", "3.0.0", split="validation", streaming=False)["article"][question_begin:question_end]

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
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

    ds_local_rank = int(os.getenv('LOCAL_RANK', '0'))
    
    # Warmup with first question
    if len(questions) > 0:
        question = questions[0]
        for _ in range(3):
            torch.manual_seed(0)
            prompt = f'''[INST] <<SYS>>
You are an intelligent chatbot. Answer the questions only using the following context:

{question}

Here are some rules you always follow:

- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.

<</SYS>>

Briefly summarize the given context. [/INST]
Summary: '''
            
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            conv.stop_str = "</s>"
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            
            torch.cuda.synchronize()
            start_time = time.time()
            output_ids, _, _, _ = forward_func(
                inputs,
                model,
                tokenizer,
                max_new_tokens,
                **kwargs,
            )
            torch.cuda.synchronize()
    print('Warmup done')

    accept_lengths_tree = []
    for question_idx, question in enumerate(tqdm(questions)):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template("vicuna")
            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            
            prompt = f'''[INST] <<SYS>>
You are an intelligent chatbot. Answer the questions only using the following context:

{question}

Here are some rules you always follow:

- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.

<</SYS>>

Briefly summarize the given context. [/INST]
Summary: '''
            
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            conv.stop_str = "</s>"
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            
            if len(input_ids[0]) > 2048:  # skip input len > 2048 tokens
                continue
                
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
            except RuntimeError as e:
                print("ERROR question ID:", question_idx)
                output = "ERROR"
                step = 0
                new_token = 0
                total_time = 0

            turns.append(output)
            steps.append(int(step))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            
            choices.append({"index": i, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time})

        if lade.get_device() == 0 and ds_local_rank == 0:
            # Dump answers
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": question_idx,
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_id,
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")
    
    if len(accept_lengths_tree) > 0 and lade.get_device() == 0 and ds_local_rank == 0:
        print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        default=0,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", 
        type=int, 
        default=100,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--level",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--guess",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--use-pp",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--use-tp-ds",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--use-tp",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--use-flash",
        type=int,
        default=0,
    )  
    parser.add_argument(
        "--do-sample",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--cpu_offloading", action="store_true"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    
    args = parser.parse_args()
    
    # NOTE: The implementation of model loading and forward functions would be provided
    # in the actual codebase importing this file. This script only defines the core
    # evaluation functions necessary for running the CNN/DM benchmark.
    
    print("This script provides the evaluation functions. Implementation required for:")
    print("1. Model loading")
    print("2. Forward function definition for generation")
    
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/cnndm/model_answer/{args.model_id}.jsonl"
    
    print(f"Output to {answer_file}")
    
    # Example of how to use the evaluation functions:
    # model, tokenizer = load_model(...)
    # forward_func = define_forward_func(...)
    # run_eval(
    #     model=model,
    #     tokenizer=tokenizer,
    #     forward_func=forward_func,
    #     model_id=args.model_id,
    #     question_begin=args.question_begin,
    #     question_end=args.question_end,
    #     answer_file=answer_file,
    #     max_new_tokens=args.max_new_token,
    #     num_choices=args.num_choices,
    #     num_gpus_per_model=args.num_gpus_per_model,
    #     num_gpus_total=args.num_gpus_total,
    #     max_gpu_memory=args.max_gpu_memory,
    # )
    # 
    # if is_main_process:
    #     reorg_answer_file(answer_file)
