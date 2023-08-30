import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch._dynamo
import subprocess
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from awq.quantize.quantizer import real_quantize_model_weight
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tinychat.demo import gen_params, stream_output
from tinychat.stream_generators import StreamGenerator
from tinychat.modules import make_quant_norm, make_quant_attn, make_fused_mlp
from tinychat.utils.prompt_templates import get_prompter

torch._dynamo.config.verbose=True

use_quant = True
use_compile = False
# model_path = "" # the path of vicuna-7b model
# load_quant_path = "quant_cache/vicuna-7b-w4-g128-awq.pt"
model_path = "/home/ubuntu/models/Llama-2-7b-chat-hf"

if use_quant:
    os.makedirs("awq_cache", exist_ok=True)
    os.makedirs("quant_cache", exist_ok=True)
    subprocess.call('python -m awq.entry --model_path /home/ubuntu/models/Llama-2-7b-chat-hf/ --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_cache/Llama-2-7b-chat-hf-w4-g128.pt'.split())
    subprocess.call('python -m awq.entry --model_path /home/ubuntu/models/Llama-2-7b-chat-hf/ --w_bit 4 --q_group_size 128 --load_awq awq_cache/Llama-2-7b-chat-hf-w4-g128.pt --q_backend real --dump_quant quant_cache/Llama-2-7b-chat-hf-awq.pt'.split())
    load_quant_path = "/home/ubuntu/src/llm-awq/scripts/quant_cache/Llama-2-7b-chat-hf-awq.pt"

config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

if use_quant:
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config,
                                                        torch_dtype=torch.float16)
    q_config = {"zero_point": True, "q_group_size": 128}
    real_quantize_model_weight(
        model, w_bit=4, q_config=q_config, init_only=True)

    model = load_checkpoint_and_dispatch(
        model, load_quant_path,
        device_map="auto",
        no_split_module_classes=["LlamaDecoderLayer"]
    )
    make_quant_attn(model, "cuda:0")
    make_quant_norm(model)
    make_fused_mlp(model)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config,
                torch_dtype=torch.float16).to("cuda:0").eval()
    if use_compile:
        model = torch.compile(model)


model_prompter = get_prompter("llama", model_path)
stream_generator = StreamGenerator
count = 0
while True:
    # Get input from the user
    input_prompt = input("USER: ")
    if input_prompt == "":
        print("EXIT...")
        break
    with torch.no_grad():
        model_prompter.insert_prompt(input_prompt)
        output_stream = stream_generator(model, tokenizer, model_prompter.model_input, gen_params, device="cuda:0")
        outputs = stream_output(output_stream)
        model_prompter.update_template(outputs)
        count += 1
