#!/usr/bin/env python
# Llama2-7B + LoRA 推理, 输出 test_predictions.json 格式(兼容当前 evaluate.py)
import os, sys, json, torch
from peft import PeftModel
from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
)
from utils.prompter import Prompter

def main():
    base_model = sys.argv[1]
    lora_weights = sys.argv[2]
    test_json = sys.argv[3]
    save_path = sys.argv[4]

    device = "cuda"
    prompter = Prompter("alpaca")
    print(f"[1/4] 加载 tokenizer: {base_model}")
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    print(f"[2/4] 加载模型 (FP16): {base_model}")
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"[3/4] 加载 LoRA: {lora_weights}")
    model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()

    data = json.load(open(test_json))
    print(f"[4/4] 推理 {len(data)} 条 (max_new_tokens=128, beam=4)")
    results = []
    for i, item in enumerate(data):
        prompt = prompter.generate_prompt(item.get("instruction", ""), item.get("input", ""))
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                generation_config=GenerationConfig(
                    temperature=0.1, top_p=0.75, top_k=40, num_beams=4
                ),
                max_new_tokens=128,
                return_dict_in_generate=True,
                output_scores=True,
            )
        decoded = tokenizer.decode(output.sequences[0])
        pred = prompter.get_response(decoded)
        results.append({
            "user_id": item.get("user_id"),
            "prediction": pred,
            "ground_truth": item.get("output", ""),
        })
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(data)}")
            json.dump(results, open(save_path, "w"), ensure_ascii=False)
    json.dump(results, open(save_path, "w"), ensure_ascii=False)
    print(f"完成: {len(results)} 条 -> {save_path}")

if __name__ == "__main__":
    main()
