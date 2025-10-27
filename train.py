import json
import torch
import re,math
import random
import numpy as np
import pandas as pd
from datasets import Dataset,DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login



seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

model_name = "meta-llama/Llama-3.2-1B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

def preprocess(text):
  text = re.sub(r'\s+',' ',text)
  text = re.sub(r'[\U0001F600-\U0001F64F|\U0001F300-\U0001F5FF|\U0001F680-\U0001F6FF|\U0001F700-\U0001F77F|\U0001F780-\U0001F7FF]',' ',text)
  text = text.replace(r'\n',' ')
  text = text.replace(r'\t',' ')
  text = re.sub(r'#+','',text)
  text = re.sub(r'http\S+|@\w+','',text)
  text = re.sub(r'\s+', ' ', text)
  text = text.strip()
  return text

def extract_json(text):
    text = text.strip()
    idx = text.find('{')
    if idx == -1:
        return None
    count = 0
    in_string = False
    esc = False
    for i in range(idx, len(text)):
        char = text[i]
        if char == '"' and not esc:
            in_string = not in_string
        elif char == '\\' and in_string:
            esc = not esc
            continue
        else:
            esc = False
        if not in_string:
            if char == '{':
                count += 1
            elif char == '}':
                count -= 1
                if count == 0:
                    json_str = text[idx:i+1]
                    return json.loads(json_str)
    return None


def train_model(train_path,dev_path,val_path,hf_token,prediction_file=True,merge_model=False,lora=True,lora_path="./lora"):
    
    login(token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train = pd.read_json(train_path, lines=True)
    dev = pd.read_json(dev_path, lines=True)
    val = pd.read_json(val_path, lines=True)
    train['input'] = train['input'].apply(preprocess)
    dev['input'] = dev['input'].apply(preprocess)
    val['input'] = val['input'].apply(preprocess)
    trainds = Dataset.from_pandas(train, preserve_index=False)
    valds = Dataset.from_pandas(val, preserve_index=False)
    devds = Dataset.from_pandas(dev, preserve_index=False)

    dataset = DatasetDict({
    "train": trainds,
    "val": valds,
    "dev":devds
    })

    tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {{ message['content'] }}<|eot_id|>{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>
    {{ message['content'] }}<|eot_id|>{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>
    {{ message['content'] }}<|eot_id|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>
    {% endif %}"""

    special_tokens = {
        'pad_token' : "<|pad|>",
        'additional_special_tokens': [
            '<|begin_of_text|>',
            '<|start_header_id|>',
            '<|end_header_id|>',
            '<|eot_id|>',
        ]
    }

    num_tokenizers_added = tokenizer.add_special_tokens(special_tokens)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer)) # this one for the model so it will not throw error cause we have increased the size of our vocalbulary
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n"

    def chat_template(row):
        completion = json.dumps(row["output"], ensure_ascii=False)
        messages = [
            {"role": "system", "content": "You are an JSON analyst. Always respond with valid JSON only."},
            {"role": "user", "content": row["input"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "completion": completion}

    dataset = dataset.map(chat_template)

    lora_config = LoraConfig(
        r=128,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
        lora_dropout=0.07,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    sft_config = SFTConfig(
        output_dir="./finetuned",
        num_train_epochs=3,
        max_steps=100,
        seed=seed,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_steps=10,
        eval_strategy="steps",
        save_steps=20,
        save_strategy='steps',
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=2,
        fp16=True,
        gradient_accumulation_steps=4,
        logging_steps=10,
        report_to="none",
        optim="adamw_8bit",
        dataset_text_field="text",
        packing=False,
        max_length=512,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
    )

    trainer.train()

    base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map='auto'
    )

    def zero_prompting_response(text, tokenizer, model):
        text = preprocess(text)
        messages = [
            {"role": "system", "content": "You are an JSON analyst. Always respond with valid JSON only."},
            {"role": "user", "content": text}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        eos_token_id = tokenizer.eos_token_id
        eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[eos_token_id, eot_token_id],
                repetition_penalty=1.2,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        return response

    text = "Drivers' Assoc., Housing Rights Group ne rural-6 me sabha announce kiyya, tmrw evening. traffic hoga! #local âš ï¸"

    # Baseline
    response_baseline = zero_prompting_response(text, tokenizer, base_model)
    # Fine tuned model
    response_finetuned = zero_prompting_response(text, tokenizer, model)

    print("Input:", text)
    print("Baseline response:", response_baseline)
    print('*'*20)
    print(' ')
    print("Fine-tuned response Without Extracting Json:", response_finetuned)
    print(" ")
    print("Fine-tuned response With Extracting Json:")
    json.dumps(extract_json(response_finetuned))

    if lora:
        lora = "lora_adapters"
        model.save_pretrained(lora)
        tokenizer.save_pretrained(lora)

    if prediction_file:
        output_data = []
        for idx,row in enumerate(dataset['val']):
            if idx == 12:
                break

        text = preprocess(row["input"])

        messages = [
            {"role": "system", "content": "You are an JSON analyst. Always respond with valid JSON only."},
            {"role": "user", "content": text}
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        eos_token_id = tokenizer.eos_token_id
        eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[eos_token_id, eot_token_id],
                repetition_penalty=1.2,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        json_output = extract_json(response)

        output_data.append({
        "input": row["input"],
        "output": json_output
        })
        print(json_output)

        with open('predictions.json', 'w') as f:
            json.dump(output_data, f, indent=2)

        print("Predictions saved")

    if merge_model:
        base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
        )
        base_model = PeftModel.from_pretrained(base_model, lora)
        base_model = base_model.merge_and_unload()
        merge_dir = "./merged_model"
        base_model.save_pretrained(merge_dir, safe_serialization=True)
        tokenizer.save_pretrained(merge_dir)


train_model('train.json','dev.json','val.json',prediction_file=True)