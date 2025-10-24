# Problem : OSINT Micro‑Event → JSON

## Data card  :

- Mixed Dataset with English,Hindi,Hinglish words in it, short posts under 200 characters with actionable json output

**Task Schema**

{
  "event_type": "protest|rally|riot|strike|curfew|threat|rumor",
  "when": "today|tomorrow|D+N",
  "where": "city_or_district",
  "actors": ["list","of","actors"],
  "confidence": 0-1,
  "priority": 0|1|2|3,
  "rationale": "1-2 sentence why"
}

**Composition**: 
- Indian Locations , rural-X, sector-Y
- Actor Tags like : ( Student Union, Drivers’ Assoc., Farmers Front )

**Generation**:
- noise (typos, emojis, hashtags, links, phone‑#, sarcasm, off‑topic, near‑miss confusions)

**Evaluation**:
Accuracy of Fields like  Event,When,where and Mae for Confidence and Priority

example : 
{'event_type': {'accuracy': 0.9285714285714286, 'correct': 13, 'total': 14},
 'when': {'accuracy': 0.14285714285714285, 'correct': 2, 'total': 14},
 'where': {'accuracy': 1.0, 'correct': 14, 'total': 14},
 'confidence': {'mae': 0.12785714285714286, 'count': 14},
 'priority': {'mae': 0.5, 'count': 14},
 'overall': 0.6904761904761905}

**Bias & Risks**: 
Indian‑centric,Dummy placeholders

**Files**:
`train.jsonl`, `dev.jsonl`, `test.jsonl`

**Data Preparation**:

- Regex for text preprocessing
- For Formatting data into the training format used Datasets library's Dataset And DataDict
- Used DataCollatorLanguageModeling and DataLoader 

**Training** :
- Used llama3.1 1B base model
- lora with 128 rank (prjecting in this dimension) and 64 alpha parameter
- SFT Trainer for training 

**Chat Template**:
- Since base model did not come with chat template so we need to make it for our training

` """{% for message in messages %}{% if message['role'] == 'system' %}<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>
{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>
{{ message['content'] }}<|eot_id|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>
{% endif %}""" `

- After making the chat template we need to add out custom special tokens and after adding do not forget to resize model embedding matrix `model.resize_token_embeddings(len(tokenizer))` 

- we used tokenize function for tokening the dataset that will give use ` Input_ids and Attention Mask ` , so we do not need other columns now, so we remove it while tokenizing

`def tokenize_func(batch):
  return tokenizer(
      batch['text'],
      truncation=False,
      max_length=512,
      padding=False,
  )

remove_cols = list(dataset["train"].column_names)
tokenize_dataset = dataset.map(tokenize_func,batched=True,remove_columns=remove_cols)`


**Trainable Parameters** : ``trainable params: 69,206,016 || all params: 1,305,020,416 || trainable%: 5.3031``

**Training Results** : ``TrainOutput(global_step=120, training_loss=2.2029616117477415, metrics={'train_runtime': 320.6775, 'train_samples_per_second': 1.497, 'train_steps_per_second': 0.374, 'total_flos': 412196064313344.0, 'train_loss': 2.2029616117477415, 'epoch': 2.26})``

#### Thankyou