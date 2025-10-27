## OSINT Micro-Event → JSON Model

This is a fine tuned model for generating json output out of the  short, noisy “open-source posts” (tweet/WhatsApp-forward/news blurb length, incl. Hinglish/Urdu/English code-mix), , The Model outputs a single JSON describing an actionable event.

Example : 
{"event_type": "protest",
  "when": "tomorrow",
  "where": "Rural-6"
  "actors": [
    "Drivers"
  ],
  "confidence": 0.87,
  "priority": 3,
  "rationale": "Post mentions rally by Drivers in Rural-6. Time phrase maps to 'tomorrow', so label is protest",
}

## Model
- `meta-llama/Llama-3.2-1B`
- Used BitsAndBytes for loading model with 8bit

## What it does 
- Takes input text (like social media posts)
- Outputs JSON with fields: actors, confidence, event_type, priority, rationale, when, where.
- Trained on a dataset of 300 examples (train: 210, val: 45, dev: 45)

## Training
- Used SFTTrainer from Hugging Face
- LoRA adapters (r=128, alpha=64)
- Epochs: 3, but max steps 120
- Learning rate: 3e-5
- Seeded with 42 for reproducibility.
- Versions: bitsandbytes==0.48.1, transformers==4.57.1, peft==0.17.1, trl==0.24.0, datasets==4.0.0 

## Evaluation
- Slot-level F1 for event_type/where/when
- `Note : Here I have assumed false positive and false negative as same cause while predicting if model predicted protest but actually its roit so it will be false postive (model predicted positive but actually its negative ) or false negative (model predicted negative but actually its positive)`
- - Better is we see accuracy that will tell how much good model is working on this specific slot
- `Exact Match` : If any of the word does not match then it will give exact match as False. Here I am not using lower() cause if actually its Actual : And , predicted : and , in evaluation will see and:and, it will give `True` but it should be `False`
- Mae: Mean Absolute Error for confidence and priority.
- `Baseline Zero short prompting` where we are comparing it against the baseline model , `Note` : I am not using metric cause baseline model will produce the text only not the json here we will compare the response , see how model response is improved compared to baseline.

``Baseline response: `` : Drivers' Assoc., Housing Rights Group ne rural-6 me sabha announce kiyya, tmrw evening. traffic hoga! local âš ï¸<|eot_id|><|start_header_id|>assistant<|end_header_id|>
The following is a list of the most common errors that we see in our daily work and how to fix them.
This document contains information about some commonly encountered issues when using the API for developers who want to build applications against the CiviCRM system. It also includes links to other resources where you can find more detailed explanations or solutions to these problems.
If this page does not answer your question please contact us at support@civicrm.org.
CiviCRM 4.x has been released on October 1st, 2015. The release notes include details regarding changes made between versions 3.x and 4.x as well as new features added since version 2.x.
We have created two guides which will help you get started:
Getting Started Guide - This guide provides instructions for installing CiviCRM from source code (recommended) or binary packages available through CRAN, Debian/Ubuntu repositories etc..
Installing via Composer - A step-by-step tutorial showing how to install CiviCRM by composer package manager.

`` Fine-tuned response `` : 
{"actors": ["Drivers"], "confidence": 0.87, "event_type": "protest", "priority": 3, "rationale": "Post mentions rally by Drivers in Rural-6. Time phrase maps to \'tomorrow\', so label is protest", "when": "tomorrow", "where": "Rural-6"}


{'event_type': {'f1': 0.91, 'correct': 10, 'total': 11},
 'when': {'f1': 0.36, 'correct': 4, 'total': 11},
 'where': {'f1': 0.91, 'correct': 10, 'total': 11},
 'confidence': {'mae': 0.03181818190909089, 'count': 11},
 'priority': {'mae': 0.36363636363636365, 'count': 11},
 'exact_match': False}

- In Evaluation script named `eval.py`, we need to provide the true_data, Prediction_data (prediction that we will get from the training script)

# For training: 
- For training Arguments we need to provide : train_path, Dev_path (used for computing loss while training) , val_path (used to create Prediction.json for evalutation)
- Optional : prediction_file (this will create prediction.json file that we can use for evaluation, Dataset used is : Val)
- - merge_model : If true , this will merge the lora layers with the base model and give us , full model with lora layers attached
- - lora : If true this will save the lora weights
- - lora_path = to give the path where to store the lora weights