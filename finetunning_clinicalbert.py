# -*- coding: utf-8 -*-
"""fineTunning_ClinicalBERT.ipynb




"""### Fine Tunning"""

!pip install transformers[torch]



#haa_trainChronologies_string = haa_trainChronologies.to_string

print(haa_trainChronologies_string)

example=haa_trainChronologies_string

from datasets import Dataset


hf_dataset = Dataset.from_pandas(haa_develAdmittimes)

hf_haa_develAdmittimes = hf_dataset.from_pandas(haa_develAdmittimes)

hf_dataset







def tokenize_data(example):
    combined_text = f"Subject ID: {example['subject_id']} Hospital Admission ID: {example['hadm_id']} Admittime: {example['admittime']}"

    # Tokenize the text and handle padding directly, ensuring output is suitable for processing
    tokenized_output = tokenizer(combined_text, truncation=True, padding='max_length', max_length=16)

    # Return the dictionary as-is if already in list format
    return tokenized_output

from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def tokenize_data(example):
    # Create a single text string from the dataset fields
    text_to_tokenize = f"Subject ID: {example['subject_id']} Hospital Admission ID: {example['hadm_id']} Admittime: {example['admittime']} Observations: {example.get('observations', '')}"

    # Tokenize the combined text with consistent padding and truncation
    return tokenizer(
        text_to_tokenize,
        padding="max_length",   # Ensures all outputs have the same length
        truncation=True,        # Ensures no output exceeds max_length
        max_length=512          # Sets the maximum length of a sequence
    )

# Example of how to apply this function using map in the Hugging Face dataset
tokenized_dataset = hf_haa_develAdmittimes.map(
    tokenize_data,
    batched=True,
    batch_size=16,
    remove_columns=hf_haa_develAdmittimes.column_names
)

from transformers import DataCollatorWithPadding

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Initialize a data collator that dynamically pads the batches
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)  # None means it will pad dynamically to the longest in the batch

# Assuming hf_haa_develAdmittimes is correctly initialized as a dataset
# Apply tokenization to the dataset
tokenized_dataset = hf_haa_develAdmittimes.map(
    tokenize_data,
    batched=True,
    batch_size=8,
    remove_columns=hf_haa_develAdmittimes.column_names
)

tokenized_dataset



train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

pip install transformers[torch] --upgrade

pip install transformers[torch] --upgrade



from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT",
    num_labels=1  # Specify the number of labels in your classification task
)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
sample_batch = data_collator([tokenized_dataset[i] for i in range(8)])
print(sample_batch)
collated_batch = data_collator(sample_batch)
print(collated_batch)

# Diagnostic to check input shapes
def check_input_shapes(data):
    print("Shapes of input tensors:")
    print("Input IDs:", data['input_ids'].shape)
    print("Attention Mask:", data['attention_mask'].shape)
    if 'token_type_ids' in data:
        print("Token Type IDs:", data['token_type_ids'].shape)

# Apply this diagnostic function to a batch from the training dataset
sample_batch = next(iter(Trainer.get_train_dataloader(trainer)))
check_input_shapes(sample_batch)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Test the data collator on a small batch manually extracted from the dataset
example_batch = [tokenized_dataset[i] for i in range(8)]  # Adjust range as necessary
collated_batch = data_collator(example_batch)
print({k: v.shape for k, v in collated_batch.items()})


# Example of inspecting the output of one tokenized example
example = {'subject_id': '1', 'hadm_id': '100', 'admittime': '2020-01-01', 'observations': 'Patient exhibits symptoms of flu.'}
tokenized_example = tokenize_function(example)
print(tokenized_example)

# Assuming 'tokenized_datasets' is a list of tokenized examples
sample_batch = [tokenized_dataset[i] for i in range(8)]
collated_batch = data_collator(sample_batch)
print({k: v.shape for k, v in collated_batch.items()})

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

input_ids = input_ids_np.squeeze(0)
outputs = model(input_ids=input_ids,attention_mask=attention_mask)

for batch in loader:
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    print(outputs)
    break

# Manually create a batch from the tokenized dataset
sample_batch = [train_dataset[i] for i in range(8)]
collated_batch = data_collator(sample_batch)

# Print the shapes of each component
print("Collated batch shapes:")
for key, tensor in collated_batch.items():
    print(f"{key}: {tensor.shape}")

# Assuming a correct initialization of your data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Manually collate a sample batch
sample_batch = [train_dataset[i] for i in range(8)]
collated_batch = data_collator(sample_batch)

# Print the structure and content of collated batch to diagnose
print("Collated batch input_ids shape and content:", collated_batch['input_ids'].shape, collated_batch['input_ids'])

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer

# Assuming you have initialized your tokenizer already
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create a DataLoader to automatically batch and collate samples
loader = DataLoader(train_dataset, batch_size=8, collate_fn=data_collator)

# Check the first batch
#for batch in loader:
#    print("Batch 'input_ids' shape:", batch['input_ids'].shape)

print("Collated input_ids shape:", collated_batch['input_ids'].shape)

# Assuming your data loader setup from previous snippets
loader = DataLoader(train_dataset, batch_size=8, collate_fn=data_collator)

# Print detailed structure of the first few batches
for batch in loader:
    if isinstance(batch, dict):
        for key, value in batch.items():
            print(f"{key}: {value}")
            if hasattr(value, 'shape'):
                print(f"Shape of {key}: {value.shape}")
    else:
        print("Batch data type:", type(batch))
        print(batch)
    break

# Check the first few items in the dataset
for i in range(3):
    print(train_dataset[i])





from transformers import DataCollatorWithPadding

# Assuming you have a tokenizer loaded as follows
# tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Assuming 'tokenized_datasets' is a dataset or a list of such tokenized examples
# Let's simulate a batch with several examples
sample_batch = [tokenized_dataset[i] for i in range(8)]  # Collect 8 examples to form a batch
collated_batch = data_collator(sample_batch)  # Apply the data collator

# Print out the shapes of the tensors in the collated batch to verify
print({k: v.shape for k, v in collated_batch.items()})

from transformers import DataCollatorWithPadding, AutoTokenizer

# Initialize the tokenizer and the data collator
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)

# Assuming you have a list of dictionaries from tokenized datasets
# Here we simulate tokenized data for demonstration
tokenized_datasets = [{
    'input_ids': tokenizer.encode("Sample text here", add_special_tokens=True),
    'token_type_ids': [0] * len(tokenizer.encode("Sample text here", add_special_tokens=True)),
    'attention_mask': [1] * len(tokenizer.encode("Sample text here", add_special_tokens=True))
} for _ in range(8)]

# Use the data collator to turn these into a batch
collated_batch = data_collator(tokenized_datasets)
print({k: v.shape for k, v in collated_batch.items()})

print(collated_batch)

print(tokenized_datasets[0])

from transformers import Trainer, TrainingArguments

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # where to save the model files
    num_train_epochs=1,              # number of training epochs
    per_device_train_batch_size=8,   # batch size per device during training
    evaluation_strategy='steps',     # evaluation is done (and model saved) every eval_steps
    eval_steps=500,                  # number of steps to run evaluation
    save_steps=500,                  # number of steps to save the model
    warmup_steps=500,                # number of steps for the warmup phase
    weight_decay=0.01                # strength of weight decay
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,               # training arguments, defined above
    train_dataset=train_dataset,      # training dataset
    eval_dataset=eval_dataset,        # evaluation dataset
    data_collator=data_collator       # our data collator
)

# Start training
trainer.train()


from rouge_score import rouge_scorer

def rouge_scores(references, predictions):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Store the scores in a list
    scores = []

    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        scores.append(score)

    return scores


references = [
   "The timestamps for observations containing "C0392747" are as follows:

- 2104-08-05
- 2104-08-07
- 2104-08-08
- 2104-08-08
- 2104-08-09
- ...
- 2194-10-01
- 2165-04-30
- 2165-04-30
- 2165-05-02
- 2165-05-09"
]
predictions = [
    "
- 2104-08-08
- 2104-08-07
- 2104-08-08
- ...
- 2194-10-01
- 2165-04-30
- 2165-04-30
- 2165-05-02
- 2165-05-09"
    "
]

# Calculate ROUGE scores
rouge_scores = rouge_scores(references, predictions)

# Print the scores
for score in rouge_scores:
    print(score)




