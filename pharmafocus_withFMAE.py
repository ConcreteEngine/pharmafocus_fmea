from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline

# Load the BioASQ Dataset from the specified source
dataset = load_dataset("kroshan/BioASQ")

# Load the BioGPT tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")

# Preprocess and tokenize the dataset
def preprocess_and_tokenize(examples):
    tokenized_inputs = tokenizer(examples['question'], padding="max_length", truncation=True, max_length=128)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

# Apply preprocessing and tokenization
tokenized_dataset = dataset.map(preprocess_and_tokenize, batched=True)

# Define a Data Collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Load the BioGPT model
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    no_cuda=True
)

# Initialize the Trainer with the Data Collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'] if 'validation' in tokenized_dataset else None,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Generate Answers Using the Trained Model
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

# Refined prompt

#prompt = "Doxycycline is an antibiotic that works by"
prompt = "Li–Fraumeni syndrome's pattern of inheritance is "
# Generate an answer
generated_text = text_generator(prompt, max_length=100)[0]['generated_text']

# FMEA Check Function
def fmea_check(text):
    failure_modes = {
        "repetition": False,
        "incomplete_sentence": False,
        "nonsensical_content": False,
        "irrelevant_content": False,
    }
    
    # Check for repetition (simple check for repeated phrases)
    if len(set(text.split())) < len(text.split()) * 0.8:  # If more than 20% of words are repeated
        failure_modes["repetition"] = True
    
    # Check for incomplete sentence (no ending punctuation)
    if not text.endswith('.'):
        failure_modes["incomplete_sentence"] = True
    
    # Check for nonsensical content (very basic check for gibberish)
    if "xxx" in text or "zzzz" in text:  # Placeholder for more sophisticated checks
        failure_modes["nonsensical_content"] = True
    
    # Check for irrelevant content (basic keyword check)
    if "bacteria" not in text and "antibiotic" not in text:
        failure_modes["irrelevant_content"] = True
    
    # Generate a report
    report = f"FMEA Check Report:\n"
    for mode, detected in failure_modes.items():
        report += f"- {mode.replace('_', ' ').title()}: {'Detected' if detected else 'Not Detected'}\n"
    
    return report

# Post-processing and FMEA Check
def clean_and_evaluate_generated_text(text):
    # Clean the text (removing incomplete words, if necessary)
    last_period = text.rfind(".")
    if last_period != -1:
        text = text[:last_period + 1]
    cleaned_text = text.strip()
    
    # Run FMEA check
    fmea_report = fmea_check(cleaned_text)
    
    return cleaned_text, fmea_report

# Apply post-processing and FMEA check
cleaned_text, fmea_report = clean_and_evaluate_generated_text(generated_text)

# Output the results
print("Generated Answer:", cleaned_text)
print(fmea_report)
