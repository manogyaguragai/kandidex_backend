import json
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from tqdm import tqdm
import os

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load JSONL dataset
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data

# Load and parse the dataset
all_data = load_jsonl("train.jsonl")

# Convert to InputExamples: matched = 1.0, unmatched = 0.0
input_examples = []
for entry in all_data:
    job = entry["Job-Description"]
    matched = entry["Resume-matched"]
    unmatched = entry["Resume-unmatched"]
    
    input_examples.append(InputExample(texts=[job, matched], label=1.0))
    input_examples.append(InputExample(texts=[job, unmatched], label=0.0))

# Shuffle and split
from sklearn.model_selection import train_test_split
train_samples, val_samples = train_test_split(input_examples, test_size=0.2, random_state=42)

# Define model using 'bert-mini'
word_embedding_model = models.Transformer("prajjwal1/bert-mini", max_seq_length=128)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
model.to(device)

# Prepare DataLoader and loss
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model=model)

# Setup evaluator
val_texts1 = [sample.texts[0] for sample in val_samples]
val_texts2 = [sample.texts[1] for sample in val_samples]
val_labels = [sample.label for sample in val_samples]
evaluator = BinaryClassificationEvaluator(val_texts1, val_texts2, val_labels)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=3,
    evaluation_steps=1000,
    output_path="./output_bert_mini_job_resume",
    warmup_steps=100,
    show_progress_bar=True
)