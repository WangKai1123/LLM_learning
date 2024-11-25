import os
import urllib.request
import tiktoken
import torch
from model import Model
file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "~/LLM_learning/LLMs-from-scratch/the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)
    
    
with open("/home/wk/LLM_learning/LLMs-from-scratch/the-verdict.txt","r",encoding="utf-8") as f:
       text_data = f.read()

print("Total number of character:", len(text_data))

print(text_data[:99])
print(text_data[-99:])

total_characters = len(text_data)
tokenizer = tiktoken.get_encoding("gpt2")
total_tokens = len(tokenizer.encode(text_data))

print("character:",total_characters)
print("Tokens:",total_tokens)



from token111 import create_dataloader_v1

#Train/validation ratio
train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


torch.manual_seed(123)


GPT_CONFIG_124M = {
    "vocab_size":50257,
    "context_length":6,
    "emb_dim":768,
    "n_heads":12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias":False
}


train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)


# Sanity check

if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")


print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

train_tokens = 0
for input_batch, target_batch in train_loader:
     train_tokens += input_batch.numel()
     
val_tokens = 0
for input_batch, target_batch in val_loader:
     val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)


def calc_loss_batch(input_batch, target_batch,model,device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model,device,num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
         return float("nan")
    elif num_batches is None:
         num_batches = len(data_loader)
    else:
         num_batches = min(num_batches,len(data_loader))

    for i,(input_batch, target_batch) in enumerate(data_loader):
        if i< num_batches:
              loss = calc_loss_batch(input_batch, target_batch, model,device)
              total_loss += loss.item()
        else:
             break

    return total_loss / num_batches     
     
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model
model.to(device)


