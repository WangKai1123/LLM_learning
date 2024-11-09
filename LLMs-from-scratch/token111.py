import os
import re
import urllib.request
from torch.utils.data import Dataset, DataLoader
import torch
import tiktoken


if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "~/LLM_learning/LLMs-from-scratch/the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)
    
    
with open("/home/wk/LLM_learning/LLMs-from-scratch/the-verdict.txt","r",encoding="utf-8") as f:
       raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])

text = "Hello, word, This, is a test"
result  = re.split(r'(\s)', text)




class GPTDatesetV1(Dataset):
       def __init__(self,txt, tokenizer, max_length,stride) -> None:
              self.input_ids=[]
              self.target_dis = []
              
              #Tokenize
              token_ids = tokenizer.encode(txt,allowed_special={"<|endoftext|>"})
              #Use a sliding window to chunk the book into overlapping sequences of max_length
              for i in range(0,len(token_ids)-max_length,stride):
                     input_chunk = token_ids[i:i+max_length]
                     target_chunk = token_ids[i+1:i+max_length+1]
                     self.input_ids.append(torch.tensor(input_chunk))
                     self.target_dis.append(torch.tensor(target_chunk))
       
       def __len__(self):
              return len(self.input_ids)
       
       def __getitem__(self, index):
              return self.input_ids[index], self.target_dis[index]
       
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle = True, drop_last = True, num_workers=0):
       
       #initialize the tokenizer
       tokenizer = tiktoken.get_encoding("gpt2")
       
       #create dataset
       dataset = GPTDatesetV1(txt,tokenizer,max_length,stride)
       
       #Create dataloader
       dataloader = DataLoader(
              dataset,
              batch_size=batch_size,
              shuffle =shuffle,
              drop_last=drop_last,
              num_workers=num_workers
       )
       return dataloader


dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)

dataloader = create_dataloader_v1(
       raw_text,batch_size=8,max_length=4,stride=1,shuffle=False
)
data_iter = iter(dataloader)
inputs,targets = next(data_iter)
print("Inputs:/n", inputs)
print("\n Targets:\n",targets)


input_ids = torch.tensor([2,3,5,1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
print(embedding_layer.weight)


#========intilizer embedding layer
vocab_size = 50257
output_dim = 256
token_embedding_layer  =torch.nn.Embedding(vocab_size,output_dim)
max_length = 4
dataloader = create_dataloader_v1(
       raw_text,batch_size=8,max_length=max_length,stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs,targets = next(data_iter)
print("Token IDs:\n",inputs)
print("\nInputs shape:\n",inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

#GPT-2 uses absolute position embedding
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length,output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)