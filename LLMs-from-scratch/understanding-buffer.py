import torch
import torch.nn as nn

class CausalAttentionWithoutBuffers(nn.Module):

    def __init__(self, d_in,d_out,context_length, dropout,qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.mask = torch.triu(torch.ones(context_length,context_length),diagonal=1)
    
    def forward(self,x):
        b,num_tokens,d_in = x.shape
        keys= self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens,:num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
    
class CausalAttentionWithBuffers(nn.Module):

    def __init__(self, d_in,d_out,context_length, dropout,qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        #self.mask = torch.triu(torch.ones(context_length,context_length),diagonal=1)
        self.register_buffer("mask",torch.triu(torch.ones(context_length,context_length),diagonal=1))
    def forward(self,x):
        b,num_tokens,d_in = x.shape
        keys= self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens,:num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
inputs = torch.tensor(
    [[0.43,0.15,0.89],
     [0.55,0.87,0.66],
     [0.57,0.85,0.64],
     [0.22,0.58,0.33],
     [0.77,0.25,0.10],
     [0.05,0.80,0.55]]
    )
batch = torch.stack((inputs,inputs),dim=0)
context_length = batch.shape[1]
d_in = inputs.shape[1]
d_out = 2

ca_without_buffer = CausalAttentionWithoutBuffers(d_in,d_out,context_length,0.0)

with torch.no_grad():
    context_vecs = ca_without_buffer(batch)
print(context_vecs)


print("Machine has GPU:",torch.cuda.is_available())
batch  = batch.to("cuda")
ca_without_buffer.to("cuda")


print("W_query.device",ca_without_buffer.W_query.weight.device)
print("mask.device:",ca_without_buffer.mask.device)

#manually move mask to cuda
ca_without_buffer.mask = ca_without_buffer.mask.to("cuda")
print("mask.device:",ca_without_buffer.mask.device)


with torch.no_grad():
    context_vecs = ca_without_buffer(batch)

print(context_vecs)

print(ca_without_buffer.state_dict())

ca_with_buffer = CausalAttentionWithBuffers(d_in,d_out,context_length,0.0)

print(ca_with_buffer.state_dict())

ca_with_buffer.mask[ca_with_buffer.mask == 1.] = 2.
print(ca_with_buffer.state_dict())

torch.save(ca_with_buffer.state_dict(),"model.pth")
new_ca_with_buffer = CausalAttentionWithBuffers(d_in,d_out,context_length,0.0)
new_ca_with_buffer.load_state_dict(torch.load("model.pth"))
new_ca_with_buffer.mask