import torch.nn as nn
import torch

# txt embedding results  eg"your, journey,starts,with,one,stop""
inputs = torch.tensor([
    [0.43,0.15,0.89],
    [0.55,0.87,0.66],
    [0.57,0.85,0.64],
    [0.22,0.58,0.33],
    [0.77,0.25,0.10],
    [0.05,0.80,0.55]
])

x_2 = inputs[1]
d_in = inputs.shape[1]  # the input embedding size d=3
d_out = 2   #the output emdedding size  d=2



class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self,x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ values.T    
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in,d_out)
print(sa_v1(inputs))
#================formulate 2=======================
class SelfAttention_v2(nn.Module):

    def __init__(self, d_in,d_out, qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self,x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5 ,dim=-1)

        context_vec = attn_weights @ values
        return context_vec
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in,d_out)
print(sa_v2(inputs))





queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
print(attn_weights)


context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length,context_length))
print(mask_simple)
masked_simple = attn_weights * mask_simple
print(masked_simple)

mask = torch.triu(torch.ones(context_length,context_length),diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

#Dropout
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6,6)
print(dropout(example))

torch.manual_seed(123)
print(dropout(attn_weights))

#implementing a compact causal self-attention
batch=torch.stack((inputs, inputs),dim=0)
#batch shape = [2,6,3]
class CausalAttention(nn.Module):
    def __init__(self, d_in,d_out,context_length,dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.w_query = nn.Linear(d_in, d_out,bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out,bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out,bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length,context_length)))

    def forward(self,x):
        b, num_tokens, d_in = x.shape
        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens,:num_tokens], -torch.inf
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in,d_out,context_length,0.0)
context_vecs = ca(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)




