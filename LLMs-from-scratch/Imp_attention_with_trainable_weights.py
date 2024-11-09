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

torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)

#Next compute Q K V
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

keys = inputs @ W_key
values = inputs @ W_value

print("keys shaple",keys.shape)
print("values shape", values.shape)


# and then compute the attention scores by query and each key vector
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)

# compute all attn_score
attn_scores_2 = query_2 @ keys.T

# and then compute the attention weights using the softmax function
d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
print(attn_weights_2)
#compute the context vector
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

#Implementing a compact SelfAttention class

