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

query = inputs[1]


#初始化一个空向量来保存自注意力分数
atten_score_2 = torch.empty(inputs.shape[0])
#计算query和每个词的注意力分数
for i ,x_i in enumerate(inputs):
    atten_score_2[i] = torch.dot(x_i,query)

print(atten_score_2) 


res = 0
for idx, element in enumerate(input[0]):
    res += inputs[0][idx] * query[idx]


#normalization by hand
attn_weights_2_tmp = atten_score_2 / atten_score_2.sum()

#normalization by softmax
def softmax_native(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

atten_weights_2_native = softmax_native(atten_score_2)


#normalization softmax by pytorch
atten_weights_2 = torch.softmax(atten_score_2,dim=0)

#compute the contecxt vector  by multiplying the embedded input tokens with the attention weights and sum th
#resulting vector
query  =inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i ,x_i in enumerate(inputs):
    context_vec_2 += atten_weights_2[i]*x_i
print(context_vec_2)



#Generalize to all input sequence tokens
attn_scores = torch.empty(6,6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i,j] = torch.dot(x_i, x_j)
# more efficiently via matrix multiplication
attn_scores = inputs @ inputs.T
#softmax
attn_weights = torch.softmax(attn_scores, dim=-1) 

all_context_vecs = attn_weights @ inputs
