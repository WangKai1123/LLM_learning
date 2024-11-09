import torch


idx = torch.tensor([2,3,1])


num_idx = max(idx)+1

# The desired embedding dimension is a hyperparameter
out_dim = 5

torch.manual_seed(123)
embedding = torch.nn.Embedding(num_idx, out_dim)


print(embedding.weight)



#Using nn.Linear
onehot = torch.nn.functional.one_hot(idx)
print(onehot)
torch.manual_seed(123)
linear = torch.nn.Linear(num_idx,out_dim,bias=False)
print(linear.weight)