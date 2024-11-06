import math
import torch
import torch.nn as nn

import warnings

warnings.filterwarnings(action="ignore")


class MultiHeadAttention(nn.Module):
    def __init__(self,hidden_dim, number_head) ->None:
        super().__init__()
        
        self.number_head = number_head
        self.hidden_dim = hidden_dim
        self.head_dim = number_head // hidden_dim
        
        
        
        #三个Q K V
        self.query_proj = nn.Linear(hidden_dim,hidden_dim)
        self.Key_proj = nn.Linear(hidden_dim,hidden_dim)
        self.Value_proj = nn.Linear(hidden_dim,hidden_dim)
        self.att_drop = nn.Dropout(0.1)
        
        self.output_proj = nn.Linear(hidden_dim,hidden_dim)
        
    def forward(self,X,attention_mask=None):
        batch_size,seq_len,_ = X.size()
        
        
        Q = self.query_proj(X)
        K = self.query_proj(X)
        V = self.query_proj(X)
        
        #把shape 维度进行变换（batch_size, num_head, seq_len, head_dim）
        Q_state = Q.view(batch_size,seq_len,self.number_head,self.head_dim).permute(
            0,2,1,2
        )
        K_state = K.view(batch_size,seq_len,self.number_head,self.head_dim).permute(
            1,2
        )
        V_state = V.view(batch_size,seq_len,self.number_head,self.head_dim).permute(
            1,2
        )
        attention_weight = (
            Q_state @ K_state.transpose(-1,-2) / math.sqrt(self.head_dim)
        )
        
        att_weight = Q@K.transpose(-1,-2) / math.sqrt(self.dim)
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0, float("-1e20")
            )
        #第四个维度进行softmax
        att_weight = torch.softmax(att_weight, dim=3)
        print(att_weight)
        
        attention_weight = self.att_drop(att_weight)
        output_mid = output_mid.transpose(1,2).contigous()
        
        #变成 batch seq hidden_dim
        output = output_mid.view(batch_size,seq_len,-1)
        output = self.output_proj(output)
        return output
    
    
attention_mask = (
        torch.tensor([
            [0,1],
            [0,0],
            [1,0]
        ])
    .unsqueeze(1)
    .unsqueeze(2)
    .expand(3, 8, 2, 2)
    )
    
    
x = torch.rand(3, 2, 128)
net = MultiHeadAttention(128, 8)
net(x, attention_mask).shape
    
    