import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, args, bert_model):
        super().__init__()
        self.bert_model = bert_model
        self.linear = nn.Linear(bert_model.config.hidden_size + 134, 3)
        self.batch_norm = nn.BatchNorm1d(134)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        nn.init.constant_(self.linear.bias, 0.)
                    
    def forward(self, text_feat, numerical_feat):
        # bert_output = self.bert_model(**text_feat)[1]
        bert_output = self.bert_model(**text_feat)[0].mean(1)
        numerical_feat = self.batch_norm(numerical_feat)
        output = self.linear(torch.cat([bert_output, numerical_feat], dim=1))

        return output