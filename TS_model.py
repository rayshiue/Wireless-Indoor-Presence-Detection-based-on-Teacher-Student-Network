
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:52:18 2021

@author: Mint
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# import matplotlib.pyplot as plt
# import numpy as np

# class Teacher(nn.Module):
#     def __init__(self, inputGRU = 224, hiddenGRU = 128, feature = 4):
#         super(Teacher, self).__init__()
#         self.gru1 = nn.GRU(input_size = inputGRU, hidden_size = hiddenGRU, num_layers = 1, batch_first = True, bidirectional = False)
#         self.fc1 = nn.Linear(in_features = hiddenGRU, out_features = feature)
#         self.bn3 = nn.BatchNorm1d(num_features = feature)
        
#         self.gru2 = nn.GRU(input_size = inputGRU, hidden_size = hiddenGRU, num_layers = 1, batch_first = True, bidirectional = False)
#         self.fc2 = nn.Linear(in_features = hiddenGRU, out_features = feature)
#         self.bn4 = nn.BatchNorm1d(num_features = feature)
        
#         self.dropout1 = nn.Dropout(0.05)
#         self.dropout2 = nn.Dropout(0.05)
#         self.dropout3 = nn.Dropout(0.1)
#         self.lrelu = nn.LeakyReLU()
#         self.relu = nn.ReLU()
#         self.bn = nn.BatchNorm1d(num_features = feature*2)
#         self.fc = nn.Linear(in_features = feature*2, out_features = 4)
#     def forward(self, STI, AMP):
#         self.gru1.flatten_parameters()
#         self.gru2.flatten_parameters()
#         x1, h_n1 = self.gru1(STI)
#         x2, h_n2 = self.gru2(AMP)
#         x1 = x1[:,-1,:]
#         x2 = x2[:,-1,:]
#         x1 = self.dropout1(self.fc1(x1))
#         x2 = self.dropout2(self.fc2(x2))
#         x1 = self.lrelu(x1)
#         x2 = self.lrelu(x2)
#         x1 = self.bn3(x1)
#         x2 = self.bn4(x2)
#         x = torch.cat((x1, x2), dim=1)
#         x = self.bn(x)
#         x = self.dropout3(self.fc(x))
#         # print(x.shape)
#         return x

# class Student(nn.Module):
#     def __init__(self, inputGRU = 224, hiddenGRU = 128, feature = 4):
#         super(Student, self).__init__()
#         self.gru1 = nn.GRU(input_size = inputGRU, hidden_size = hiddenGRU, num_layers = 1, batch_first = True, bidirectional = False)
#         self.fc1 = nn.Linear(in_features = hiddenGRU, out_features = feature)
#         self.bn3 = nn.BatchNorm1d(num_features = feature)
        
#         self.gru2 = nn.GRU(input_size = inputGRU, hidden_size = hiddenGRU, num_layers = 1, batch_first = True, bidirectional = False)
#         self.fc2 = nn.Linear(in_features = hiddenGRU, out_features = feature)
#         self.bn4 = nn.BatchNorm1d(num_features = feature)
        
#         self.dropout1 = nn.Dropout(0.05)
#         self.dropout2 = nn.Dropout(0.05)
#         self.dropout3 = nn.Dropout(0.1)
#         self.lrelu = nn.LeakyReLU()
#         self.relu = nn.ReLU()
#         self.bn = nn.BatchNorm1d(num_features = feature*2)
#         self.fc = nn.Linear(in_features = feature*2, out_features = 4)
#     def forward(self, STI, AMP):
#         self.gru1.flatten_parameters()
#         self.gru2.flatten_parameters()
#         x1, h_n1 = self.gru1(STI)
#         x2, h_n2 = self.gru2(AMP)
#         x1 = x1[:,-1,:]
#         x2 = x2[:,-1,:]
#         x1 = self.dropout1(self.fc1(x1))
#         x2 = self.dropout2(self.fc2(x2))
#         x1 = self.lrelu(x1)
#         x2 = self.lrelu(x2)
#         x1 = self.bn3(x1)
#         x2 = self.bn4(x2)
#         x = torch.cat((x1, x2), dim=1)
#         x = self.bn(x)
#         x = self.dropout3(self.fc(x))
#         # print(x.shape)
#         return x
    
# class MultiStudent(nn.Module):
#     def __init__(self):
#         super(MultiStudent, self).__init__()
#         self.S1 = Student(inputGRU = 56, hiddenGRU = 128, feature = 64)
#         self.S2 = Student(inputGRU = 56, hiddenGRU = 128, feature = 64)
#         self.S3 = Student(inputGRU = 56, hiddenGRU = 128, feature = 64)
#         self.S4 = Student(inputGRU = 56, hiddenGRU = 128, feature = 64)
#         self.lrelu = nn.LeakyReLU()
#         self.bn = nn.BatchNorm1d(num_features = 16)
#         self.fc = nn.Linear(in_features = 16, out_features = 4)
#     def forward(self, STI, AMP):
#         STI = STI.view(STI.shape[0], 50, 4, 56)
#         AMP = AMP.view(AMP.shape[0], 50, 4, 56)
#         w1 = torch.mean(torch.var(STI[:,:,0,:],1),1)
#         w2 = torch.mean(torch.var(STI[:,:,1,:],1),1)
#         w3 = torch.mean(torch.var(STI[:,:,2,:],1),1)
#         w4 = torch.mean(torch.var(STI[:,:,3,:],1),1)
#         w = w1 + w2 + w3 + w4
#         x1 = self.S1(STI[:,:,0,:], AMP[:,:,0,:])
#         x2 = self.S2(STI[:,:,1,:], AMP[:,:,1,:])
#         x3 = self.S3(STI[:,:,2,:], AMP[:,:,2,:])
#         x4 = self.S4(STI[:,:,3,:], AMP[:,:,3,:])
#         # x = torch.cat((x1, x2, x3, x4), dim=1)
#         # x = self.lrelu(x)
#         # x = self.bn(x)
#         # x = self.fc(x)
#         return x1, x2, x3, x4, ((w-w1)/w, (w-w2)/w, (w-w3)/w, (w-w4)/w)
#%% 2


'''
class Teacher(nn.Module):
    def __init__(self, inputdim = 224, feature = 64):
        super(Teacher, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model = inputdim)
        encoderlayer1 = nn.TransformerEncoderLayer(d_model = inputdim, nhead = 4, batch_first = True, dim_feedforward = 128)
        self.transformerencoder1 = nn.TransformerEncoder(encoder_layer = encoderlayer1, num_layers = 3)
        encoderlayer2 = nn.TransformerEncoderLayer(d_model = inputdim, nhead = 4, batch_first = True, dim_feedforward = 128)
        self.transformerencoder2 = nn.TransformerEncoder(encoder_layer = encoderlayer2, num_layers = 3)
        self.fc1 = nn.Linear(in_features = inputdim, out_features = feature)
        self.bn1 = nn.BatchNorm1d(num_features = feature)
        self.fc2 = nn.Linear(in_features = inputdim, out_features = feature)
        self.bn2 = nn.BatchNorm1d(num_features = feature)        
        # self.fc3 = nn.Linear(in_features = 50, out_features = 1)
        # self.fc4 = nn.Linear(in_features = 50, out_features = 1)
        self.dropout1 = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.05)
        self.dropout = nn.Dropout(0.1)
        self.lrelu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(num_features = feature*2)
        self.fc = nn.Linear(in_features = feature*2, out_features = 4)

    def forward(self, STI, AMP):
        STI = self.pos_encoder(STI)
        AMP = self.pos_encoder(AMP)
        x1 = self.transformerencoder1(STI) #(batch, 50, 224)->(batch, 50, 224)
        x2 = self.transformerencoder2(AMP) #(batch, 50, 224)->(batch, 50, 224)
        # x1 = x1.transpose(1,2) #(batch, 224, 50)
        # x2 = x2.transpose(1,2) #(batch, 224, 50)
        # x1 = self.fc3(x1)
        # x2 = self.fc4(x2)
        # x1 = x1.squeeze(2)
        # x2 = x2.squeeze(2)
        x1 = torch.mean(x1, dim=1) #(batch, 224)
        x2 = torch.mean(x2, dim=1) #(batch, 224)
        x1 = self.dropout1(self.fc1(x1))
        x2 = self.dropout2(self.fc2(x2))
        x1 = self.lrelu(x1)
        x2 = self.lrelu(x2)
        x1 = self.bn1(x1)
        x2 = self.bn2(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn(x)
        x = self.dropout(self.fc(x))
        print("Shape:")
        x.shape
        return x

class Student(nn.Module):
    def __init__(self, inputdim = 224, feature = 64):
        super(Student, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model = inputdim)
        encoderlayer1 = nn.TransformerEncoderLayer(d_model = inputdim, nhead = 4, batch_first = True, dim_feedforward = 128)
        self.transformerencoder1 = nn.TransformerEncoder(encoder_layer = encoderlayer1, num_layers = 3)
        encoderlayer2 = nn.TransformerEncoderLayer(d_model = inputdim, nhead = 4, batch_first = True, dim_feedforward = 128)
        self.transformerencoder2 = nn.TransformerEncoder(encoder_layer = encoderlayer2, num_layers = 3)
        self.fc1 = nn.Linear(in_features = inputdim, out_features = feature)
        self.bn1 = nn.BatchNorm1d(num_features = feature)
        self.fc2 = nn.Linear(in_features = inputdim, out_features = feature)
        self.bn2 = nn.BatchNorm1d(num_features = feature)        
        # self.fc3 = nn.Linear(in_features = 50, out_features = 1)
        # self.fc4 = nn.Linear(in_features = 50, out_features = 1)
        self.dropout1 = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.05)
        self.dropout = nn.Dropout(0.1)
        self.lrelu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(num_features = feature*2)
        self.fc = nn.Linear(in_features = feature*2, out_features = 4)

    def forward(self, STI, AMP):
        STI = self.pos_encoder(STI)
        AMP = self.pos_encoder(AMP)
        x1 = self.transformerencoder1(STI) #(batch, 50, 224)->(batch, 50, 224)
        x2 = self.transformerencoder2(AMP) #(batch, 50, 224)->(batch, 50, 224)
        # x1 = x1.transpose(1,2) #(batch, 224, 50)
        # x2 = x2.transpose(1,2) #(batch, 224, 50)
        # x1 = self.fc3(x1)
        # x2 = self.fc4(x2)
        # x1 = x1.squeeze(2)
        # x2 = x2.squeeze(2)
        x1 = torch.mean(x1, dim=1) #(batch, 224)
        x2 = torch.mean(x2, dim=1) #(batch, 224)
        x1 = self.dropout1(self.fc1(x1))
        x2 = self.dropout2(self.fc2(x2))
        x1 = self.lrelu(x1)
        x2 = self.lrelu(x2)
        x1 = self.bn1(x1)
        x2 = self.bn2(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn(x)
        x = self.dropout(self.fc(x))
        print("Shape:")
        x.shape
        return x


'''
class Teacher(nn.Module):
    def __init__(self, inputdim = 224, feature = 64):
        super().__init__()
        #self.pos_encoder = PositionalEncoding(d_model = inputdim)
        #encoderlayer1 = nn.TransformerEncoderLayer(d_model = inputdim, nhead = 4, batch_first = True, dim_feedforward = 128)
        #self.transformerencoder1 = nn.TransformerEncoder(encoder_layer = encoderlayer1, num_layers = 3)
        #encoderlayer2 = nn.TransformerEncoderLayer(d_model = inputdim, nhead = 4, batch_first = True, dim_feedforward = 128)
        #self.transformerencoder2 = nn.TransformerEncoder(encoder_layer = encoderlayer2, num_layers = 3)
        
        self.dropout1 = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.05)
        self.dropout = nn.Dropout(0.1)
        #self.lrelu = nn.LeakyReLU()
        
        self.conv_s1 = nn.Conv1d(50, 100, 10, 2) #(batch_size, 50, 224)->(batch_size, 100, 108)
        self.conv_a1 = nn.Conv1d(50, 100, 10, 2) #(batch_size, 50, 224)->(batch_size, 100, 108)
        self.pool = nn.MaxPool1d(2, 2) #(batch_size, 100, 108)->(batch_size, 100, 54) , (batch_size, 100, 12)->(batch_size, 100, 6)
        self.conv_s2 = nn.Conv1d(100, 100, 10, 4) #(batch_size, 100, 54)->(batch_size, 100, 12)
        self.conv_a2 = nn.Conv1d(100, 100, 10, 4) #(batch_size, 100, 54)->(batch_size, 100, 12)
        self.fc_s1 = nn.Linear(600, 256)
        self.fc_a1 = nn.Linear(600, 256)
        self.bn_s1 = nn.BatchNorm1d(num_features = 256)
        self.bn_a1 = nn.BatchNorm1d(num_features = 256)
        
        self.fc_s2 = nn.Linear(256, 64)
        self.fc_a2 = nn.Linear(256, 64)
        self.bn_s2 = nn.BatchNorm1d(num_features = 64)
        self.bn_a2 = nn.BatchNorm1d(num_features = 64)
        #cat two here
        
        self.bn3 = nn.BatchNorm1d(num_features = 4)
        self.fc3 = nn.Linear(128, 4)
        #self.bn1 = nn.BatchNorm1d(num_features = 2)

    def forward(self, STI, AMP):
        
        #STI = self.pos_encoder(STI)
        #AMP = self.pos_encoder(AMP)
        #STI = self.transformerencoder1(STI) #(batch_size, 50, 224)->(batch_size, 50, 224)
        #AMP = self.transformerencoder2(AMP) #(batch_size, 50, 224)->(batch_size, 50, 224)
        
        STI = self.pool(F.relu(self.conv_s1(STI)))#(batch_size, 50, 224)->(batch_size, 100, 108)->(batch_size, 100, 54)
        AMP = self.pool(F.relu(self.conv_a1(AMP)))#(batch_size, 50, 224)->(batch_size, 100, 108)->(batch_size, 100, 54)
        
        STI = self.pool(F.relu(self.conv_s2(STI)))#(batch_size, 100, 54)->(batch_size, 100, 12)->(batch_size, 100, 6)
        AMP = self.pool(F.relu(self.conv_a2(AMP)))#(batch_size, 100, 54)->(batch_size, 100, 12)->(batch_size, 100, 6)
        
        STI = torch.flatten(STI, 1) # (batch_size, 100, 6)->(batch_size, 600)
        AMP = torch.flatten(AMP, 1) # (batch_size, 100, 6)->(batch_size, 600)
        STI = self.bn_s1(F.relu(self.fc_s1(STI)))
        AMP = self.bn_a1(F.relu(self.fc_a1(AMP)))
        
        STI = self.bn_s2(F.relu(self.fc_s2(STI)))
        AMP = self.bn_a2(F.relu(self.fc_a2(AMP)))
        
        x = torch.cat((STI, AMP), dim=1)
        
        x = self.bn3(self.fc3(x))
        
        
        #x = self.bn(x)
        #x = self.dropout(self.fc(x))
        return x

class Student(nn.Module):
    def __init__(self, inputdim = 224, feature = 64):
        super().__init__()
        #self.pos_encoder = PositionalEncoding(d_model = inputdim)
        #encoderlayer1 = nn.TransformerEncoderLayer(d_model = inputdim, nhead = 4, batch_first = True, dim_feedforward = 128)
        #self.transformerencoder1 = nn.TransformerEncoder(encoder_layer = encoderlayer1, num_layers = 3)
        #encoderlayer2 = nn.TransformerEncoderLayer(d_model = inputdim, nhead = 4, batch_first = True, dim_feedforward = 128)
        #self.transformerencoder2 = nn.TransformerEncoder(encoder_layer = encoderlayer2, num_layers = 3)
        
        self.dropout1 = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.05)
        self.dropout = nn.Dropout(0.1)
        #self.lrelu = nn.LeakyReLU()
        
        self.conv_s1 = nn.Conv1d(50, 100, 10, 2) #(batch_size, 50, 224)->(batch_size, 100, 108)
        self.conv_a1 = nn.Conv1d(50, 100, 10, 2) #(batch_size, 50, 224)->(batch_size, 100, 108)
        self.pool = nn.MaxPool1d(2, 2) #(batch_size, 100, 108)->(batch_size, 100, 54) , (batch_size, 100, 12)->(batch_size, 100, 6)
        self.conv_s2 = nn.Conv1d(100, 100, 10, 4) #(batch_size, 100, 54)->(batch_size, 100, 12)
        self.conv_a2 = nn.Conv1d(100, 100, 10, 4) #(batch_size, 100, 54)->(batch_size, 100, 12)
        self.fc_s1 = nn.Linear(600, 128)
        self.fc_a1 = nn.Linear(600, 128)
        self.bn_s1 = nn.BatchNorm1d(num_features = 128)
        self.bn_a1 = nn.BatchNorm1d(num_features = 128)
        
        self.fc_s2 = nn.Linear(128, 16)
        self.fc_a2 = nn.Linear(128, 16)
        self.bn_s2 = nn.BatchNorm1d(num_features = 16)
        self.bn_a2 = nn.BatchNorm1d(num_features = 16)
        #cat  two here
        self.bn3 = nn.BatchNorm1d(num_features = 4)
        self.fc3 = nn.Linear(32, 4)

        
        #self.bn1 = nn.BatchNorm1d(num_features = 2)

    def forward(self, STI, AMP):
        
        #STI = self.pos_encoder(STI)
        #AMP = self.pos_encoder(AMP)
        #STI = self.transformerencoder1(STI) #(batch_size, 50, 224)->(batch_size, 50, 224)
        #AMP = self.transformerencoder2(AMP) #(batch_size, 50, 224)->(batch_size, 50, 224)
        
        STI = self.pool(F.relu(self.conv_s1(STI)))#(batch_size, 50, 224)->(batch_size, 100, 108)->(batch_size, 100, 54)
        AMP = self.pool(F.relu(self.conv_a1(AMP)))#(batch_size, 50, 224)->(batch_size, 100, 108)->(batch_size, 100, 54)
        
        STI = self.pool(F.relu(self.conv_s2(STI)))#(batch_size, 100, 54)->(batch_size, 100, 12)->(batch_size, 100, 6)
        AMP = self.pool(F.relu(self.conv_a2(AMP)))#(batch_size, 100, 54)->(batch_size, 100, 12)->(batch_size, 100, 6)
        
        STI = torch.flatten(STI, 1) # (batch_size, 100, 6)->(batch_size, 600)
        AMP = torch.flatten(AMP, 1) # (batch_size, 100, 6)->(batch_size, 600)
        STI = self.bn_s1(F.relu(self.fc_s1(STI)))
        AMP = self.bn_a1(F.relu(self.fc_a1(AMP)))
        
        STI = self.bn_s2(F.relu(self.fc_s2(STI)))
        AMP = self.bn_a2(F.relu(self.fc_a2(AMP)))
        
        x = torch.cat((STI, AMP), dim=1)
        
        x = self.bn3(self.fc3(x))
     
        #x = self.bn(x)
        #x = self.dropout(self.fc(x))
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)
    
class MultiStudent(nn.Module):
    def __init__(self):
        super(MultiStudent, self).__init__()
        self.S1 = Student(inputdim = 56, feature = 64)
        self.S2 = Student(inputdim = 56, feature = 64)
        self.S3 = Student(inputdim = 56, feature = 64)
        self.S4 = Student(inputdim = 56, feature = 64)
        self.lrelu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(num_features = 16)
        self.fc = nn.Linear(in_features = 16, out_features = 4)
    def forward(self, STI, AMP):
        STI = STI.view(STI.shape[0], 50, 4, 56)
        AMP = AMP.view(AMP.shape[0], 50, 4, 56)
        w1 = torch.mean(torch.var(STI[:,:,0,:],1),1)
        w2 = torch.mean(torch.var(STI[:,:,1,:],1),1)
        w3 = torch.mean(torch.var(STI[:,:,2,:],1),1)
        w4 = torch.mean(torch.var(STI[:,:,3,:],1),1)
        w = w1 + w2 + w3 + w4
        x1 = self.S1(STI[:,:,0,:], AMP[:,:,0,:])
        x2 = self.S2(STI[:,:,1,:], AMP[:,:,1,:])
        x3 = self.S3(STI[:,:,2,:], AMP[:,:,2,:])
        x4 = self.S4(STI[:,:,3,:], AMP[:,:,3,:])
        # x = torch.cat((x1, x2, x3, x4), dim=1)
        # x = self.lrelu(x)
        # x = self.bn(x)
        # x = self.fc(x)
        return x1, x2, x3, x4, ((w-w1)/w, (w-w2)/w, (w-w3)/w, (w-w4)/w)

#%% 2
# class Teacher(nn.Module):
#     def __init__(self, inputdim = 448, feature = 448):
#         super(Teacher, self).__init__()
#         self.pos_encoder = PositionalEncoding(d_model = inputdim)
#         encoderlayer = nn.TransformerEncoderLayer(d_model = inputdim, nhead = 4, batch_first = True, dim_feedforward = 128)
#         self.transformerencoder = nn.TransformerEncoder(encoder_layer = encoderlayer, num_layers = 3)    
#         self.dropout = nn.Dropout(0.1)
#         self.bn = nn.BatchNorm1d(num_features = feature)
#         self.fc = nn.Linear(in_features = feature, out_features = 4)
#     def forward(self, input):
#         x = self.pos_encoder(input)
#         x = self.transformerencoder(x) #(batch, 50, 448)->(batch, 50, 448)
#         x = torch.mean(x, dim=1) #(batch, 448)
#         # x = self.bn(x)
#         x = self.dropout(self.fc(x))
#         return x

# class Student(nn.Module):
#     def __init__(self, inputdim = 448, feature = 448):
#         super(Student, self).__init__()
#         self.pos_encoder = PositionalEncoding(d_model = inputdim)
#         encoderlayer = nn.TransformerEncoderLayer(d_model = inputdim, nhead = 4, batch_first = True, dim_feedforward = 128)
#         self.transformerencoder = nn.TransformerEncoder(encoder_layer = encoderlayer, num_layers = 3)    
#         self.dropout = nn.Dropout(0.1)
#         self.bn = nn.BatchNorm1d(num_features = feature)
#         self.fc = nn.Linear(in_features = feature, out_features = 4)
#     def forward(self, input):
#         x = self.pos_encoder(input)
#         x = self.transformerencoder(x) #(batch, 50, 448)->(batch, 50, 448)
#         x = torch.mean(x, dim=1) #(batch, 448)
#         # x = self.bn(x)
#         x = self.dropout(self.fc(x))
#         return x
    
# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         pe = pe.transpose(0,1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:,:x.size(1)]
#         return self.dropout(x)    
    
#%% 3
# class Teacher(nn.Module):
#     def __init__(self):
#         super(Teacher, self).__init__()
#         self.gru = nn.GRU(input_size = 448, hidden_size = 32, num_layers = 1, batch_first = True, bidirectional = False)
#         self.bn = nn.BatchNorm1d(num_features = 50)
#         self.dropout = nn.Dropout(0.1)
#         self.fc = nn.Linear(in_features = 1600, out_features = 4)
        
#     def forward(self, input):
#         self.gru.flatten_parameters()
#         x, h_n = self.gru(input)
#         x = self.bn(x)
#         x = torch.flatten(x, start_dim = 1)
#         x = self.dropout(self.fc(x))
#         return x

# class Student(nn.Module):
#     def __init__(self):
#         super(Student, self).__init__()
#         self.gru = nn.GRU(input_size = 448, hidden_size = 32, num_layers = 1, batch_first = True, bidirectional = False)
#         self.bn = nn.BatchNorm1d(num_features = 50)
#         self.dropout = nn.Dropout(0.1)
#         self.fc = nn.Linear(in_features = 1600, out_features = 4)
#     def forward(self, input):
#         self.gru.flatten_parameters()
#         x, h_n = self.gru(input)
#         x = self.bn(x)
#         x = torch.flatten(x, start_dim = 1)
#         x = self.dropout(self.fc(x))
#         return x

#%%
# class Teacher(nn.Module):
#     def __init__(self, inputGRU = 224, hiddenGRU = 32):
#         super(Teacher, self).__init__()
#         self.gru1 = nn.GRU(input_size = inputGRU, hidden_size = hiddenGRU, num_layers = 1, batch_first = True, bidirectional = False)
#         self.bn1 = nn.BatchNorm1d(num_features = 1600)
#         self.fc1 = nn.Linear(in_features = 1600, out_features = 256)
#         self.bn3 = nn.BatchNorm1d(num_features = 256)
        
#         self.gru2 = nn.GRU(input_size = inputGRU, hidden_size = hiddenGRU, num_layers = 1, batch_first = True, bidirectional = False)
#         self.bn2 = nn.BatchNorm1d(num_features = 1600)
#         self.fc2 = nn.Linear(in_features = 1600, out_features = 256)
        
#         self.bn4 = nn.BatchNorm1d(num_features = 256)
        
#         self.dropout1 = nn.Dropout(0.05)
#         self.dropout2 = nn.Dropout(0.05)
#         self.dropout3 = nn.Dropout(0.1)
#         self.lrelu = nn.LeakyReLU()
#         self.bn = nn.BatchNorm1d(num_features = 512)
#         self.fc = nn.Linear(in_features = 512, out_features = 4)
#     def forward(self, STI, AMP):
#         self.gru1.flatten_parameters()
#         self.gru2.flatten_parameters()
#         x1, h_n1 = self.gru1(STI)
#         x2, h_n2 = self.gru2(AMP)
#         x1 = torch.flatten(x1, start_dim = 1)
#         x2 = torch.flatten(x2, start_dim = 1)
#         x1 = self.bn1(x1)
#         x2 = self.bn2(x2)
#         x1 = self.dropout1(self.fc1(x1))
#         x2 = self.dropout2(self.fc2(x2))
#         x1 = self.lrelu(x1)
#         x2 = self.lrelu(x2)
#         x1 = self.bn3(x1)
#         x2 = self.bn4(x2)
#         x = torch.cat((x1, x2), dim=1)
#         x = self.bn(x)
#         x = self.dropout3(self.fc(x))
#         return x

# class Student(nn.Module):
#     def __init__(self, inputGRU = 224, hiddenGRU = 32):
#         super(Student, self).__init__()
#         self.gru1 = nn.GRU(input_size = inputGRU, hidden_size = hiddenGRU, num_layers = 1, batch_first = True, bidirectional = False)
#         self.bn1 = nn.BatchNorm1d(num_features = 1600)
#         self.fc1 = nn.Linear(in_features = 1600, out_features = 256)
#         self.bn3 = nn.BatchNorm1d(num_features = 256)
        
#         self.gru2 = nn.GRU(input_size = inputGRU, hidden_size = hiddenGRU, num_layers = 1, batch_first = True, bidirectional = False)
#         self.bn2 = nn.BatchNorm1d(num_features = 1600)
#         self.fc2 = nn.Linear(in_features = 1600, out_features = 256)
#         self.bn4 = nn.BatchNorm1d(num_features = 256)
        
#         self.dropout1 = nn.Dropout(0.05)
#         self.dropout2 = nn.Dropout(0.05)
#         self.dropout3 = nn.Dropout(0.1)
#         self.lrelu = nn.LeakyReLU()
#         self.bn = nn.BatchNorm1d(num_features = 512)
#         self.fc = nn.Linear(in_features = 512, out_features = 4)
#     def forward(self, STI, AMP):
#         self.gru1.flatten_parameters()
#         self.gru2.flatten_parameters()
#         x1, h_n1 = self.gru1(STI)
#         x2, h_n2 = self.gru2(AMP)
#         x1 = torch.flatten(x1, start_dim = 1)
#         x2 = torch.flatten(x2, start_dim = 1)
#         x1 = self.bn1(x1)
#         x2 = self.bn2(x2)
#         x1 = self.dropout1(self.fc1(x1))
#         x2 = self.dropout2(self.fc2(x2))
#         x1 = self.lrelu(x1)
#         x2 = self.lrelu(x2)
#         x1 = self.bn3(x1)
#         x2 = self.bn4(x2)
#         x = torch.cat((x1, x2), dim=1)
#         x = self.bn(x)
#         x = self.dropout3(self.fc(x))
#         return x
    
# class MultiStudent(nn.Module):
#     def __init__(self):
#         super(MultiStudent, self).__init__()
#         self.S1 = Student(inputGRU = 56)
#         self.S2 = Student(inputGRU = 56)
#         self.S3 = Student(inputGRU = 56)
#         self.S4 = Student(inputGRU = 56)
#         self.lrelu = nn.LeakyReLU()
#         self.bn = nn.BatchNorm1d(num_features = 16)
#         self.fc = nn.Linear(in_features = 16, out_features = 4)
#     def forward(self, STI, AMP):
#         STI = STI.view(STI.shape[0], 50, 4, 56)
#         AMP = AMP.view(AMP.shape[0], 50, 4, 56)
#         x1 = self.S1(STI[:,:,0,:], AMP[:,:,0,:])
#         x2 = self.S2(STI[:,:,1,:], AMP[:,:,1,:])
#         x3 = self.S3(STI[:,:,2,:], AMP[:,:,2,:])
#         x4 = self.S4(STI[:,:,3,:], AMP[:,:,3,:])
#         x = torch.cat((x1, x2, x3, x4), dim=1)
#         x = self.lrelu(x)
#         x = self.bn(x)
#         x = self.fc(x)
#         return x