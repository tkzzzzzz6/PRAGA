import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

act_layer = nn.ReLU
ls_init_value = 1e-6

class CED_Graph_Improved(nn.Module):
    """
    改进的CED模块，科学设计
    """
    def __init__(self, dim, enhancement_ratio=0.1):
        super().__init__()
        
        # 使用更科学的初始化和结构
        self.norm = nn.LayerNorm(dim)  # 稳定训练
        
        # 双分支设计：保持+增强
        self.enhancement_branch = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            # nn.Dropout(0.1)  # 注释掉dropout，后续单独测试
        )
        
        # 可学习的混合权重，初始化为较小值
        self.alpha = nn.Parameter(torch.tensor(enhancement_ratio))
        
        # 科学初始化
        for layer in self.enhancement_branch:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # 标准化输入
        normed_x = self.norm(x)
        
        # 增强分支
        enhanced = self.enhancement_branch(normed_x)
        
        # 科学的残差连接
        return x + self.alpha * enhanced

class Encoder_overall_ced_3M(Module):
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, 
                 dim_in_feat_omics3, dim_out_feat_omics3, dropout=0.0, act=F.relu, data_type='Simulation'):
        super(Encoder_overall_ced_3M, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_in_feat_omics3 = dim_in_feat_omics3
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dim_out_feat_omics3 = dim_out_feat_omics3
        self.dropout = dropout
        self.act = act

        self.conv1X1_omics1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1X1_omics2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1X1_omics3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.MLP = MLP(self.dim_out_feat_omics1 * 3, self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        
        self.encoder_omics1 = Encoder_ced(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.encoder_omics2 = Encoder_ced(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)
        self.encoder_omics3 = Encoder_ced(self.dim_in_feat_omics3, self.dim_out_feat_omics3)
        self.decoder_omics3 = Decoder(self.dim_out_feat_omics3, self.dim_in_feat_omics3)
        
    def forward(self, features_omics1, features_omics2, features_omics3, 
                adj_spatial_omics1, adj_feature_omics1, 
                adj_spatial_omics2, adj_feature_omics2,
                adj_spatial_omics3, adj_feature_omics3):

        _adj_spatial_omics1 = adj_spatial_omics1.unsqueeze(0)
        _adj_feature_omics1 = adj_feature_omics1.unsqueeze(0)

        _adj_spatial_omics2 = adj_spatial_omics2.unsqueeze(0)
        _adj_feature_omics2 = adj_feature_omics2.unsqueeze(0)

        _adj_spatial_omics3 = adj_spatial_omics3.unsqueeze(0)
        _adj_feature_omics3 = adj_feature_omics3.unsqueeze(0)

        cat_adj_omics1 = torch.cat((_adj_spatial_omics1, _adj_feature_omics1), dim=0)
        cat_adj_omics2 = torch.cat((_adj_spatial_omics2, _adj_feature_omics2), dim=0)
        cat_adj_omics3 = torch.cat((_adj_spatial_omics3, _adj_feature_omics3), dim=0)

        adj_feature_omics1 = self.conv1X1_omics1(cat_adj_omics1).squeeze(0)
        adj_feature_omics2 = self.conv1X1_omics2(cat_adj_omics2).squeeze(0)
        adj_feature_omics3 = self.conv1X1_omics3(cat_adj_omics3).squeeze(0)

        feat_embeding1, emb_latent_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        feat_embeding2, emb_latent_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)
        feat_embeding3, emb_latent_omics3 = self.encoder_omics3(features_omics3, adj_feature_omics3)

        cat_emb_latent = torch.cat((emb_latent_omics1, emb_latent_omics2, emb_latent_omics3), dim=1)
        emb_latent_combined = self.MLP(cat_emb_latent)

        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)
        emb_recon_omics3 = self.decoder_omics3(emb_latent_combined, adj_spatial_omics3)

        results = {'emb_latent_omics1':emb_latent_omics1,
                   'emb_latent_omics2':emb_latent_omics2,
                   'emb_latent_omics3':emb_latent_omics3,
                   'emb_latent_combined':emb_latent_combined,
                   'emb_recon_omics1':emb_recon_omics1,
                   'emb_recon_omics2':emb_recon_omics2,
                   'emb_recon_omics3':emb_recon_omics3,
                   }
        
        return results     

class Encoder_ced(Module): 
    """
    Enhanced Encoder with CED module integration
    
    Features:
    - Preserves original graph convolution operations
    - Adds CED feature enhancement after linear transformation
    - Includes residual connections for better gradient flow
    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu, ced_ratio=0.1):
        super(Encoder_ced, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        # Original linear transformation
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        # CED module for feature enhancement - Scientific version
        self.ced_module = CED_Graph_Improved(self.out_feat, enhancement_ratio=ced_ratio)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        # 1. 线性变换
        feat_embedding = torch.mm(feat, self.weight)
        
        # 2. 图卷积（核心操作）
        graph_conv_output = torch.spmm(adj, feat_embedding)
        
        # 3. CED增强（后处理）
        enhanced_output = self.ced_module(graph_conv_output)
        
        return feat_embedding, enhanced_output
    
class Decoder(Module):
    """
    Modality-specific GNN decoder.
    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        
        return x                  

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out