import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

act_layer = nn.ReLU
ls_init_value = 1e-6

class CED_Graph(nn.Module):
    """
    Ultra-conservative CED module for graph data processing
    Minimal enhancement to avoid numerical instability
    """
    def __init__(self, dim, drop_path=0., **kwargs):
        super().__init__()
        
        # Single linear layer for very subtle enhancement
        self.enhancement = nn.Linear(dim, dim, bias=False)
        
        # Initialize
        nn.init.normal_(self.enhancement.weight, mean=0.0, std=0.0)
        
        # Extremely small scaling factor
        self.alpha = nn.Parameter(torch.tensor(0.01), requires_grad=True)

    def forward(self, x):
        """
        x: input tensor of shape (N, dim) where N is number of nodes
        """
        # Very subtle enhancement
        enhanced = self.enhancement(x)
        
        # Ultra-conservative residual connection
        output = x + self.alpha * enhanced
        
        return output

class Encoder_overall_ced(Module):
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, dropout=0.0, act=F.relu):
        super(Encoder_overall_ced, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act

        self.conv1X1_omics1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1X1_omics2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.MLP = MLP(self.dim_out_feat_omics1 * 2, self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        
        self.encoder_omics1 = Encoder_ced(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.encoder_omics2 = Encoder_ced(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)
        
    def forward(self, features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1, adj_spatial_omics2, adj_feature_omics2):

        _adj_spatial_omics1 = adj_spatial_omics1.unsqueeze(0)  # shape: (1, N, N)
        _adj_feature_omics1 = adj_feature_omics1.unsqueeze(0)  # shape: (1, N, N)

        _adj_spatial_omics2 = adj_spatial_omics2.unsqueeze(0)  # shape: (1, N, N)
        _adj_feature_omics2 = adj_feature_omics2.unsqueeze(0)  # shape: (1, N, N)

        # shape: (2, N, N)
        cat_adj_omics1 = torch.cat((_adj_spatial_omics1, _adj_feature_omics1), dim=0)
        cat_adj_omics2 = torch.cat((_adj_spatial_omics2, _adj_feature_omics2), dim=0)

        adj_feature_omics1 = self.conv1X1_omics1(cat_adj_omics1).squeeze(0)
        adj_feature_omics2 = self.conv1X1_omics2(cat_adj_omics2).squeeze(0)

        feat_embeding1, emb_latent_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        feat_embeding2, emb_latent_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)

        cat_emb_latent = torch.cat((emb_latent_omics1, emb_latent_omics2), dim=1)
        emb_latent_combined = self.MLP(cat_emb_latent)

        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)

        results = {'emb_latent_omics1':emb_latent_omics1,
                   'emb_latent_omics2':emb_latent_omics2,
                   'emb_latent_combined':emb_latent_combined,
                   'emb_recon_omics1':emb_recon_omics1,
                   'emb_recon_omics2':emb_recon_omics2,
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
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu, drop_path=0.05):
        super(Encoder_ced, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        # Original linear transformation
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        # CED module for feature enhancement - Very conservative version
        self.ced_module = CED_Graph(self.out_feat, drop_path=drop_path)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        # Original linear transformation
        feat_embeding = torch.mm(feat, self.weight)
        
        # Apply CED feature enhancement (currently identity)
        feat_embeding_enhanced = self.ced_module(feat_embeding)
        
        # Graph convolution with enhanced features
        x = torch.spmm(adj, feat_embeding_enhanced)
        
        return feat_embeding_enhanced, x
    
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