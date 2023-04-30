import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
import numpy as np

"""
    Graph Transformer Layer with edge features
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func

def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func

# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}
    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func




"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim + 2, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim + 2, out_dim * num_heads, bias=False)
    
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        
        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        
        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))
        
        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))
        
        # softmax
        g.apply_edges(exp('score'))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
    
    def forward(self, g, h, e):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(g)
        
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6)) # adding eps to all values here
        e_out = g.edata['e_out']
        
        return h_out, e_out
    

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        
        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)
        
    def forward(self, g, h, e):
        h_in1 = h # for first residual connection
        e_in1 = e # for first residual connection
        
        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(g, h, e)
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h # residual connection
            e = e_in1 + e # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
        
class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=1): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class GraphTransformerNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        self.e_n_classes = net_params['e_n_classes']
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GraphTransformerLayer(in_dim_node, hidden_dim, num_heads,
                                              dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GraphTransformerLayer(in_dim_node, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual))
        self.eMLP_layer = MLPReadout(out_dim * 3, self.e_n_classes)
        
    def apply_edges(self, edges):
        h_u = edges.src['h']
        e_w = edges.data['h']
        h_v = edges.dst['h']
        h_u = h_u.reshape(h_u.shape[0], -1)
        h_v = h_v.reshape(h_v.shape[0], -1)
        e_w = e_w.reshape(e_w.shape[0], -1)
        # e_w = F.normalize(e_w)
        score = self.eMLP_layer(torch.cat([h_u, e_w, h_v], 1))
        return {'escore': score}


    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):
        
        # GraphTransformer Layers
        for conv in self.layers:
            h, e = conv(g, h, e)
        
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['h'] = e
            g.apply_edges(self.apply_edges)
            return g.edata['escore']

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss