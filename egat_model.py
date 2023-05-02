import torch
import torch.nn as nn
import torch.nn.functional as F

class EGATConv(nn.Module):

    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 num_heads):
        
        super().__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_nodes = nn.Linear(in_node_feats, out_node_feats * num_heads, bias=True)
        self.fc_edges = nn.Linear(in_edge_feats + 2 * in_node_feats, out_edge_feats * num_heads, bias=False)
        self.fc_attn = nn.Linear(out_edge_feats, num_heads, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_edges.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_attn.weight, gain=gain)

    def edge_attention(self, edges):
        #extract features
        h_src = edges.src['h'].reshape(edges.src['h'].shape[0], -1)
        h_dst = edges.dst['h'].reshape(edges.dst['h'].shape[0], -1)
        f = edges.data['f'].reshape(edges.data['f'].shape[0], -1)
        #stack h_i | f_ij | h_j
        stack = torch.cat([h_src, f, h_dst], dim=-1)
        # apply FC and activation
        f_out = self.fc_edges(stack)
        f_out = F.leaky_relu(f_out)
        f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
        # apply FC to reduce edge_feats to scalar
        a = self.fc_attn(f_out).sum(-1).unsqueeze(-1)

        return {'a': F.leaky_relu(a), 'f' : f_out}

    def message_func(self, edges):
        return {'h': edges.src['h'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        alpha = nn.functional.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'h': h}

    def forward(self, graph, nfeats, efeats):
        with graph.local_scope():
        ##TODO allow node src and dst feats
            graph.edata['f'] = efeats
            graph.ndata['h'] = nfeats

            graph.apply_edges(self.edge_attention)

            nfeats_ = self.fc_nodes(nfeats)
            nfeats_ = nfeats_.view(-1, self._num_heads, self._out_node_feats)

            graph.ndata['h'] = nfeats_

            graph.update_all(message_func = self.message_func,
                         reduce_func = self.reduce_func)

            return graph.ndata.pop('h'), graph.edata.pop('f')

class EGAT(nn.Module):
    def __init__(self, args):
        super(EGAT, self).__init__()
        self.layer1 = EGATConv(args.in_node_feats, args.in_edge_feats, args.out_node_feats, args.out_edge_feats, args.num_heads)
        self.W = nn.Linear(args.out_node_feats * 3 * args.num_heads, args.classes)
    
    def apply_edges(self, edges):
        h_u = edges.src['h']
        e_w = edges.data['h']
        h_v = edges.dst['h']
        h_u = h_u.reshape(h_u.shape[0], -1)
        h_v = h_v.reshape(h_v.shape[0], -1)
        e_w = e_w.reshape(e_w.shape[0], -1)
        score = self.W(torch.cat([h_u, h_v, e_w], 1))
        return {'score': score}

    def forward(self, g, h, e):
        h, e = self.layer1(g, h, e)
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['h'] = e
            g.apply_edges(self.apply_edges)
            return g.edata['score']