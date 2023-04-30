import gurobipy as gp
from gurobipy import GRB
import numpy as np
from sklearn import metrics
import random
from dgl.data.utils import load_graphs
from sklearn.preprocessing import MultiLabelBinarizer
import logging
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--edge_pred_confidence_file', type=str, required=True,
                        help='path to the saved edge predictions confidence file')
    parser.add_argument('--edge_ground_truth_file', type=str, required=True,
                        help='path to the saved edge ground truth file')
    parser.add_argument('--edge_pred_file', type=str, required=True,
                        help='path to the saved edge predictions file')
    parser.add_argument('--edge_feats_file', type=str, required=True,
                        help='path to the saved edge features file')
    parser.add_argument('--test_graphs_file', type=str, required=True,
                        help='path to the saved test graphs to perform ILP optimization')
    parser.add_argument('--seed', type=int, default=9222,
                        help='random seed (default: 9222)')
    parser.add_argument('--logger_file', type=str, default='./logger_ILP.log',
                        help='path to save the logs of ILP optimizer.')
    return parser.parse_args()

def printResults(args, gurobi_vars, logger):

    tot_gurobi_preds = []
    tot_gnn_preds = []
    tot_org_trues = []

    gnn_acc = 0.0
    gurobi_acc = 0.0

    if m.status == GRB.OPTIMAL:
        logger.info('\nCost: %g' % m.ObjVal)
        tot_gurobi_preds = []
        for doc_idx, doc in enumerate(args.preds):
            sub_gnn_preds, sub_gnn_trues, sub_gurobi_preds = [], [], []
            for widx, w in enumerate(doc):
                sub_gnn_preds.append(list(w).index(max(w)))
                sub_gnn_trues.append(trues[doc_idx][widx])
                for k in range(len(w)):
                    if gurobi_vars[doc_idx][widx, k].X > 0.0:
                        sub_gurobi_preds.append(k)
            tot_gurobi_preds.append(sub_gurobi_preds)
            tot_gnn_preds.append(sub_gnn_preds)
            tot_org_trues.append(sub_gnn_trues)

        mlb = MultiLabelBinarizer()
        mlb.fit(tot_org_trues)
        bin_tot_org_trues = mlb.transform(tot_org_trues)
        
        mlb.fit(tot_gnn_preds)
        bin_tot_gnn_preds = mlb.transform(tot_gnn_preds)

        mlb.fit(tot_gurobi_preds)
        bin_tot_gurobi_preds = mlb.transform(tot_gurobi_preds)

        gnn_acc = metrics.f1_score(bin_tot_org_trues, bin_tot_gnn_preds, average = "macro")
        logger.info(f"Total GNN Pred Accuracy: {gnn_acc}")
        gurobi_acc = metrics.f1_score(bin_tot_org_trues, bin_tot_gurobi_preds, average = "macro")
        logger.info(f"Total Gurobi Pred Accuracy: {gurobi_acc}")
    else:
        logger('No solution')
    
def gurobi_objective_fn(args, m):
    gurobi_vars = []
    for doc in args.preds:
        var = m.addVars(doc.shape[0], doc.shape[1], vtype=GRB.BINARY)
        gurobi_vars.append(var)

    m.setObjective(gp.quicksum(preds[doc_idx][idx][k] * gurobi_vars[doc_idx][idx, k] \
    for doc_idx, doc in enumerate(args.preds) for idx, ln in enumerate(doc) for k in range(len(args.classes))), GRB.MAXIMIZE)
    
    return gurobi_vars

def set_constraint2(args, gurobi_vars):

    for doc_idx, doc in enumerate(args.preds):
        for wrd_idx, _ in enumerate(doc):
            if wrd_idx + 1 < len(doc):
                if abs(args.edge_feats[doc_idx][wrd_idx + 1][1]) < 0.01:
                    m.addConstr(
                        gp.quicksum(gurobi_vars[doc_idx].select(wrd_idx, 0)) <= (gp.quicksum(gurobi_vars[doc_idx].select(wrd_idx + 1, 3)) + \
                            gp.quicksum(gurobi_vars[doc_idx].select(wrd_idx + 1, 2))), name = "C2"
                    )

def set_constraint3(args, gurobi_vars):

    for doc_idx, doc in enumerate(args.preds):
        for wrd_idx, w in enumerate(doc):
            suu = []
            for k in range(len(w)):
                suu.append(gurobi_vars[doc_idx].select(wrd_idx, k)[0])
            m.addConstr(gp.quicksum(suu) == 1, name = "C3")

def set_constraint4(args, gurobi_vars):
    for graph_idx, graph in enumerate(args.fin_test_graphs):
        edges = graph.edges()[0]
        i, j = 0, 1
        while i <= len(edges) - 1:
            constrs = []
            constrs.append(gurobi_vars[graph_idx].select(i, 0)[0])
            constrs.append(gurobi_vars[graph_idx].select(i, 1)[0])
            constrs.append(gurobi_vars[graph_idx].select(i, 2)[0])
            constrs.append(gurobi_vars[graph_idx].select(i, 3)[0])
            while j <= len(edges) - 1 and edges[i] == edges[j]:
                constrs.append(gurobi_vars[graph_idx].select(j, 0)[0])
                constrs.append(gurobi_vars[graph_idx].select(j, 1)[0])
                constrs.append(gurobi_vars[graph_idx].select(j, 2)[0])
                constrs.append(gurobi_vars[graph_idx].select(j, 3)[0])
                j += 1
            m.addConstr(gp.quicksum(constrs) >= 1, name = "C4")
            i = j
            j = i + 1

def load_required_files(args):

    preds = np.load(f"{args.edge_pred_confidence_file}", allow_pickle = True)
    trues = np.load(f"{args.edge_ground_truth_file}", allow_pickle = True)
    gnn_preds = np.load(f"{args.edge_pred_file}", allow_pickle = True)
    edge_feats = np.load(f"{args.edge_feats_file}", allow_pickle = True)
    test_graphs = load_graphs(f"{args.test_graphs_file}")[0]

    m_preds, m_trues, m_gnn_preds, m_edge_feats = [], [], [], []

    fin_test_graphs = []
    for tg_idx, tg in enumerate(test_graphs):
        elabels = tg.edata["label"]
        mask = elabels != 5
        fin_test_graphs.append(tg.edge_subgraph(mask))
        sub_lst_preds, sub_lst_trues, sub_lst_gnn_preds, sub_lst_edge_feats = [], [], [], []
        for p_idx, p in enumerate(elabels):
            if p != 5:
                sub_lst_preds.append(preds[tg_idx][p_idx])
                sub_lst_trues.append(trues[tg_idx][p_idx])
                sub_lst_gnn_preds.append(gnn_preds[tg_idx][p_idx])
                sub_lst_edge_feats.append(edge_feats[tg_idx][p_idx])

        m_preds.append(np.array(sub_lst_preds))
        m_trues.append(np.array(sub_lst_trues))
        m_gnn_preds.append(np.array(sub_lst_gnn_preds))
        m_edge_feats.append(np.array(sub_lst_edge_feats))

    preds, trues, gnn_preds, edge_feats = np.array(m_preds), np.array(m_trues), np.array(m_gnn_preds), np.array(m_edge_feats)
    return preds, trues, gnn_preds, edge_feats, fin_test_graphs

if __name__ == "__main__":

    args = parse_args()

    logging.basicConfig(filename=args.logger_file,
                        format='%(asctime)s %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filemode='a')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    np.random.seed(args.seed)
    random.seed(args.seed)

    classes = ["question-answer", "header-question", "proximate_H", "same-entity", "proximate_V", "no-link"]
    args.classes = classes
    
    m = gp.Model("edge_classification")

    preds, trues, gnn_preds, edge_feats, fin_test_graphs = load_required_files(args)
    args.preds, args.trues, args.gnn_preds, args.edge_feats, args.fin_test_graphs = preds, trues, gnn_preds, edge_feats, fin_test_graphs
    
    gurobi_vars = gurobi_objective_fn(args, m)

    set_constraint2(args, gurobi_vars)
    set_constraint3(args, gurobi_vars)
    set_constraint4(args, gurobi_vars)

    m.optimize()
    printResults(args, gurobi_vars, logger)