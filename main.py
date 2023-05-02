from graph_transformer import GraphTransformerNet
from egat_model import EGAT
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dgl import seed
import time
import logging
from sklearn import metrics
from dgl.data.utils import load_graphs
import random
import torch.nn.functional as F
import dgl
import numpy as np
import os
from tqdm import tqdm
import torch
import matplotlib
matplotlib.use('Agg')


def parse_args():
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to use for training (default: "cuda")')
    parser.add_argument('--best_checkpoint_path', type=str, required=True,
                        help='path to save the best checkpoint')
    parser.add_argument('--early_stop', type=int, default=100,
                        help='number of epochs to wait before early stopping (default: 100)')
    parser.add_argument('--logger_file', type=str, required=True,
                        help='path to save the logger file')
    parser.add_argument('--seed', type=int, default=6222,
                        help='random seed (default: 6222)')
    parser.add_argument('--create_graphs', action='store_true',
                        help='create graphs flag (default: False)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='path to the data directory')
    parser.add_argument('--lang', type=str, required=True, choices=['en', 'fr', 'zh', 'ja', 'pt', 'es', 'it', 'de'],
                        help='language choice (required): en, fr, zh, ja, pt, es, it, de')
    parser.add_argument('--graph_save_dir', type=str, required=True,
                        help='path to save the graph')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=100,
                        help='number of epochs to wait before reducing the learning rate (default: 100)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train flag (default: False)')
    parser.add_argument('--in_node_feats', type=int, default=4,
                        help='dimension size of node features')
    parser.add_argument('--out_node_feats', type=int, default=64,
                        help='hidden dimension size of node features')
    parser.add_argument('--in_edge_feats', type=int, default=6,
                        help='dimension size of edge features')
    parser.add_argument('--out_edge_feats', type=int, default=64,
                        help='hidden dimension size of edge features')
    parser.add_argument('--num_heads', type=int, default=3,
                        help='number of heads for attention')

    args = parser.parse_args()

    if args.lang != 'en':
        args.train = False

    return args


def user_friendly_time(s):
    """ Display a user friendly time from number of second. """
    s = int(s)
    if s < 60:
        return "{}s".format(s)

    m = s // 60
    s = s % 60
    if m < 60:
        return "{}m {}s".format(m, s)

    h = m // 60
    m = m % 60
    if h < 24:
        return "{}h {}m {}s".format(h, m, s)

    d = h // 24
    h = h % 24
    return "{}d {}h {}m {}s".format(d, h, m, s)


def time_since(t):
    """ Function for time. """
    return time.time() - t


def create_graph_list(edge_class_to_idx, data_dir, lang, type, graph_save_dir):
    fin_graphs_list = []

    for _, file in tqdm(enumerate(os.listdir(os.path.join(data_dir, lang, type)))):
        j = 0
        with open(os.path.join(data_dir, lang, type, file), "r") as f:
            graph_text = f.read().split('\n')[:-1]

        if len(graph_text) > 0:
            G = dgl.DGLGraph()
            node0, node1, labels, edge_features, node_features = [], [], [], [], []
            n1 = 0
            for _, node in enumerate(graph_text):
                if len(node.split("\t")) == 13:
                    j += 1
                    label = node.split("\t")[4]
                if label == "question-header" or label == "header-header" or label == "question-question":
                    label = "proximate_H"
                if label == "question-other":
                    label = "question-answer"
                if label == "line-line":
                    label = "proximate_V"
                box_2 = [float(coord)
                         for coord in node.split("\t")[2][1:-1].split(", ")]
                box_3 = [float(coord)
                         for coord in node.split("\t")[3][1:-1].split(", ")]
                if box_2 not in node_features:
                    node_features.append(box_2)
                    node0.append(n1)
                    n1 += 1
                else:
                    indx = node_features.index(box_2)
                    node0.append(indx)
                if box_3 not in node_features:
                    node_features.append(box_3)
                    node1.append(n1)
                    n1 += 1
                else:
                    indx = node_features.index(box_3)
                    node1.append(indx)
                labels.append(edge_class_to_idx[label])
                edge_features.append(
                    np.array(node.split("\t")[-7:-1], dtype="float"))

            G.add_nodes(len(node_features), {
                        "nfeats": torch.Tensor(np.array(node_features))})
            G.add_edges(node0, node1, {"label": torch.Tensor(
                np.array(labels)), "edge_features": torch.Tensor(np.array(edge_features))})
            fin_graphs_list.append(G)

    if not os.path.exists(os.path.join(graph_save_dir, lang)):
        os.mkdir(os.path.join(graph_save_dir, lang))
    dgl.save_graphs(os.path.join(graph_save_dir, lang,
                    f"semantic_form_graph_{type}.bin"), fin_graphs_list)


def train_model(args, model, opt, train_graphs_list, eval_graphs_list, scheduler):
    logger.info("Training Started")
    model.train()
    best_eval_acc = 0.0
    not_improved = 0
    for epoch in range(args.n_epochs):
        start = time.time()
        preds, trues = [], []
        model.train()
        for graph in train_graphs_list:
            graph = graph.to(args.device)
            node_features = graph.ndata['nfeats']
            edge_features = graph.edata['edge_features']
            edge_label = graph.edata['label'].long()
            graph.edata['train_mask'] = torch.ones(
                len(edge_label), dtype=torch.bool).to(args.device)
            train_mask = graph.edata['train_mask']
            opt.zero_grad()
            pred = model(graph, node_features, edge_features.float())
            loss = F.cross_entropy(pred[train_mask], edge_label[train_mask])
            pred = F.log_softmax(pred, 1)
            pred_label = pred.argmax(1)
            preds.extend(list(pred_label[train_mask].cpu().numpy()))
            trues.extend(list(edge_label[train_mask].cpu().numpy()))
            loss.backward()
            opt.step()

        eval_acc = eval_model(args, model, eval_graphs_list)
        if eval_acc > best_eval_acc:
            not_improved = 0
            best_eval_acc = eval_acc
            torch.save(model.state_dict(), args.best_checkpoint_path)
        else:
            not_improved += 1
        if not_improved == args.early_stop:
            logger.info(
                "Early stop since evaluation accuracy is not improving")
            break
        if epoch == 0 or epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}: Training Accuracy: {round(metrics.f1_score(trues, preds, average = 'macro'), 4)} Evaluation Accuracy: {eval_acc} {user_friendly_time(time_since(start))}")


def eval_model(args, model, eval_graphs_list):
    epreds, etrues = [], []
    for graph in eval_graphs_list:
        model.eval()
        graph = graph.to(args.device)
        node_features = graph.ndata['nfeats']
        edge_features = graph.edata['edge_features']
        edge_label = graph.edata['label'].long()
        graph.edata['test_mask'] = torch.ones(
            len(edge_label), dtype=torch.bool).to(args.device)
        etest_mask = graph.edata['test_mask']
        with torch.no_grad():
            epred = model(graph, node_features, edge_features.float())
            epred_label = epred.argmax(1)

        epreds.extend(list(epred_label[etest_mask].cpu().numpy()))
        etrues.extend(list(edge_label[etest_mask].cpu().numpy()))

    return round(metrics.f1_score(etrues, epreds, average="macro"), 4)


def test_model(args, model, test_graphs_list):
    logger.info("\n")
    logger.info("Testing Started")
    epreds, etrues, epreds_list, etrues_list = [], [], [], []
    econfidence, etrues_latest, epreds_latest, edge_feats, final_test_pred_confidence, final_test_preds, final_test_trues = [], [], \
        [], [], [], [], []
    start = time.time()
    model.load_state_dict(torch.load(args.best_checkpoint_path), strict = False)
    for _, graph in enumerate(test_graphs_list):
        model.eval()
        graph = graph.to(args.device)
        node_features = graph.ndata['nfeats']
        edge_features = graph.edata['edge_features']
        edge_label = graph.edata['label'].long()
        graph.edata['test_mask'] = torch.ones(
            len(edge_label), dtype=torch.bool).to(args.device)
        etest_mask = graph.edata['test_mask']

        with torch.no_grad():
            epred = model(graph, node_features, edge_features.float())
            epred = F.softmax(epred, 1)
            pred_edge_label = epred.argmax(1)
            final_test_pred_confidence.append(epred.cpu().numpy())
            final_test_preds.append(pred_edge_label.cpu().numpy())
            final_test_trues.append(edge_label.cpu().numpy())

        epreds.extend(list(pred_edge_label[etest_mask].cpu().numpy()))
        etrues.extend(list(edge_label[etest_mask].cpu().numpy()))
        epreds_list.append(list(pred_edge_label[etest_mask].cpu().numpy()))
        etrues_list.append(list(edge_label[etest_mask].cpu().numpy()))

        econfidence.append(epred.cpu().numpy())
        etrues_latest.append(edge_label[etest_mask].cpu().numpy())
        epreds_latest.append(pred_edge_label[etest_mask].cpu().numpy())
        edge_feats.append(edge_features[etest_mask].cpu().numpy())

    if not os.path.exists(f"./prediction_files"):
        os.mkdir(f"./prediction_files")
    np.save(f"./prediction_files/econfidence_{args.lang}.npy", econfidence)
    np.save(f"./prediction_files/etrues_latest_{args.lang}.npy", etrues_latest)
    np.save(f"./prediction_files/epreds_latest_{args.lang}.npy", epreds_latest)
    np.save(f"./prediction_files/edge_feats_{args.lang}.npy", edge_feats)

    logger.info(round(metrics.f1_score(etrues, epreds, average="macro"), 4))
    logger.info(metrics.classification_report(etrues, epreds))
    logger.info(f"Time: {user_friendly_time(time_since(start))}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    args = parse_args()

    logging.basicConfig(filename=args.logger_file,
                        format='%(asctime)s %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filemode='a')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    seed(args.seed)

    classes = ["question-answer", "header-question", "proximate_H", "same-entity", "proximate_V", "no-link"]

    args.classes = len(classes)
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"

    edge_class_to_idx = {}
    for idx, c in enumerate(classes):
        edge_class_to_idx[c] = idx

    if args.create_graphs:
        if args.lang == "en":
            create_graph_list(edge_class_to_idx, args.data_dir,
                            args.lang, "train", args.graph_save_dir)
            create_graph_list(edge_class_to_idx, args.data_dir,
                            args.lang, "val", args.graph_save_dir)
        create_graph_list(edge_class_to_idx, args.data_dir,
                          args.lang, "test", args.graph_save_dir)

    if args.lang == "en":
        train_graphs_full = load_graphs(os.path.join(
            args.graph_save_dir, args.lang, f"semantic_form_graph_train.bin"))
        val_graphs_full = load_graphs(os.path.join(
            args.graph_save_dir, args.lang, "semantic_form_graph_val.bin"))
    test_graphs_full = load_graphs(os.path.join(
        args.graph_save_dir, args.lang, "semantic_form_graph_test.bin"))

    model = EGAT(args)
    model.to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, patience=args.patience, mode="max")

    logger.info(f"Args \n {args}")
    logger.info(f"Model parameter count: {int(count_parameters(model))}")

    if args.train:
        train_model(args, model, opt,
                    train_graphs_full[0], val_graphs_full[0], scheduler)
    test_model(args, model, test_graphs_full[0])