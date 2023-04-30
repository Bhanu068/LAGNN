#!/bin/bash

python semantic_form_ilp.py \
--edge_pred_confidence_file ./prediction_files/econfidence_en.npy \
--edge_ground_truth_file ./prediction_files/etrues_latest_en.npy \
--edge_pred_file ./prediction_files/epreds_latest_en.npy \
--edge_feats_file ./prediction_files/edge_feats_en.npy \
--test_graphs_file ./form_graphs/en/semantic_form_graph_all_test.bin \
--logger_file ./logs/logger_ilp.log