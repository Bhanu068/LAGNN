#!/bin/bash

python main.py --lang en --data_dir FUNSD --best_checkpoint_path checkpoints/egat.pth \
--logger_file logs/logger.log --graph_save_dir form_graphs --train --seed 6222