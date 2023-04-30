#!/bin/bash

python main.py --lang en --data_dir FUNSD --best_checkpoint_path checkpoints/gt_9222.pth \
--logger_file logs/logger.log --graph_save_dir form_graphs --train --seed 9222

python main.py --lang en --data_dir FUNSD --best_checkpoint_path checkpoints/gt_3145.pth \
--logger_file logs/logger.log --graph_save_dir form_graphs --train --seed 3145

python main.py --lang en --data_dir FUNSD --best_checkpoint_path checkpoints/gt_9300.pth \
--logger_file logs/logger.log --graph_save_dir form_graphs --train --seed 9300

python main.py --lang en --data_dir FUNSD --best_checkpoint_path checkpoints/gt_8222.pth \
--logger_file logs/logger.log --graph_save_dir form_graphs --train --seed 8222

python main.py --lang en --data_dir FUNSD --best_checkpoint_path checkpoints/gt_7222.pth \
--logger_file logs/logger.log --graph_save_dir form_graphs --train --seed 7222

python main.py --lang en --data_dir FUNSD --best_checkpoint_path checkpoints/gt_6222.pth \
--logger_file logs/logger.log --graph_save_dir form_graphs --train --seed 6222

python main.py --lang en --data_dir FUNSD --best_checkpoint_path checkpoints/gt_5222.pth \
--logger_file logs/logger.log --graph_save_dir form_graphs --train --seed 5222

python main.py --lang en --data_dir FUNSD --best_checkpoint_path checkpoints/gt_4222.pth \
--logger_file logs/logger.log --graph_save_dir form_graphs --train --seed 4222

# python main.py --lang ja --data_dir FUNSD --best_checkpoint_path checkpoints/test_egat.pth \
# --logger_file logs/logger.log --graph_save_dir form_graphs --create_graphs

# python main.py --lang zh --data_dir FUNSD --best_checkpoint_path checkpoints/test_egat.pth \
# --logger_file logs/logger.log --graph_save_dir form_graphs --create_graphs

# python main.py --lang de --data_dir FUNSD --best_checkpoint_path checkpoints/test_egat.pth \
# --logger_file logs/logger.log --graph_save_dir form_graphs --create_graphs

# python main.py --lang fr --data_dir FUNSD --best_checkpoint_path checkpoints/test_egat.pth \
# --logger_file logs/logger.log --graph_save_dir form_graphs --create_graphs

# python main.py --lang pt --data_dir FUNSD --best_checkpoint_path checkpoints/test_egat.pth \
# --logger_file logs/logger.log --graph_save_dir form_graphs --create_graphs

# python main.py --lang es --data_dir FUNSD --best_checkpoint_path checkpoints/test_egat.pth \
# --logger_file logs/logger.log --graph_save_dir form_graphs --create_graphs

# python main.py --lang it --data_dir FUNSD --best_checkpoint_path checkpoints/test_egat.pth \
# --logger_file logs/logger.log --graph_save_dir form_graphs --create_graphs