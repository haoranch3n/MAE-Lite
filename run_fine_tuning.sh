cd projects/eval_tools
ssl_train -b 1024 -d 0-7 -e 300 -f finetuning_exp.py --amp --ckpt /cnvrg/model/mae_tiny_400e.pth.tar --exp-options pretrain_exp_name=mae_lite/mae_tiny_400e
