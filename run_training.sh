cd projects/mae_lite
ssl_train -b 2048 -d 0-7 -e 500 -f mae_lite_exp.py --amp --ckpt /cnvrg/model/mae_tiny_400e.pth.tar --exp-options pretrain_exp_name=/cnvrg/outputs/mae_tiny_400e_pretrained 
