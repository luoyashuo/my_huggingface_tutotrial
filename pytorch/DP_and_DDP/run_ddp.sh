     CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 --master_port=7890 pytorch/DP_and_DDP/ddp_demo.py