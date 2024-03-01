import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl',
                        init_method='tcp://<服务器IP>:<端口>',
                        world_size=2,
                        rank=1)

received_logits = torch.zeros(10, 100)
dist.recv(tensor=received_logits, src=0)

print("Edge side: received logits from server")
print("Received logits:\n", received_logits)