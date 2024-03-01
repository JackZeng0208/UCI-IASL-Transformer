import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl',
                        init_method='tcp://<服务器IP>:<端口>',
                        world_size=2,
                        rank=0)

# Example logits
logits = torch.randn(10, 100)

dist.barrier()
dist.send(tensor=logits, dst=1)

print("Message has been sent successfully")
