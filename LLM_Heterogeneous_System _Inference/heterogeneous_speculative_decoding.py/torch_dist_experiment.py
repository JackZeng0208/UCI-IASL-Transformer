"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '192.168.0.132'
    os.environ['MASTER_PORT'] = '5000'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
# if __name__ == "__main__":
#     os.environ["0NCCL_AVOID_RECORD_STREAMS"] = "0"
#     print(dist.is_available())
#     print(dist.is_nccl_available()) 
#     # SERVER_IP = '0.0.0.0'
#     EDGE_IP = '192.168.0.208'
#     SERVER_IP = '192.168.0.132' 
#     Port = "5000"
#     init_processes(rank=0,
#                    size=2,
#                    IP=SERVER_IP,
#                    Port=Port)
#     print('connect to edge')
#     double_send(0)
        