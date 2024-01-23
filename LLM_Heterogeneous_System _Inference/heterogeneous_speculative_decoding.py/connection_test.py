import os
import torch.distributed as dist
# from torch.multiprocessing import Process
from  time import sleep
import datetime
import torch

def double_send(rank):
    if rank == 0: # server
        draft_output_shape = torch.empty((2),dtype=torch.long)
        req = dist.irecv(tensor=draft_output_shape,src=1)
        req.wait()
        print(f"draft_output_shape from src 1 should be [1,1] is {draft_output_shape}")
        # draft_output = torch.empty((draft_output_shape[0],draft_output_shape[1]),dtype=torch.long)
        # req = dist.irecv(tensor=draft_output,src =1)
        # req.wait()
    if rank ==1: # edge
        shape = torch.tensor([1,1])
        req = dist.isend(tensor=shape,dst=0)
        req.wait()


def init_processes(rank, size,IP, Port ,backend='nccl'):
    """ Initialize the distributed environment. """    
    print('{} : started process for rank : {}'.format(os.getpid(),rank))
	
	#Remove init_method if initializing through environment variable
    dist.init_process_group(backend = backend, 
                            init_method=f'tcp://{IP}:{Port}',
                            rank=rank,
                            world_size=size,
                            timeout=datetime.timedelta(0,seconds =  20))
    #dist.init_process_group(backend, rank=rank, world_size=size)


if __name__ == "__main__":
    init_processes(0,2,'66.42.104.193','8233')
    double_send(rank=0)