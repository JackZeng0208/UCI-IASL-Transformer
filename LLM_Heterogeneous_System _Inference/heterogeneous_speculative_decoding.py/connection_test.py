import os
import torch.distributed as dist
# from torch.multiprocessing import Process
from  time import sleep
import datetime
import torch

def double_send(rank):
    if rank == 0: # server
        draft_output_shape = torch.empty((2),dtype=torch.long)
        req = dist.recv(tensor=draft_output_shape,src=1)
        req.wait()
        print(f"draft_output_shape from src 1 should be [1,1] is {draft_output_shape}")
        # draft_output = torch.empty((draft_output_shape[0],draft_output_shape[1]),dtype=torch.long)
        # req = dist.irecv(tensor=draft_output,src =1)
        # req.wait()
    if rank ==1: # edge
        shape = torch.tensor([1,1])
        req = dist.send(tensor=shape,dst=0)
        req.wait()


def init_processes(rank, size,IP, Port ,backend='gloo'):
    """ Initialize the distributed environment. """    
    print('{} : started process for rank : {}'.format(os.getpid(),rank))
	
    # os.environ['MASTER_ADDR'] = "66.42.104.193"
    # os.environ['MASTER_PORT'] = '6100'
	#Remove init_method if initializing through environment variable
    dist.init_process_group(backend = backend, 
                            init_method=f'tcp://{IP}:{Port}',
                            rank=rank,
                            world_size=size,
                            timeout=datetime.timedelta(1,seconds =  20))
    #dist.init_process_group(backend, rank=rank, world_size=size)


if __name__ == "__main__":
    print(dist.is_available())
    print(dist.is_nccl_available())
    # init_processes(0,2,'66.42.104.193','6100')
    # # init_processes(1,2,'192.168.0.44','1234')
    # double_send(rank=0)
    # os.environ['MASTER_ADDR'] = '66.42.104.193'  # IP of Machine 1
    # os.environ['MASTER_PORT'] = '6100'    
    IP =  '66.42.104.193'   
    # IP = "192.168.0.146"
    Port = "6100"
    dist.init_process_group(backend='nccl', init_method=f'tcp://{IP}:{Port}',rank=0, world_size=2,timeout=datetime.timedelta(1,seconds =  20))
    print('connected')
    double_send(0)