import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import time

def main_worker(rank, world_size, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:9001",
        world_size=world_size,
        rank=rank,
    )
    print("process begin", rank)

    for datatype in [None, torch.float,torch.float16,torch.float32,torch.float64]:
        if rank == 0:
            print(f"Current datatype: {datatype}.")
            t = torch.rand([4,4],dtype=datatype).to(torch.device('cuda',rank))      
            print(f"send tensor\n{t}")
            dist.send(t,1)

        elif rank == 1:
            r = torch.rand([4,4],dtype=datatype).to(torch.device('cuda',rank)) 
            dist.recv(r,0)
            print(f"recv tensor\n{r}")
        print()
        time.sleep(1)

def main():
    mp.spawn(main_worker, nprocs=2, args=(2, 2))

if __name__ == "__main__":
    main()