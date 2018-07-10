import torch
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel

device = torch.device("cuda")


class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 5), nn.Sigmoid()
        )

    def forward(self, input):
        print("I am device", torch.cuda.current_device())
        return self.model(input)


# d = nn.DataParallel(M(), device_ids=[0, 1], output_device=-1).to(device)

d = DistributedDataParallel(M(), output_device=-1)
i = torch.ones(2, 3)

print(d.forward(i))
