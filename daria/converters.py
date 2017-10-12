import torch

from torch.autograd import Variable


def tuple_converter(batch, device, train=True):
    data, target = batch
    if device is None:
        data, target = data.cuda(), target.cuda()
    elif device >= 0:
        data, target = data.cuda(device), target.cuda(device)

    if torch.is_tensor(data):
        data, target = Variable(data, volatile=not train), Variable(target)

    return data, target
