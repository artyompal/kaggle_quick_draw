import torch

def precision(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def mapk(output, target, topk=3):
    """
    Computes mapk
    """
    batch_size = target.size(0)

    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = torch.Tensor()
    prev_k = torch.Tensor()

    for i in range(topk):
        correct_k = correct[i].view(-1).float().sum(0, keepdim=True)
        if i == 0:
            res.data = correct_k.clone()
        else:
            res.add_(correct_k.mul(1. / (i + 1)))

    res.mul_(100.0 / batch_size)
    return res.item()
