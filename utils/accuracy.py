import torch
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == "__main__":
    import torch.nn.functional as F
    targets = torch.randint(0,5, size=(256,))
    targets_ = F.one_hot(targets, num_classes=5).type(torch.float)*1000
    outputs = torch.randn((256,5))
    print(targets.shape)
    print(targets_.shape)
    print(outputs.shape)
    print(
        accuracy(targets_, targets, (1,3))
    )
    print(accuracy(outputs, targets))