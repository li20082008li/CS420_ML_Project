def to_numpy(tensor):
    return tensor.cpu().detach().numpy().transpose(0, 2, 3, 1)  # (Batch, H, W, C)


def classify_class(x):
    return 1.0 * (x > 0.5)
