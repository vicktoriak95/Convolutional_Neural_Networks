import torch


class TensorView(object):
    """
    A transform that returns a new view of a tensor.
    """

    def __init__(self, *view_dims):
        self.view_dims = view_dims

    def __call__(self, tensor: torch.Tensor):
        # TODO: Use Tensor.view() to implement the transform.
        # ====== YOUR CODE: ======
        cur = tensor.view(self.view_dims)
        return cur
        # ========================


class BiasTrick(object):
    """
    A transform that applies the "bias trick": Adds an element equal to 1 to
    a given tensor.
    """

    def __call__(self, tensor: torch.Tensor):
        assert tensor.dim() == 1, "Only 1-d tensors supported"

        # TODO: Add a 1 at the end of the given tensor.
        # Make sure to use the same data type.

        # ====== YOUR CODE: ======
        bias = torch.FloatTensor([1])
        trans = torch.cat((tensor, bias), 0)
        return trans
        # ========================


