import torch
from typing import Callable
from collections.abc import Sequence
from torch.nn import Module
from torch import Tensor
from torchmetrics.metric import Metric, apply_to_collection
import torch

from hierarchicalsoftmax import nodes
from hierarchicalsoftmax.inference import ShapeError


RANKS = [
    'Realm', 'Subrealm',
    'Kingdom', 'Subkingdom', 'Phylum', 'Subphylum', 'Class', 'Subclass',
    'Order', 'Suborder', 'Family', 'Subfamily', 'Genus', 'Subgenus',
    'Species',
]
RANK_NUMBER = {rank: i+1 for i, rank in enumerate(RANKS)}

def rank_accurate(prediction_tensor, target_tensor, root:nodes.SoftmaxNode, max_depth:int=0):
    """ Returns a tensor of shape (samples,) with the depth of predictions which were accurate """
    ranks_correct = []

    if root.softmax_start_index is None:
        raise nodes.IndexNotSetError(f"The index of the root node {root} has not been set. Call `set_indexes` on this object.")

    if isinstance(prediction_tensor, tuple) and len(prediction_tensor) == 1:
        prediction_tensor = prediction_tensor[0]

    if prediction_tensor.shape[-1] != root.layer_size:
        raise ShapeError(
            f"The predictions tensor given to {__name__} has final dimensions of {prediction_tensor.shape[-1]}. "
            "That is not compatible with the root node which expects prediciton tensors to have a final dimension of {root.layer_size}."
        )

    for predictions, target in zip(prediction_tensor, target_tensor):
        node = root
        depth = 0
        target_node = root.node_list[target]
        target_path = target_node.path
        target_path_length = len(target_path)

        rank_correct = 0

        while (node.children):
            # This would be better if we could use torch.argmax but it doesn't work with MPS in the production version of pytorch
            # See https://github.com/pytorch/pytorch/issues/98191
            # https://github.com/pytorch/pytorch/pull/104374
            if len(node.children) == 1:
                # if this node use just one child, then we don't check the prediction
                prediction_child_index = 0
            else:
                prediction_child_index = torch.max(predictions[node.softmax_start_index:node.softmax_end_index], dim=0).indices

            node = node.children[prediction_child_index]
            depth += 1

            if depth < target_path_length and node != target_path[depth]:
                break

            rank_correct = RANK_NUMBER[node.rank]

            # Stop if we have reached the maximum depth
            if max_depth and depth >= max_depth:
                break

        ranks_correct.append(rank_correct)
    
    return torch.tensor(ranks_correct, dtype=int)


class ICTVTorchMetric(Metric):
    def __init__(self, root, name: str = "rank_accuracy"):
        super().__init__()
        self.root = root
        self.name = name

        # Use `add_state` for metrics to handle distributed reduction and device placement
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        for rank_name in RANKS:
            self.add_state(rank_name, default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        if isinstance(predictions, tuple) and len(predictions) == 1:
            predictions = predictions[0]

        # Ensure tensors match the device
        predictions = predictions.to(self.device)
        targets = targets.to(self.device)

        self.total += targets.size(0)
        rank_accurate_tensor = rank_accurate(predictions, targets, self.root)

        for rank_name, rank_number in RANK_NUMBER.items():
            accurate_at_rank = (rank_accurate_tensor >= rank_number).sum()
            setattr(self, rank_name, getattr(self, rank_name) + accurate_at_rank)

    def compute(self):
        # Compute final metric values
        return {
            rank_name: getattr(self, rank_name) / self.total
            for rank_name in RANKS
        }

    def _apply(self, fn: Callable, exclude_state: Sequence[str] = "") -> Module:
        """Overwrite `_apply` function such that we can also move metric states to the correct device.

        This method is called by the base ``nn.Module`` class whenever `.to`, `.cuda`, `.float`, `.half` etc. methods
        are called. Dtype conversion is guarded and will only happen through the special `set_dtype` method.

        Overriding because there is an issue device in the parent class.

        Args:
            fn: the function to apply
            exclude_state: list of state variables to exclude from applying the function, that then needs to be handled
                by the metric class itself.
        """
        this = super(Metric, self)._apply(fn)
        fs = str(fn)
        cond = any(f in fs for f in ["Module.type", "Module.half", "Module.float", "Module.double", "Module.bfloat16"])
        if not self._dtype_convert and cond:
            return this

        # Also apply fn to metric states and defaults
        for key, value in this._defaults.items():
            if key in exclude_state:
                continue

            if isinstance(value, Tensor):
                this._defaults[key] = fn(value)
            elif isinstance(value, Sequence):
                this._defaults[key] = [fn(v) for v in value]

            current_val = getattr(this, key)
            if isinstance(current_val, Tensor):
                setattr(this, key, fn(current_val))
            elif isinstance(current_val, Sequence):
                setattr(this, key, [fn(cur_v) for cur_v in current_val])
            else:
                raise TypeError(
                    f"Expected metric state to be either a Tensor or a list of Tensor, but encountered {current_val}"
                )

        # Additional apply to forward cache and computed attributes (may be nested)
        if this._computed is not None:
            this._computed = apply_to_collection(this._computed, Tensor, fn)
        if this._forward_cache is not None:
            this._forward_cache = apply_to_collection(this._forward_cache, Tensor, fn)

        return this
