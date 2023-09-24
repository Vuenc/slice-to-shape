from typing import Iterator, List, Sequence, Union
import torch
from torch.utils.data import Sampler

class BlockRandomBatchSampler(Sampler[Sequence[int]]):
    """
    A batch sampler that can be passed to a DataLoader with the `batch_sampler` argument.
    Can be used if the dataset consists of several contiguous blocks of known sizes,
    and the items within the blocks, as well as the blocks themselves should be shuffled,
    but each batch should only contain samples from one block.
    """
    def __init__(self, block_sizes: Union[List[int], torch.Tensor], batch_size: int, allow_block_overlap=False):
        self.batch_size = batch_size
        self.block_sizes = torch.tensor(block_sizes)
        self.block_start_indices = torch.hstack([torch.tensor([0]), torch.cumsum(self.block_sizes, 0)[:-1]])
        self.allow_block_overlap = allow_block_overlap
        if allow_block_overlap:
            self.length = int(torch.ceil(self.block_sizes.sum() / batch_size).item())
        else:
            self.length = int(torch.ceil(self.block_sizes / batch_size).sum().item())

    def __len__(self):
        return self.length
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        block_permutation = torch.randperm(len(self.block_sizes))
        block_sizes_permuted, block_start_indices_permuted = self.block_sizes[block_permutation], self.block_start_indices[block_permutation]
        if not self.allow_block_overlap:
            for block_size, block_start_index in zip(block_sizes_permuted, block_start_indices_permuted):
                permutation = torch.randperm(block_size) + block_start_index # type: ignore
                yield from [permutation[idx:idx + self.batch_size] for idx in range(0, permutation.shape[0], self.batch_size)]
        else:
            full_permutation = torch.cat([torch.randperm(block_size) + block_start_index  # type: ignore
                for block_size, block_start_index in zip(block_sizes_permuted, block_start_indices_permuted)])
            yield from [full_permutation[idx:idx + self.batch_size] for idx in range(0, full_permutation.shape[0], self.batch_size)]