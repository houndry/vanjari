from pathlib import Path
import random
from zlib import adler32
from torchapp.download import cached_download
import torch
import numpy as np
from pathlib import Path
import lightning as L
from corgi.seqtree import SeqTree
from torch.utils.data import DataLoader
from rich.console import Console
from torch.utils.data import Dataset
from dataclasses import dataclass

console = Console()


def get_vmr(local_path) -> Path:
    url = "https://ictv.global/sites/default/files/VMR/VMR_MSL39.v4_20241106.xlsx"
    return cached_download(url, local_path)


@dataclass(kw_only=True)
class Species():
    accession:str
    index:int
    count:int


@dataclass(kw_only=True)
class VanjariStackTrainingDataset(Dataset):
    species: list[Species]
    seqtree: SeqTree
    array:np.memmap|np.ndarray
    stack_size:int = 16
    deterministic:bool = False

    def __len__(self):
        return len(self.species)

    def __getitem__(self, idx):
        species = self.species[idx]
        stack_size = min(self.stack_size, species.count)
        rng = np.random.RandomState(adler32(species.accession.encode("ascii"))) if self.deterministic else np.random
        array_index = species.index + rng.randint(0, species.count-stack_size+1)

        with torch.no_grad():
            data = np.array(self.array[array_index:array_index+stack_size, :], copy=False)
            embedding = torch.tensor(data, dtype=torch.float16)
            del data

        seq_detail = self.seqtree[species.accession+":0"]
        node_id = int(seq_detail.node_id)
        del seq_detail
        
        return embedding, node_id


@dataclass
class VanjariStackDataModule(L.LightningDataModule):
    seqtree: SeqTree
    array:np.memmap|np.ndarray
    accession_to_array_index:dict[str,int]
    batch_size: int = 1
    num_workers: int = 0
    max_items:int = 0
    seed:int = 42
    validation_partition:int=0
    stack_size:int = 16

    def __post_init__(self):
        super().__init__()

    def setup(self, stage=None):
        current_accession = None
        start_index = 0
        self.training = []
        self.validation = []

        random.seed(self.seed)

        current_list = None

        for accession, index in self.accession_to_array_index.items():
            species_accession = accession.split(":")[0]
            if current_accession != species_accession:
                if current_list is not None:
                    current_list.append(Species(accession=current_accession, index=start_index, count=index-start_index))
                current_accession = species_accession
                start_index = index

                detail = self.seqtree[accession]
                current_list = self.validation if detail.partition == self.validation_partition else self.training
        
        current_list.append(Species(accession=current_accession, index=start_index, count=index-start_index))

        if self.max_items:
            self.training = self.training[:self.max_items]
            self.validation = self.validation[:self.max_items]

        self.train_dataset = self.create_dataset(self.training, deterministic=False)
        self.val_dataset = self.create_dataset(self.validation, deterministic=True)

    def create_dataset(self, species:list[Species], deterministic:bool) -> VanjariStackTrainingDataset:
        return VanjariStackTrainingDataset(
            species=species,
            seqtree=self.seqtree, 
            array=self.array,
            stack_size=self.stack_size,
            deterministic=deterministic,
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


###########

# @dataclass(kw_only=True)
# class VanjariStackTrainingDataset(Dataset):
#     species: list[Species]
#     seqtree: SeqTree
#     array:np.memmap|np.ndarray
#     stack_size:int = 16

#     def __len__(self):
#         return len(self.species)

#     def __getitem__(self, idx):
#         species = self.species[idx]
#         stack_size = min(self.stack_size, species.count)
#         array_index = species.index + random.randint(0, species.count-stack_size)

#         with torch.no_grad():
#             data = np.array(self.array[array_index:array_index+stack_size, :], copy=False)
#             embedding = torch.tensor(data, dtype=torch.float16)
#             del data

#         seq_detail = self.seqtree[species.accession+":0"]
#         node_id = int(seq_detail.node_id)
#         del seq_detail
        
#         return embedding, node_id


# @dataclass
# class VanjariStackDataModule(L.LightningDataModule):
#     seqtree: SeqTree
#     array:np.memmap|np.ndarray
#     accession_to_array_index:dict[str,int]
#     batch_size: int = 1
#     num_workers: int = 0
#     max_items:int = 0
#     seed:int = 42
#     validation_proportion:float = 0.2
#     stack_size:int = 16
#     validation_partition:int=0

#     def __post_init__(self):
#         super().__init__()

#     def setup(self, stage=None):
#         current_accession = None
#         start_index = 0
#         self.training = []
#         self.validation = []

#         random.seed(self.seed)

#         for accession, index in self.accession_to_array_index.items():
#             species_accession = accession.split(":")[0]
#             if current_accession != species_accession:
#                 if current_accession:
#                     seq_detail = self.seqtree[species_accession+":0"]
#                     current_list = self.validation if seq_detail.partition == self.validation_partition else self.training
#                     # current_list = self.validation if random.random() < self.validation_proportion else self.training
#                     current_list.append(Species(accession=current_accession, index=start_index, count=index-start_index))
#                 current_accession = species_accession
#                 start_index = index
        
#         seq_detail = self.seqtree[species_accession+":0"]
#         current_list = self.validation if seq_detail.partition == self.validation_partition else self.training
#         #current_list = self.validation if random.random() < self.validation_proportion else self.training
#         current_list.append(Species(accession=current_accession, index=start_index, count=index-start_index))

#         if self.max_items:
#             self.training = self.training[:self.max_items]
#             self.validation = self.validation[:self.max_items]

#         self.train_dataset = self.create_dataset(self.training)
#         self.val_dataset = self.create_dataset(self.validation)

#     def create_dataset(self, species:list[Species]) -> VanjariStackTrainingDataset:
#         return VanjariStackTrainingDataset(
#             species=species,
#             seqtree=self.seqtree, 
#             array=self.array,
#             stack_size=self.stack_size,
#         )
    
#     def train_dataloader(self):
#         print('train dataloader', self.num_workers)
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

#     def val_dataloader(self):
#         print('val_dataloader', self.num_workers)
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)