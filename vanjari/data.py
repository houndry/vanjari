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
from rich.progress import Progress
from bloodhound.data import read_memmap
from bloodhound.embedding import generate_overlapping_intervals

import pyfastx

console = Console()

from .nucleotidetransformer import NucleotideTransformerEmbedding 


def get_vmr(local_path) -> Path:
    url = "https://ictv.global/sites/default/files/VMR/VMR_MSL39.v4_20241106.xlsx"
    return cached_download(url, local_path)


@dataclass(kw_only=True)
class Species():
    accession:str
    index:int
    count:int


@dataclass(kw_only=True)
class Stack():
    start:int
    end:int


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


@dataclass(kw_only=True)
class VanjariStackPredictionDataset(Dataset):
    stacks: list[Stack]
    array:np.memmap|np.ndarray

    def __len__(self):
        return len(self.stacks)

    def __getitem__(self, idx):
        stack = self.stacks[idx]
        with torch.no_grad():
            data = np.array(self.array[stack.start:stack.end, :], copy=False)
            embedding = torch.tensor(data, dtype=torch.float16)
            del data
        
        return (embedding, )


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
    train_all:bool = False

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

        if self.train_all:
            self.training += self.validation

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


@dataclass(kw_only=True)
class VanjariNTPredictionDataset(Dataset):
    array:np.memmap|np.ndarray

    def __len__(self):
        return len(self.array)

    def __getitem__(self, array_index):
        with torch.no_grad():
            data = np.array(self.array[array_index, :], copy=False)
            embedding = torch.tensor(data, dtype=torch.float16)
            del data
        return embedding, 0 # .unsqueeze(dim=0)


def build_memmap_array(
    input:list[Path],
    memmap_array_path:Path=None,
    memmap_index:Path=None,
    model_name:str="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    length:int=1000,
) -> tuple[np.memmap, list[str]]:
    # Get list of fasta files
    base_extensions = {".fa", ".fasta", ".fna"}

    # Function to check if a file matches allowed extensions (including .gz)
    def matches_extensions(file: Path):
        return (
            file.suffix in base_extensions or
            (file.suffix == ".gz" and any(file.stem.endswith(ext) for ext in base_extensions))
        )

    # Expand the list
    files = []
    for path in input:
        if path.is_dir():
            # If it's a directory, find all files with the specified extensions
            files.extend([file for file in path.rglob("*") if matches_extensions(file)])
        else:
            # If it's not a directory, add the file to the list
            if matches_extensions(path):
                files.append(path)

    # TODO get from module.hparams.embedding_model
    embedding_model = NucleotideTransformerEmbedding()
    embedding_model.setup(model_name=model_name)

    # TODO GET length from module.hparams

    dtype = 'float16'

    # Get count of sequences
    index = 0
    for file in files:
        # Read the sequence
        for _, seq in pyfastx.Fasta(str(file), build_index=False):
            seq = seq.replace("N","")
            for ii, chunk in enumerate(range(0, len(seq), length)):
                # Add the sequence to the SeqTree
                index += 1
    count = index

    assert memmap_index is not None # hack

    index = 0
    assert memmap_array_path is not None # hack
    if not memmap_array_path.exists() or not memmap_index.exists():
        memmap_index.parent.mkdir(parents=True, exist_ok=True)
        memmap_array = None
        with Progress() as progress:
            task = progress.add_task("Processing...", total=count)
            with open(memmap_index, "w") as f: 
                for file in files:
                    print(file)
                    for accession, seq in pyfastx.Fasta(str(file), build_index=False):
                        seq = seq.replace("N","")
                        for ii, chunk in enumerate(range(0, len(seq), length)):
                            subseq = seq[chunk:chunk+length]

                            key = f"{accession}:{ii}"

                            try:
                                embedding = embedding_model.embed(subseq)
                            except Exception as err:
                                print(f"{key}: {err}\n{subseq}")
                                continue
                            
                            if memmap_array is None:
                                shape = (count, len(embedding))
                                memmap_array_path.parent.mkdir(parents=True, exist_ok=True)
                                memmap_array = np.memmap(memmap_array_path, dtype=dtype, mode='w+', shape=shape)

                            memmap_array[index,:] = embedding.half().numpy()
                            
                            print(key, file=f)

                            index += 1
                            progress.update(task, completed=index)
    else:
        accessions = memmap_index.read_text().strip().split("\n")
        count = len(accessions)
        memmap_array = read_memmap(memmap_array_path, count, dtype)

    accessions = memmap_index.read_text().strip().split("\n")

    return memmap_array, accessions



def build_species_list(accessions:list[str]) -> list[Species]:
    species_list = []
    current_species = None
    species_index_start = 0
    for index, accession in enumerate(accessions):
        species_accession = accession.split(":")[0]
        if current_species is None:
            current_species = species_accession

        # Create new stack if we have a new species or if we get to the stack size
        if current_species != species_accession:
            species_list.append(Species(accession=current_species, index=species_index_start, count=index-species_index_start))
            species_index_start = index
            current_species = species_accession

    # Create a new stack at the end of the loop
    species_list.append(Species(accession=current_species, index=species_index_start, count=index-species_index_start))

    return species_list


def build_stacks(accessions:list[str], stack_size:int=32, overlap:int=8) -> tuple[list[Stack], list[str]]:
    species_list = build_species_list(accessions)
    stacks = []
    species_names = []
    for species in species_list:
        species_name = species.accession
        if species.count <= stack_size:
            stacks.append(Stack(start=species.index, end=species.index+species.count))
            species_names.append(species_name)
        else:
            intervals = generate_overlapping_intervals(species.count, stack_size, overlap)
            for interval in intervals:
                stacks.append(Stack(start=species.index+interval[0], end=species.index+interval[1]))
                species_names.append(species_name)
    return stacks, species_names