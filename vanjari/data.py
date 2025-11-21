from pathlib import Path
import random
from zlib import adler32
from torchapp.download import cached_download
import torch
import numpy as np
from pathlib import Path
import lightning as L
from hierarchicalsoftmax.treedict import TreeDict
from torch.utils.data import DataLoader
from rich.console import Console
from torch.utils.data import Dataset
from dataclasses import dataclass
from rich.progress import Progress
from barbet.data import read_memmap
from barbet.embedding import generate_overlapping_intervals

import pyfastx

console = Console()

from .nucleotidetransformer import NucleotideTransformerEmbedding 

COMPLEMENT_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")

def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT_TABLE)[::-1]


def get_vmr(local_path) -> Path:
    url = "https://ictv.global/sites/default/files/VMR/VMR_MSL39.v4_20241106.xlsx"
    return cached_download(url, local_path)


@dataclass(kw_only=True)
class Genome():
    accession:str
    index:int
    count:int


@dataclass(kw_only=True)
class Stack():
    start:int
    end:int


@dataclass(kw_only=True)
class VanjariStackTrainingDataset(Dataset):
    genomes: list[Genome]
    treedict: TreeDict
    array:np.memmap|np.ndarray
    stack_size:int = 16
    deterministic:bool = False

    def __len__(self):
        return len(self.genomes)

    def __getitem__(self, idx):
        genome = self.genomes[idx]
        stack_size = min(self.stack_size, genome.count)
        rng = np.random.RandomState(adler32(genome.accession.encode("ascii"))) if self.deterministic else np.random
        array_index = genome.index + rng.randint(0, genome.count-stack_size+1)

        with torch.no_grad():
            data = np.array(self.array[array_index:array_index+stack_size, :], copy=False)
            embedding = torch.tensor(data, dtype=torch.float16)
            del data

        seq_detail = self.treedict[genome.accession+":0"]
        node_id = int(seq_detail.node_id)
        del seq_detail
        
        # assert embedding.shape[-1] == 1024, f"Embedding shape restricted to 1024, here is {embedding.shape[-1]}"
        
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
        
        return embedding


@dataclass
class VanjariStackDataModule(L.LightningDataModule):
    treedict: TreeDict
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

        # assert self.array.shape[-1] == 1024, f"Embedding shape restricted to 1024, here is {self.array.shape[-1]}"

        random.seed(self.seed)

        current_list = None

        for accession, index in self.accession_to_array_index.items():
            if isinstance(index, list):
                assert len(index) == 1
                index = index[0]
            genome_accession = accession.split(":")[0]
            if current_accession != genome_accession:
                if current_list is not None:
                    current_list.append(Genome(accession=current_accession, index=start_index, count=index-start_index))
                current_accession = genome_accession
                start_index = index

                detail = self.treedict[accession]
                current_list = self.validation if detail.partition == self.validation_partition else self.training
        
        current_list.append(Genome(accession=current_accession, index=start_index, count=index-start_index))

        if self.max_items:
            self.training = self.training[:self.max_items]
            self.validation = self.validation[:self.max_items]

        if self.train_all:
            self.training += self.validation

        self.train_dataset = self.create_dataset(self.training, deterministic=False)
        self.val_dataset = self.create_dataset(self.validation, deterministic=True)

    def create_dataset(self, genomes:list[Genome], deterministic:bool) -> VanjariStackTrainingDataset:
        return VanjariStackTrainingDataset(
            genomes=genomes,
            treedict=self.treedict, 
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
    model_name:str="",
    force:bool=False,
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

    # If input is a string or Path, convert it to a list
    if isinstance(input, (str,Path)):
        input = [Path(input)]

    # Expand the list
    files = []
    for path in input:
        path = Path(path)
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

    dtype = 'float16'

    # Get count of sequences
    index = 0
    for file in files:
        # Read the sequence
        for _, seq in pyfastx.Fasta(str(file), build_index=False):
            seq = seq.replace("N","")
            for ii, chunk in enumerate(range(0, len(seq), length)):
                # Add the sequence to the TreeDict
                index += 1
    count = index

    assert memmap_index is not None # hack
    assert memmap_array_path is not None # hack

    if not memmap_array_path.exists() or not memmap_index.exists() or force:
        index = 0
        embedding = embedding_model.embed("A"*length)
        shape = (count*2, len(embedding))

        memmap_array_path.parent.mkdir(parents=True, exist_ok=True)
        memmap_array = np.memmap(memmap_array_path, dtype=dtype, mode='w+', shape=shape)
        memmap_index.parent.mkdir(parents=True, exist_ok=True)

        def add_embedding(subseq, key, file) -> bool:
            try:
                embedding = embedding_model.embed(subseq)
                memmap_array[index,:] = embedding.half().numpy()
                print(key, file=file)
            except Exception as err:
                print(f"{key}: {err}\n{subseq}")
                return False
            
            return True

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

                            index += add_embedding(subseq, key, file=f)
                            index += add_embedding(reverse_complement(subseq), key + "r", file=f)                            

                            progress.update(task, completed=index)
    else:
        accessions = memmap_index.read_text().strip().split("\n")
        count = len(accessions)
        memmap_array = read_memmap(memmap_array_path, count, dtype)

    accessions = memmap_index.read_text().strip().split("\n")

    return memmap_array, accessions


def build_genome_list(accessions:list[str]) -> list[Genome]:
    genome_list = []
    current_genome = None
    genome_index_start = 0
    for index, accession in enumerate(accessions):
        genome_accession = accession.split(":")[0]
        if current_genome is None:
            current_genome = genome_accession

        # Create new stack if we have a new genome or if we get to the stack size
        if current_genome != genome_accession:
            genome_list.append(Genome(accession=current_genome, index=genome_index_start, count=index-genome_index_start))
            genome_index_start = index
            current_genome = genome_accession

    # Create a new stack at the end of the loop
    genome_list.append(Genome(accession=current_genome, index=genome_index_start, count=index-genome_index_start))

    return genome_list


def build_stacks(accessions:list[str], stack_size:int=32, overlap:int=8) -> tuple[list[Stack], list[str]]:
    genome_list = build_genome_list(accessions)
    stacks = []
    genome_names = []
    for genome in genome_list:
        genome_name = genome.accession
        if genome.count <= stack_size:
            stacks.append(Stack(start=genome.index, end=genome.index+genome.count))
            genome_names.append(genome_name)
        else:
            intervals = generate_overlapping_intervals(genome.count, stack_size, overlap)
            for interval in intervals:
                stacks.append(Stack(start=genome.index+interval[0], end=genome.index+interval[1]))
                genome_names.append(genome_name)
    return stacks, genome_names


def assign_partition(name: str, seed: int = 42, partitions: int = 5) -> int:
    import hashlib

    # Combine the seed with the name to make the bin assignment configurable
    combined = f"{seed}-{name}"
    # Use a stable hash function (e.g., SHA256)
    hash_bytes = hashlib.sha256(combined.encode('utf-8')).digest()
    # Convert to int and map to bin
    return int.from_bytes(hash_bytes, 'big') % partitions
