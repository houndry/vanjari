import random
import torch
import torchapp as ta
from corgi.apps import Corgi
from corgi.seqtree import SeqTree
from torchapp.download import cached_download
from pathlib import Path
import pandas as pd
from seqbank import SeqBank
from rich.progress import track
from dataclasses import dataclass
from torch.utils.data import DataLoader


from hierarchicalsoftmax.metrics import RankAccuracyTorchMetric, GreedyAccuracy
from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions, render_probabilities

from torchmetrics import Metric
import numpy as np
from torch.utils.data import Dataset
import pyfastx

from bloodhound.embedding import generate_overlapping_intervals
from bloodhound.apps import Bloodhound
from bloodhound.data import read_memmap

from hierarchicalsoftmax import SoftmaxNode

from .nucleotidetransformer import NucleotideTransformerEmbedding 
from .data import VanjariStackDataModule
from .models import VanjariAttentionModel

class Vanjari(ta.TorchApp):
    @ta.method    
    def metrics(self) -> list[tuple[str,Metric]]:
        rank_accuracy = RankAccuracyTorchMetric(
            root=self.classification_tree, 
            ranks={1+i:f"rank_{i}" for i in range(9)},
        )
        return [('species_accuracy', GreedyAccuracy(root=self.classification_tree, name="species_accuracy")), ('rank_accuracy', rank_accuracy)]

    @ta.method
    def monitor(self) -> str:
        return "species_accuracy"

    @ta.tool
    def max_depth(
        self, 
        seqtree:Path=ta.Param(..., help="Path to the SeqTree"), 
    ):
        seqtree = SeqTree.load(Path(seqtree))
        max_depth = 0
        for leaf in track(seqtree.classification_tree.leaves):
            max_depth = max(max_depth, len(leaf.ancestors))
        print(f"Max depth: {max_depth}")

    def taxonomy_df(self, max_accessions:int=0):
        url = "https://ictv.global/sites/default/files/VMR/VMR_MSL39.v4_20241106.xlsx"
        local_path = self.cache_dir() / "VMR_MSL39.v4_20241106.xlsx"
        cached_download(url, local_path)
        df = pd.read_excel(local_path, sheet_name="VMR MSL39")
        df = df.fillna('')
        if max_accessions:
            df = df.head(max_accessions)
        return df


class VanjariCorgi(Vanjari, Corgi):
    @ta.tool
    def preprocess(
        self, 
        seqtree:Path=ta.Param(..., help="Path to save the SeqTree"), 
        seqbank:Path=ta.Param(..., help="Path to save the SeqBank"),
        max_accessions:int=ta.Param(0, help="Maximum number of accessions to add"),
        fasta_dir:Path=ta.Param(..., help="Path to the FASTA directory"),
    ):
        seqtree_path = Path(seqtree)
        seqbank_path = Path(seqbank)
        fasta_dir = Path(fasta_dir)

        # Build the SeqBank
        seqbank_path.parent.mkdir(parents=True, exist_ok=True)
        seqbank = SeqBank(path=seqbank_path, write=True)

        df = self.taxonomy_df(max_accessions)
        taxonomic_columns = [
            'Realm', 'Subrealm',
            'Kingdom', 'Subkingdom', 'Phylum', 'Subphylum', 'Class', 'Subclass',
            'Order', 'Suborder', 'Family', 'Subfamily', 'Genus', 'Subgenus',
            'Species',
        ]
        root = SoftmaxNode(name="Virus", rank="Root")
        seqtree = SeqTree(root)
        
        print("Building classification tree")
        for _, row in track(df.iterrows(), total=len(df)):
            current_node = root
            genbank_accession = row['Virus GENBANK accession'].strip()
            if not genbank_accession:
                continue

            for rank in taxonomic_columns:
                value = row[rank]
                if not value:
                    continue

                found = False
                for child in current_node.children:
                    if child.name == value:
                        found = True
                        break
                current_node = child if found else SoftmaxNode(name=value, parent=current_node, rank=rank) 

            
            accessions = genbank_accession.split(";")
            for accession in accessions:
                accession = accession.strip()
                if ":" in accession:
                    accession = accession.split(":")[1].strip()
                accession = accession.split(" ")[0]

                # Add the sequence to the SeqBank
                fasta = fasta_dir / f"{accession}.fasta.gz"
                assert fasta.exists(), f"FASTA file not found: {fasta}"
                try:
                    seqbank.add_sequence_from_file(accession, fasta)
                except RuntimeError as e:
                    print(f"Error adding {accession}: {e}")
                    continue

                # Add the sequence to the SeqTree
                partition = random.randint(0, 4)       
                seqtree.add(accession, current_node, partition)

        root.render(filepath="viruses.dot")
        with open("tree.txt", "w") as f:
            f.write(str(root.render()))

        # Save the SeqTree
        seqtree_path.parent.mkdir(parents=True, exist_ok=True)
        seqtree.save(seqtree_path)


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
        return embedding


class VanjariNT(Vanjari, Bloodhound):
    @ta.tool
    def preprocess(
        self, 
        seqtree:Path=ta.Param(..., help="Path to save the SeqTree"), 
        output_dir:Path=ta.Param(default=..., help="A directory to store the output which includes the memmap array, the listing of accessions and an error log."),
        max_accessions:int=ta.Param(0, help="Maximum number of accessions to add"),
        fasta_dir:Path=ta.Param(..., help="Path to the FASTA directory"),
        model_name:str="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",  
        length:int=1000,      
    ):
        seqtree_path = Path(seqtree)
        fasta_dir = Path(fasta_dir)

        model = NucleotideTransformerEmbedding()
        model.setup(model_name=model_name)

        df = self.taxonomy_df(max_accessions)
        taxonomic_columns = [
            'Realm', 'Subrealm',
            'Kingdom', 'Subkingdom', 'Phylum', 'Subphylum', 'Class', 'Subclass',
            'Order', 'Suborder', 'Family', 'Subfamily', 'Genus', 'Subgenus',
            'Species',
        ]
        root = SoftmaxNode(name="Virus", rank="Root")
        seqtree = SeqTree(root)
        
        print("Building classification tree")
        # 
        index = 0
        # for _, row in df.iterrows():
        for _, row in track(df.iterrows(), total=len(df)):
            genbank_accession = row['Virus GENBANK accession'].strip()
            if not genbank_accession:
                continue

            accessions = genbank_accession.split(";")
            for accession in accessions:
                accession = accession.strip()
                if ":" in accession:
                    accession = accession.split(":")[1].strip()
                accession = accession.split(" ")[0]

                # Add the sequence to the SeqBank
                fasta = fasta_dir / f"{accession}.fasta.gz"
                assert fasta.exists(), f"FASTA file not found: {fasta}"

                # Read the sequence
                for _, seq in pyfastx.Fasta(str(fasta), build_index=False):
                    seq = seq.replace("N","")
                    for ii, chunk in enumerate(range(0, len(seq), length)):
                        # Add the sequence to the SeqTree
                        index += 1

        # root.render(filepath="viruses.dot")
        # with open("tree.txt", "w") as f:
        #     f.write(str(root.render()))

        count = index
        print("count", count)

        dtype = 'float16'


        index = 0
        output_dir.mkdir(exist_ok=True, parents=True)
        memmap_array = None
        with open(output_dir/f"{output_dir.name}.txt", "w") as f: 
            for _, row in track(df.iterrows(), total=len(df)):

                genbank_accession = row['Virus GENBANK accession'].strip()
                if not genbank_accession:
                    continue

                current_node = root
                for rank in taxonomic_columns:
                    value = row[rank]
                    if not value:
                        continue

                    found = False
                    for child in current_node.children:
                        if child.name == value:
                            found = True
                            break

                    current_node = child if found else SoftmaxNode(name=value, parent=current_node, rank=rank) 

                accessions = genbank_accession.split(";")
                for accession in accessions:
                    print(accession)
                    accession = accession.strip()
                    if ":" in accession:
                        accession = accession.split(":")[1].strip()
                    accession = accession.split(" ")[0]
                    fasta = fasta_dir / f"{accession}.fasta.gz"
                    for _, seq in pyfastx.Fasta(str(fasta), build_index=False):
                        seq = seq.replace("N","")
                        for ii, chunk in enumerate(range(0, len(seq), length)):
                            subseq = seq[chunk:chunk+length]

                            key = f"{accession}:{ii}"

                            try:
                                embedding = model.embed(subseq)
                            except Exception as err:
                                print(f"{key}: {err}\n{subseq}")
                                continue
                            
                            if memmap_array is None:
                                memmap_array_path = output_dir/f"{output_dir.name}.npy"
                                shape = (count, len(embedding))
                                memmap_array = np.memmap(memmap_array_path, dtype=dtype, mode='w+', shape=shape)

                            memmap_array[index,:] = embedding.half().numpy()
                            
                            print(key, file=f)
                            # print(key)
                            partition = random.randint(0, 4)       
                            seqtree.add(key, current_node, partition)

                            index += 1

        # Save the SeqTree
        seqtree_path.parent.mkdir(parents=True, exist_ok=True)
        seqtree.save(seqtree_path)

    @ta.method
    def prediction_dataloader(
        self,
        module,
        input:Path=ta.Param(help="A path to a directory of fasta files or a single fasta file."),
        extension='fasta',
        num_workers: int = 0,
        memmap_array_path:Path=None, # TODO explain
        memmap_index:Path=None, # TODO explain
        batch_size:int = 16,
        model_name:str="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",  # hack
        length:int=1000, # hack
    ) -> DataLoader:
        # Get list of fasta files
        files = []
        input = Path(input)
        if input.is_dir():
            for path in input.rglob(f"*.{extension}"):
                files.append(str(path))
        else:
            files.append(str(input))

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
        self.memmap_index = memmap_index

        assert memmap_array_path is not None # hack
        if not memmap_array_path.exists() or not memmap_index.exists():
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
                                memmap_array = np.memmap(memmap_array_path, dtype=dtype, mode='w+', shape=shape)

                            memmap_array[index,:] = embedding.half().numpy()
                            
                            print(key, file=f)

                            index += 1
        else:
            memmap_index_data = memmap_index.read_text().strip().split("\n")
            count = len(memmap_index_data)
            memmap_array = read_memmap(memmap_array_path, count, dtype)

        dataset = VanjariNTPredictionDataset(array=memmap_array)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        return dataloader

    @ta.method
    def output_results(
        self, 
        results, 
        output_csv: Path = ta.Param(default=None, help="A path to output the results as a CSV."),
        image_dir: Path = ta.Param(default=None, help="A path to output the results as images."),
        image_threshold:float = 0.005,
        prediction_threshold:float = ta.Param(default=0.0, help="The threshold value for making hierarchical predictions."),
        **kwargs,
    ):
        assert self.classification_tree

        classification_probabilities = node_probabilities(results, root=self.classification_tree)
        
        category_names = [self.node_to_str(node) for node in self.classification_tree.node_list_softmax if not node.is_root]

        results_df = pd.DataFrame(classification_probabilities.numpy(), columns=category_names)

        # Average over chunks
        results_df["original_index"] = results_df.index
        keys = self.memmap_index.read_text().strip().split("\n")
        results_df["SequenceID"] = [key.split(":")[0] for key in keys]
        results_df = results_df.groupby(["SequenceID"]).mean().reset_index()

        # sort to get original order
        results_df = results_df.sort_values(by="original_index").drop(columns=["original_index"])
        
        classification_probabilities = torch.as_tensor(results_df[category_names].to_numpy()) 

        # get greedy predictions which can use the raw activation or the softmax probabilities
        predictions = greedy_predictions(
            classification_probabilities, 
            root=self.classification_tree, 
            threshold=prediction_threshold,
        )

        # Sort out headers = 
        header_string = "SequenceID,Realm (-viria),Realm_score,Subrealm (-vira),Subrealm_score,Kingdom (-virae),Kingdom_score,Subkingdom (-virites),Subkingdom_score,Phylum (-viricota),Phylum_score,Subphylum (-viricotina),Subphylum_score,Class (-viricetes),Class_score,Subclass (-viricetidae),Subclass_score,Order (-virales),Order_score,Suborder (-virineae),Suborder_score,Family (-viridae),Family_score,Subfamily (-virinae),Subfamily_score,Genus (-virus),Genus_score,Subgenus (-virus),Subgenus_score,Species (binomial),Species_score"
        header_names = header_string.split(",")

        rank_to_header = {header.split(" ")[0]:header for header in header_names[1::2]}

        output_df = pd.DataFrame(columns=header_names)
        output_df[:] = "NA"

        for index, node in enumerate(predictions):
            output_df.loc[index, "SequenceID"] = results_df.loc[index, "SequenceID"]

            for ancestor in node.ancestors:
                header = rank_to_header[ancestor.rank]
                output_df.loc[index, header] = ancestor.name
                output_df.loc[index, ancestor.rank+"_score"] = results_df.loc[index, ancestor.name]

        if output_csv:
            print(f"Writing inference results to: {output_csv}")
            output_df.to_csv(output_csv, index=False)

        # Output images
        if image_dir:
            print(f"Writing inference probability renders to: {image_dir}")
            image_dir = Path(image_dir)
            image_paths = [image_dir/f"{name}.png" for name in results_df["SequenceID"]]
            render_probabilities(
                root=self.classification_tree, 
                filepaths=image_paths,
                probabilities=classification_probabilities,
                predictions=predictions,
                threshold=image_threshold,
            )

        return output_df


class VanjariStack(VanjariNT):
    @ta.method    
    def data(
        self,
        max_items:int=0,
        num_workers:int=0,
        stack_size:int=16,
        validation_csv:Path=ta.Param(..., help="Path to the validation CSV file"),
    ) -> VanjariStackDataModule:
        validation_df = pd.read_csv(validation_csv)
        validation_accessions = set(validation_df['SequenceID'])
        for accession, detail in self.seqtree.items():
            if accession in validation_accessions:
                detail['partition']


        return VanjariStackDataModule(
            array=self.array,
            accession_to_array_index=self.accession_to_array_index,
            seqtree=self.seqtree,
            max_items=max_items,
            num_workers=num_workers,
            stack_size=stack_size,
            validation_accessions=validation_accessions,
        )

    @ta.method
    def model(
        self,
        features:int=1024,
        intermediate_layers:int=2,
        growth_factor:float=2.0,
        dropout:float=0.0,
    ) -> VanjariAttentionModel:
        return VanjariAttentionModel(
            classification_tree=self.classification_tree,
            features=features,
            intermediate_layers=intermediate_layers,
            growth_factor=growth_factor,
            dropout=dropout,
        )
    
    @ta.method
    def input_count(self) -> int:
        return 1

    # @method
    # def extra_hyperparameters(self, embedding_model:str="", max_length:int=None) -> dict:
    #     """ Extra hyperparameters to save with the module. """
    #     assert embedding_model, f"Please provide an embedding model."
    #     embedding_model = embedding_model.lower()
    #     if embedding_model.startswith("esm"):
    #         layers = embedding_model[3:].strip()
    #         embedding_model = ESMEmbedding(max_length=max_length)
    #         embedding_model.setup(layers=layers)
    #     else:
    #         raise ValueError(f"Cannot understand embedding model: {embedding_model}")

    #     return dict(
    #         embedding_model=embedding_model,
    #         classification_tree=self.seqtree.classification_tree,
    #         gene_id_dict=self.gene_id_dict,
    #     )
