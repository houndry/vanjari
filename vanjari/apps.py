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
from torch.utils.data import DataLoader
from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions, render_probabilities
from torchmetrics import Metric
import numpy as np
import pyfastx

from bloodhound.apps import Bloodhound
from bloodhound.data import read_memmap

from hierarchicalsoftmax import SoftmaxNode

from .nucleotidetransformer import NucleotideTransformerEmbedding 
from .data import VanjariStackDataModule, VanjariNTPredictionDataset, build_memmap_array, Stack, VanjariStackPredictionDataset
from .models import VanjariAttentionModel
from .metrics import ICTVTorchMetric, RANKS

class Vanjari(ta.TorchApp):
    @ta.tool
    def evaluate_csv(
        self, 
        prediction:Path=None, # TODO explain
        truth:Path=None, # TODO explain
        stem_only:bool=True, # TODO explain
    ):
        prediction_df = pd.read_csv(prediction)
        truth_df = pd.read_csv(truth)

        if stem_only:
            prediction_df['SequenceID'] = prediction_df['SequenceID'].apply(lambda x: x.split(".")[0])
            truth_df['SequenceID'] = truth_df['SequenceID'].apply(lambda x: x.split(".")[0])

        intersection = set(truth_df['SequenceID']) & set(prediction_df['SequenceID'])
        missing = set(truth_df['SequenceID']) - intersection

        print(f"Missing: {len(missing)}")

        truth_df = truth_df[truth_df['SequenceID'].isin(intersection)]
        prediction_df = prediction_df[prediction_df['SequenceID'].isin(intersection)]

        # sort
        prediction_df = prediction_df.sort_values(by="SequenceID").reset_index(drop=True)
        truth_df = truth_df.sort_values(by="SequenceID").reset_index(drop=True)

        assert all(prediction_df['SequenceID'] == truth_df['SequenceID'])

        header_string = "SequenceID,Realm (-viria),Realm_score,Subrealm (-vira),Subrealm_score,Kingdom (-virae),Kingdom_score,Subkingdom (-virites),Subkingdom_score,Phylum (-viricota),Phylum_score,Subphylum (-viricotina),Subphylum_score,Class (-viricetes),Class_score,Subclass (-viricetidae),Subclass_score,Order (-virales),Order_score,Suborder (-virineae),Suborder_score,Family (-viridae),Family_score,Subfamily (-virinae),Subfamily_score,Genus (-virus),Genus_score,Subgenus (-virus),Subgenus_score,Species (binomial),Species_score"        
        header_names = header_string.split(",")

        rank_to_header = {header.split(" ")[0]:header for header in header_names[1::2]}

        for rank in RANKS:
            column = rank_to_header[rank]
            data_available = (prediction_df[column] != "NA") & (truth_df[column] != "NA")
            result = (prediction_df[column][data_available] == truth_df[column][data_available]).mean()
            print( rank, result*100 )

    @ta.tool
    def filter_memmap(
        self, 
        memmap_array_path:Path=None, # TODO explain
        memmap_index:Path=None, # TODO explain
        output_array_path:Path=None, # TODO explain
        output_index:Path=None, # TODO explain
        filter_csv:Path=ta.Param(..., help="Path to the filter CSV file"),
    ):
        filter_df = pd.read_csv(filter_csv)
        filter_accessions = set(filter_df['SequenceID'])

        memmap_index_data = memmap_index.read_text().strip().split("\n")
        count = len(memmap_index_data)
        memmap_array = read_memmap(memmap_array_path, count, dtype='float16')

        new_accessions = [accession for accession in memmap_index_data if accession.split(":")[0] in filter_accessions]
        new_count = len(new_accessions)
        output_array_path.parent.mkdir(parents=True, exist_ok=True)
        new_array = np.memmap(output_array_path, dtype='float16', mode='w+', shape=(new_count, memmap_array.shape[1]))
        for ii, accession in enumerate(new_accessions):
            index = memmap_index_data.index(accession)
            new_array[ii,:] = memmap_array[index,:]
        
        output_index.parent.mkdir(parents=True, exist_ok=True)
        output_index.write_text("\n".join(new_accessions))

    @ta.method    
    def metrics(self) -> list[tuple[str,Metric]]:
        return [
            # ('species_accuracy', GreedyAccuracy(root=self.classification_tree, name="species_accuracy")), 
            ('rank_accuracy', ICTVTorchMetric(root=self.classification_tree)),
        ]

    @ta.method
    def monitor(self) -> str:
        return "Genus"

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

    def node_to_str(self, node:SoftmaxNode) -> str:
        """ 
        Converts the node to a string
        """
        return str(node).split(",")[-1].strip()


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

    @ta.method
    def output_results(
        self,
        results,
        output_csv: Path = ta.Param(default=None, help="A path to output the results as a CSV."),
        probability_csv: Path = ta.Param(default=None, help="A path to output the probabilities as a CSV."),
        image_dir: Path = ta.Param(default=None, help="A directory to output the results as images."),
        image_threshold:float = 0.005,
        prediction_threshold:float = ta.Param(default=0.5, help="The threshold value for making hierarchical predictions."),
        **kwargs,
    ):
        assert self.classification_tree # This should be saved from the learner
        
        classification_probabilities = node_probabilities(results[0], root=self.classification_tree)
        category_names = [self.node_to_str(node) for node in self.classification_tree.node_list_softmax if not node.is_root]
        chunk_details = pd.DataFrame(self.dataloader.chunk_details, columns=["file", "original_id", "chunk"])
        predictions_df = pd.DataFrame(classification_probabilities.numpy(), columns=category_names)

        results_df = pd.concat(
            [chunk_details.drop(columns=['chunk']), predictions_df],
            axis=1,
        )

        # Average over chunks
        results_df["chunk_index"] = results_df.index
        results_df = results_df.groupby(["file", "original_id"]).mean().reset_index()

        # sort to get original order
        results_df = results_df.sort_values(by="chunk_index").drop(columns=["chunk_index"]).reset_index()
        
        # Get new tensors now that we've averaged over chunks
        classification_probabilities = torch.as_tensor(results_df[category_names].to_numpy()) 

        # get greedy predictions which can use the raw activation or the softmax probabilities
        predictions = greedy_predictions(
            classification_probabilities, 
            root=self.classification_tree, 
            threshold=prediction_threshold,
        )

        results_df['greedy_prediction'] = [
            self.node_to_str(node)
            for node in predictions
        ]

        # Sort out headers = 
        header_string = "SequenceID,Realm (-viria),Realm_score,Subrealm (-vira),Subrealm_score,Kingdom (-virae),Kingdom_score,Subkingdom (-virites),Subkingdom_score,Phylum (-viricota),Phylum_score,Subphylum (-viricotina),Subphylum_score,Class (-viricetes),Class_score,Subclass (-viricetidae),Subclass_score,Order (-virales),Order_score,Suborder (-virineae),Suborder_score,Family (-viridae),Family_score,Subfamily (-virinae),Subfamily_score,Genus (-virus),Genus_score,Subgenus (-virus),Subgenus_score,Species (binomial),Species_score"
        header_names = header_string.split(",")

        rank_to_header = {header.split(" ")[0]:header for header in header_names[1::2]}

        output_df = pd.DataFrame(columns=header_names)
        output_df[:] = "NA"

        for index, node in enumerate(predictions):
            output_df.loc[index, "SequenceID"] = results_df.loc[index, "original_id"]
            current_probability = 1.0
            for ancestor in node.ancestors[1:] + (node,):
                if ancestor.rank == "Root":
                    continue
                header = rank_to_header[ancestor.rank]
                output_df.loc[index, header] = ancestor.name
                
                if ancestor.name in results_df.columns:
                    current_probability = results_df.loc[index, ancestor.name]

                output_df.loc[index, ancestor.rank+"_score"] = current_probability                    

        if output_csv:
            print(f"Writing inference results to: {output_csv}")
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_csv, index=False)

        # Output images
        if image_dir:
            print(f"Writing inference probability renders to: {image_dir}")
            image_dir = Path(image_dir)
            image_paths = [image_dir/f"{name}.png" for name in results_df["original_id"]]
            render_probabilities(
                root=self.classification_tree, 
                filepaths=image_paths,
                probabilities=classification_probabilities,
                predictions=predictions,
                threshold=image_threshold,
            )

        return output_df


class VanjariNT(Vanjari, Bloodhound):
    @ta.tool
    def set_validation_partition(
        self, 
        seqtree:Path=ta.Param(..., help="Path to the SeqTree"),
        validation_csv:Path=ta.Param(..., help="Path to the validation CSV file"),
        output:Path=ta.Param(..., help="Path to save the SeqTree"),
    ):
        seqtree = SeqTree.load(Path(seqtree))
        validation_df = pd.read_csv(validation_csv)
        validation_accessions = set(validation_df['SequenceID'])
        for accession, detail in seqtree.items():
            detail.partition = 0 if accession.split(":")[0] in validation_accessions else 1
        
        output.parent.mkdir(parents=True, exist_ok=True)
        seqtree.save(output)

    @ta.tool
    def taxonomy_csv(
        self, 
        csv:Path=ta.Param(..., help="Path to save the CSV"), 
        filter:Path=ta.Param(None, help="Path with accessions to use as a filter"),
    ):
        header_string = "SequenceID,Realm (-viria),Realm_score,Subrealm (-vira),Subrealm_score,Kingdom (-virae),Kingdom_score,Subkingdom (-virites),Subkingdom_score,Phylum (-viricota),Phylum_score,Subphylum (-viricotina),Subphylum_score,Class (-viricetes),Class_score,Subclass (-viricetidae),Subclass_score,Order (-virales),Order_score,Suborder (-virineae),Suborder_score,Family (-viridae),Family_score,Subfamily (-virinae),Subfamily_score,Genus (-virus),Genus_score,Subgenus (-virus),Subgenus_score,Species (binomial),Species_score"        
        header_names = header_string.split(",")

        rank_to_header = {header.split(" ")[0]:header for header in header_names[1::2]}

        if filter:
            filter = set(Path(filter).read_text().strip().split("\n"))

        df = self.taxonomy_df()
        taxonomic_columns = [
            'Realm', 'Subrealm',
            'Kingdom', 'Subkingdom', 'Phylum', 'Subphylum', 'Class', 'Subclass',
            'Order', 'Suborder', 'Family', 'Subfamily', 'Genus', 'Subgenus',
            'Species',
        ]
        
        data = []
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

                if filter and accession not in filter:
                    continue

                insert_row = dict(SequenceID=accession)
                for rank in taxonomic_columns:
                    value = row[rank]
                    score = 1.0
                    if not value:
                        value = "NA"
                        score = "NA"

                    insert_row[rank_to_header[rank]] = value
                    insert_row[rank+"_score"] = score

                data.append(insert_row)

        output_df = pd.DataFrame(data, columns=header_names)

        csv.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing taxonomy to {csv}")
        output_df.to_csv(csv, index=False)

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
    def build_dataset_sequence_ids(self, memmap_array, accessions, **kwargs):
        sequence_ids = [accession.split(":")[0] for accession in accessions]
        dataset = VanjariNTPredictionDataset(array=memmap_array)
        return dataset, sequence_ids

    @ta.method("build_dataset_sequence_ids")
    def prediction_dataloader(
        self,
        module,
        input:Path=ta.Param(help="A path to a directory of fasta files or a single fasta file."),
        extension='fasta',
        memmap_array_path:Path=None, # TODO explain
        memmap_index:Path=None, # TODO explain
        model_name:str="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",  # hack
        length:int=1000, # hack
        batch_size:int = 16,
        num_workers: int = 0,
    ) -> DataLoader:

        self.classification_tree = module.hparams.classification_tree

        memmap_array, accessions = build_memmap_array(
            input=input,
            extension=extension,
            memmap_array_path=memmap_array_path,
            memmap_index=memmap_index,
            model_name=model_name,
            length=length,
        )
        dataset, self.sequence_ids = self.build_dataset_sequence_ids(memmap_array, accessions)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        return dataloader

    @ta.method
    def output_results(
        self, 
        results, 
        output_csv: Path = ta.Param(default=None, help="A path to output the results as a CSV."),
        probability_csv: Path = ta.Param(default=None, help="A path to output the probabilities as a CSV."),
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
        results_df["SequenceID"] = self.sequence_ids
        results_df = results_df.groupby(["SequenceID"]).mean().reset_index()

        # sort to get original order
        results_df = results_df.sort_values(by="original_index").drop(columns=["original_index"]).reset_index()

        if probability_csv:
            probability_csv.parent.mkdir(parents=True, exist_ok=True)
            print(f"Writing probabilities to {probability_csv}")
            results_df.to_csv(probability_csv, index=False)
        
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
            current_probability = 1.0
            for ancestor in node.ancestors[1:] + (node,):
                header = rank_to_header[ancestor.rank]
                output_df.loc[index, header] = ancestor.name
                
                if ancestor.name in results_df.columns:
                    current_probability = results_df.loc[index, ancestor.name]

                output_df.loc[index, ancestor.rank+"_score"] = current_probability                    

        if output_csv:
            print(f"Writing inference results to: {output_csv}")
            output_csv.parent.mkdir(parents=True, exist_ok=True)
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
        validation_partition:int=0,
        seed:int = 42,
    ) -> VanjariStackDataModule:

        return VanjariStackDataModule(
            array=self.array,
            accession_to_array_index=self.accession_to_array_index,
            seqtree=self.seqtree,
            max_items=max_items,
            num_workers=num_workers,
            stack_size=stack_size,
            seed=seed,
            validation_partition=validation_partition,
        )

    @ta.method
    def model(
        self,
        features:int=ta.Param(
            default=1024,
            help="The size of the initial layer after the embedding.",
            tune=True,
            log=True,
            tune_min=256,
            tune_max=2048,
        ),
        intermediate_layers:int=ta.Param(
            default=2,
            help="The number of intermediate layers.",
            tune=True,
            tune_min=0,
            tune_max=4,
        ),
        growth_factor:float=ta.Param(
            default=2.0,
            help="The factor to multiply the initial layers.",
            tune=True,
            tune_min=1.0,
            tune_max=2.5,
        ),
        dropout:float=ta.Param(
            default=0.0,
            help="The amount of dropout.",
            tune=True,
            tune_min=0.0,
            tune_max=0.8,
        ),
        attention_hidden_size:int=ta.Param(
            default=512,
            help="The size of the initial layer after the embedding.",
            tune=True,
            log=True,
            tune_min=256,
            tune_max=2048,
        ),
    ) -> VanjariAttentionModel:
        return VanjariAttentionModel(
            classification_tree=self.classification_tree,
            features=features,
            intermediate_layers=intermediate_layers,
            growth_factor=growth_factor,
            dropout=dropout,
            attention_hidden_size=attention_hidden_size,
        )
    
    @ta.method
    def input_count(self) -> int:
        return 1

    @ta.method
    def build_dataset_sequence_ids(self, memmap_array, accessions, stack_size:int=16, **kwargs):
        sequence_ids = []
        stacks = []
        current_species = None
        current_stack_start = None
        for index, accession in enumerate(accessions):
            species_accession = accession.split(":")[0]
            if current_stack_start is None:
                current_stack_start = index

            if current_species is None:
                current_species = species_accession

            # Create new stack if we have a new species or if we get to the stack size
            if current_species != species_accession or len(index-current_stack_start) >= stack_size:
                stacks.append(Stack(start=current_stack_start, end=index))
                sequence_ids.append(species_accession)
                current_stack_start = index
                current_species = species_accession

        # Create a new stack at the end of the loop
        stacks.append(Stack(start=current_stack_start, end=index))
        sequence_ids.append(species_accession)

        dataset = VanjariStackPredictionDataset(array=memmap_array, stacks=stacks)
        return dataset, sequence_ids

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
