from enum import Enum
import typer
import torch
from torchapp.cli import method
from bloodhound.embedding import Embedding
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

from .dnaembedding import DNAEmbedding

# class NucleotideTransformerLayers(Enum):
#     T6 = "6"
#     T12 = "12"
#     T30 = "30"
#     T33 = "33"
#     T36 = "36"
#     T48 = "48"

#     @classmethod
#     def from_value(cls, value: int|str) -> "NucleotideTransformerLayers":
#         for layer in cls:
#             if layer.value == str(value):
#                 return layer
#         return None
    
#     def __int__(self):
#         return int(self.value)
    
#     def __str__(self):
#         return str(self.value)

#     def model_name(self) -> str:
#         match self:
#             case NucleotideTransformerLayers.T48:
#                 return "NucleotideTransformer2_t48_15B_UR50D"
#             case NucleotideTransformerLayers.T36:
#                 return "NucleotideTransformer2_t36_3B_UR50D"
#             case NucleotideTransformerLayers.T33:
#                 return "NucleotideTransformer2_t33_650M_UR50D"
#             case NucleotideTransformerLayers.T30:
#                 return "NucleotideTransformer2_t30_150M_UR50D"
#             case NucleotideTransformerLayers.T12:
#                 return "NucleotideTransformer2_t12_35M_UR50D"
#             case NucleotideTransformerLayers.T6:
#                 return "NucleotideTransformer2_t6_8M_UR50D"

#     def get_model_alphabet(self) -> tuple["NucleotideTransformer2", "Alphabet"]:
#         return torch.hub.load("facebookresearch/NucleotideTransformer:main", self.model_name())


class NucleotideTransformerEmbedding(DNAEmbedding):
    @method
    def setup(
        self, 
    ):
        self.model = None
        self.device = None
        self.tokenizer = None

    def __getstate__(self):
        return dict(max_length=self.max_length, layers=str(self.layers))

    def __setstate__(self, state):
        self.__init__()

        # Restore the object state from the unpickled state
        self.__dict__.update(state)
        self.model = None
        self.device = None
        self.tokenizer = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species").to(self.device)

    def embed(self, seq:str) -> torch.Tensor:
        """ Takes a DNA sequence as a string and returns an embedding tensor. """
        if not self.model:
            self.load()

        max_length = self.tokenizer.model_max_length
        tokens_ids = self.tokenizer.batch_encode_plus([seq[:max_length]], return_tensors="pt", padding="max_length", max_length = max_length)["input_ids"].to(self.device)

        # Compute the embeddings
        attention_mask = tokens_ids != self.tokenizer.pad_token_id
        torch_outs = self.model(
            tokens_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )
        embeddings = torch_outs['hidden_states'][-1].mean(dim=1)[0].cpu().detach()

        return embeddings
