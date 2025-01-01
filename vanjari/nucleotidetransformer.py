import torch
from torchapp.cli import method
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from pathlib import Path
from appdirs import user_cache_dir
from torchapp.download import cached_download

from .dnaembedding import DNAEmbedding


class NucleotideTransformerEmbedding(DNAEmbedding):
    @method
    def setup(
        self, 
        model_name:str="",
    ):
        self.model_name = model_name
        self.model = None
        self.device = None
        self.tokenizer = None

        if not self.model_name:
            cache_dir = Path(user_cache_dir("torchapps"), "Vanjari")
            local_path = cache_dir/"nucleotide-transformer-v2-500m-virus"
            if not local_path.exists():
                url = f"https://figshare.unimelb.edu.au/ndownloader/files/51434933"
                tarball_path = cached_download(url, local_path=cache_dir / "nucleotide-transformer-v2-500m-virus.tar.gz")

                # extract the tarball to local_path
                import tarfile
                with tarfile.open(tarball_path) as tar:
                    tar.extractall(local_path)

            self.model_name = str(local_path)

    def __getstate__(self):
        return dict(max_length=self.max_length, model_name=str(self.model_name))

    def __setstate__(self, state):
        self.__init__()

        # Restore the object state from the unpickled state
        self.__dict__.update(state)
        self.model = None
        self.device = None
        self.tokenizer = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)

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


