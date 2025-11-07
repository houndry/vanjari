import random
from rich.progress import track
from torchapp.cli import tool

from barbet.embedding import Embedding

class DNAEmbedding(Embedding):
    @tool("setup")
    def test_lengths(
        self,
        end:int=5_000,
        start:int=1000,
        retries:int=5,
        **kwargs,
    ):
        def random_nucleotides(k):
            nucleotides = "ACGTN" 
            return ''.join(random.choice(nucleotides) for _ in range(k))
        
        self.max_length = None
        self.setup(**kwargs)
        for ii in track(range(start,end)):
            for _ in range(retries):
                seq = random_nucleotides(ii)
                try:
                    self(seq)
                except Exception as err:
                    print(f"{ii}: {err}")
                    return
