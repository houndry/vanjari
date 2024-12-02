from pathlib import Path
from torchapp.download import cached_download

def get_vmr(local_path) -> Path:
    url = "https://ictv.global/sites/default/files/VMR/VMR_MSL39.v4_20241106.xlsx"
    return cached_download(url, local_path)


