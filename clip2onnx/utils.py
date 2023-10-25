from pathlib import Path
from typing import Callable


def print_available_open_clip_models():
    import open_clip

    def lookup(mol: Callable[[], list]):
        for model_name in mol():
            if tags := open_clip.list_pretrained_tags_by_model(model_name):
                for tag in tags:
                    card = {"model": model_name, "tag": tag}
                    print(f"ModelCard | {card}")
            card = {"model": model_name, "tag": ""}
            print(f"ModelCard | {card}")

    lookup(open_clip.list_openai_models)
    lookup(open_clip.list_models)


def request_resource(url: str, save_path: Path):
    # TODO: Might need to implement a simple download progress bar?
    import httpx
    from tqdm import tqdm

    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.203"
    }
    with open(save_path, "wb") as download_file:
        with httpx.Client(headers=headers, follow_redirects=True, http2=True) as client:
            with client.stream("GET", url) as response:
                total = int(response.headers["Content-Length"])
                with tqdm(
                    total=total,
                    unit_scale=True,
                    unit_divisor=1024,
                    unit="B",
                    desc=f"Installing {save_path.parent.name}/{save_path.name}",
                ) as progress:
                    num_bytes_downloaded = response.num_bytes_downloaded
                    for chunk in response.iter_bytes():
                        download_file.write(chunk)
                        progress.update(response.num_bytes_downloaded - num_bytes_downloaded)
                        num_bytes_downloaded = response.num_bytes_downloaded
