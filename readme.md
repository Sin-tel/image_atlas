## Image Atlas

Local image browser with automatic embedding similarity search (using SigLIP2).

<img src="media/screenshot.png" style="width:800px;"/>

# Installation

Using `uv`, just run:
```bash
uv init
uv run python server.py
```
This will automatically open http://localhost:8765/.

You will need to edit `config.py` to match your local setup. The SigLIP2 model will be automatically downloaded from huggingface.

Note that the first run will take a while to calculate all the image embeddings, after that it should run fast. I recommend starting with a small folder of a few hundred images if you just want to try it out.
