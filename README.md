# Phonological Sequence Generation

Trained on children's speech in English with the 
[IPA-CHILDES dataset](https://huggingface.co/datasets/phonemetransformers/IPA-CHILDES). 
Uses phonological feature vectors for tokens. 
[Getheode](https://github.com/Cielbird/getheode) library used for parsing.

The example can be run like so:

## CUDA users

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn

# Use the --release flag to really speed up training.
export TORCH_CUDA_VERSION=cu128
cargo run --example train --release
```

## Mac users

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn

# Use the --release flag to really speed up training.
cargo run --example train --release
```
