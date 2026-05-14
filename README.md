# Phonological Sequence Generation

Trained on children's speech in English with the 
[IPA-CHILDES dataset](https://huggingface.co/datasets/phonemetransformers/IPA-CHILDES). 

Using the [Burn](https://github.com/tracel-ai/burn) framework

Uses phonological feature vectors for tokens. 
[Getheode](https://github.com/Cielbird/getheode) library used for parsing.

Training :

```bash
cargo run --example train --release
```
