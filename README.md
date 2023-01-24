## Description Image BERT
BERT trained on a vocabulary and sequence format concatenating verbal description tokens to token-based representations of images. Model implementation reference [here](https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial); though this BERT is heavily modified from its source reference (no next-sentence prediction in training, training negates attention to [PAD] tokens, and the nature of the data (explained in next sections) is novel).
### Sequence Format
```
[DESC] <description tokens + [PAD] tokens> [IMG] <image tokens>
```
*Training data only applies [MASK] tokens to the description tokens.*
### Image -> [IMG] Token Sequence
60x60 grayscale images are unrolled into sequences of 3x3 sub-matrices of pixels where brightness values are rounded to trinary enumerations (0, 1, or 2); then they are further unrolled into a sequence 9-length trinary sequences.
### Trinary Token
<img src="/static/trinary_demo.png" height="300">
### Attention Visuals
This project is less interested in
