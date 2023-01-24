## Description Image BERT
BERT trained on a vocabulary and sequence format concatenating verbal description tokens to token-based representations of images.
### Data
#### Sequence Format
```
[DESC] <description tokens + [PAD] tokens> [IMG] <image tokens>
```
*Training data only applies [MASK] tokens to the description tokens.*
#### Image -> token sequence
60x60 grayscale images are unrolled into sequences of 3x3 sub-matrices of pixels where brightness values are rounded to trinary enumerations (0, 1, or 2); then they are further unrolled into a sequence 9-length trinary sequences.
