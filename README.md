# Description Image BERT
BERT trained on a vocabulary and sequence format concatenating verbal description tokens to token-based representations of images.
### Data / Sequence Format
```
[DESC] <description tokens + [PAD] tokens> [IMG] <image tokens>
```
(Training data only applies [MASK] tokens to the description tokens)
