## Description Image BERT
BERT trained on a vocabulary and sequence format concatenating verbal description tokens to token-based representations of images. Classic BERT model implementation reference [here](https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial); though this BERT is heavily modified from its source material (no next-sentence prediction, training negates logits at [PAD] tokens, and the nature of the data is novel).
### Sequence Format
```
[DESC] <description tokens + [PAD] tokens> [IMG] <image trinary tokens>
```
*Training data only applies [MASK] tokens to the description tokens.*
### Image -> Trinary Tokens
60x60 grayscale images are unrolled into sequences of 3x3 sub-matrices of pixels where brightness values are bucketed then enumerated to trinary values (0, 1, or 2); then they are further unrolled into a "sentence" of 9-length trinary "words".
### Trinary Token
<img src="/static/trinary_demo.png" height="300">

### Attention Visuals
One reason for this BERT is to see the contextual-embedding / attention relationships between [DESC] tokens and [IMG] tokens. Here are visuals showing how much attention select [DESC] tokens gave the [IMG] tokens comprising entire images (dataset limited to images of playing cards and their token vocabulary). **Zoom in to the plotted images** to see the attention more clearly; brighter red highlights represent greater attention values.

<img src="/attention_plots/attn_plot.png" width="1500">

### Plans / Thoughts
May run this on a modified version of the CIFAR-100 dataset (but with better image descriptions). The inspiration for this BERT is that all information can be thought of as an interconnected whole / graph - and that we pay more attention to some information over the rest. (Hence why verbal tokens and image tokens belong to the same sequence on which self-attention is performed.)
