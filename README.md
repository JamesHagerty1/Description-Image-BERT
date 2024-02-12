## Description Image BERT
BERT trained on data where the sequence format is a concatenation of verbal description tokens to token-based representations of images. Classic BERT model implementation reference [here](https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial); though my BERT is heavily modified from its source material (no next-sentence prediction, training negates logits at [PAD] tokens, and the dataset is unique).
### Sequence Format
```
[DESC] <description tokens + [PAD] tokens> [IMG] <image trinary tokens>
```
*Training data only applies [MASK] tokens to the description tokens.*
### Image -> Trinary Tokens
60x60 grayscale images are unrolled into sequences of 3x3 sub-matrices of pixels where brightness values are bucketed then enumerated to trinary values (0, 1, or 2); then they are further unrolled into a sequence of 9-length trinary "words".
### Trinary Token
<img src="/static/trinary_demo.png" height="300">

### Attention Visuals
This BERT was used to see the contextual-embedding / attention relationships between [DESC] tokens and [IMG] tokens. Here are visuals showing how much attention select [DESC] tokens gave the [IMG] tokens comprising entire images (dataset limited to images of playing cards and their token vocabulary). **Zoom in to the plotted images** to see the attention more clearly. The header above each card image is a [DESC] token, and there is an attention value between that one token and every [IMG] token comprising the card image. Red highlights cover those [IMG] tokens with greater attention values relative to the header [DESC] token (the darker the red, the greater the attention value).

<img src="/attention_plots/attn_plot.png" width="1500">
