# LLM-guided Vision Encoder (LIVE)

An implementation scaffold for **Language-Instructed Vision Embeddings (LIVE)**, based on the ICLR 2026 paper by Mao, Lin, and Chu.

LIVE makes image representations conditional on a natural-language instruction. The reference design reuses a pretrained vision-language model's frozen text tower to encode the instruction, projects that representation into the vision-transformer width, and inserts it as a conditioning token alongside the image tokens. Training uses synthetic image/instruction/answer examples and a contrastive objective against answer text embeddings. At inference, the conditioned image embedding can be used for retrieval or classification without running a large language model.

Paper: [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/file/96a54c09569ebbdd9ecb22f5012e6b66-Paper-Conference.pdf) · [Project page](https://live-embedding.github.io/)

## Status and scope

This repository implements the core conditioning and contrastive-training path in a small, auditable PyTorch package. It is a research reproduction scaffold, not a claim of bit-for-bit reproduction: the project page does not publish every training detail, dataset artifact, or checkpoint. Backbones, tokenization, data generation, and training hyperparameters are therefore supplied by the experimenter.

## Install

```bash
pip install -e .
```

PyTorch is the only required dependency. Install a compatible image library such as Pillow and your chosen CLIP/OpenCLIP or Hugging Face backbone separately.

## Model integration

`LiveVisionEncoder` accepts a timm-style ViT and a frozen text encoder. The text encoder must return one pooled vector per prompt. The vision backbone should expose `patch_embed`, `cls_token`, `pos_embed`, `pos_drop`, `norm_pre`, `blocks`, and `norm` (the standard timm ViT interface). The wrapper produces an L2-normalized image embedding conditioned on the instruction. If the text tower returns projected joint-space embeddings whose width differs from the ViT hidden width, set `embedding_dim` to that width so the learned visual projection aligns the spaces.

```python
import torch
from live_encoder import LiveVisionEncoder

# vision_vit: a pretrained timm VisionTransformer
# text_encoder: a pretrained text tower returning pooled prompt embeddings
# tokenize: callable mapping list[str] to tokens accepted by text_encoder
model = LiveVisionEncoder(vision_vit, text_encoder, text_dim=512, embedding_dim=512)
embeddings = model.encode_image(images, ["focus on the printed label"] * len(images), tokenize)
```

Only the vision encoder and the query projection are trainable by default; the text encoder is frozen. Set `train_text_encoder=True` only for a deliberate ablation.

## Training data

Prepare JSONL rows with paths relative to the manifest:

```json
{"image":"images/0001.jpg","instruction":"Ignore the background and identify the fruit","answer":"A red apple","negatives":["A printed word on the sign"]}
```

Generate varied instruction/answer pairs from image content with an LLM, then review samples and keep the generation prompt and model/version with the experiment. Avoid leaking evaluation images or labels into training data. `LiveTripletDataset` loads these records; `live_contrastive_loss` contrasts each conditioned image against its matched answer, other batch answers, and optional explicit hard negatives.

```python
from torch.utils.data import DataLoader
from live_encoder import LiveTripletDataset, collate_triplets, live_contrastive_loss

dataset = LiveTripletDataset("train.jsonl", image_transform=your_transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_triplets)
for batch in loader:
    # tokenize text and move tensors to your accelerator
    image_z = model.encode_image(batch["images"], batch["instructions"], tokenize)
    answer_z = model.encode_text(batch["answers"], tokenize)
    loss = live_contrastive_loss(image_z, answer_z, temperature=0.07)
    # optimizer step; include hard negatives with the optional argument
```

## Reproduction checklist

1. Choose and record the pretrained vision/text checkpoint and tokenizer.
2. Generate and validate image/instruction/answer triplets from the training split only.
3. Train the ViT and query projection with symmetric image-answer contrastive loss.
4. Evaluate instruction-conditioned retrieval and downstream benchmarks with frozen encoders.
5. Report seeds, data generation details, compute, checkpoints, and both standard and instruction-conditioned baselines.

## Citation

```bibtex
@inproceedings{mao2026live,
  title={Language-Instructed Vision Embeddings for Controllable and Generalizable Perception},
  author={Mao, Chengzhi and Lin, Xudong and Chu, Wen-Sheng},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```
