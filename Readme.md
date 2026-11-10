# DeiT from Scratch: Experiment Report

Nov 10, 2026 · @Harshit Singh

## Summary

Distilling a CNN teacher into a small Vision Transformer gave only small gains: about +1.8 points on CIFAR-10, +1.5 on CIFAR-100 with a ResNet50 teacher, and none on MNIST. Every number below comes from a single run.

- **MNIST:** 90.45% for the distilled student against 90.44% for the student-only prediction. The teacher was never fine-tuned, so distillation carried no useful signal.
- **CIFAR-10:** the ResNet50 teacher reached 94.26%. The DeiT student scored 80.04%, against 78.24% for a plain ViT.
- **CIFAR-100:** the teachers reached 81.52% (ResNet50) and 88.96% (ViT-B/16). The DeiT students scored 61.17% and 59.75%, against 59.72% for a plain ViT.

## Background

The project tests one idea: a Vision Transformer trained from scratch learns poorly on small datasets, and a CNN teacher can help it.

- **CNN.** Shared filters slide over the image, so a feature is found wherever it appears. Locality and weight sharing are built-in assumptions (inductive bias) that make CNNs data-efficient, but each filter sees only a small neighbourhood.
- **ViT.** The image is cut into patches, each patch becomes a token, and self-attention lets every patch see every other patch from the first layer. It has almost no built-in image prior, so it normally needs very large datasets.
- **DeiT.** A second learned token, the distillation token, sits next to the class token. The class token is trained on the true label; the distillation token is trained to match the teacher's softened predictions. At test time the two heads are averaged.

## Method

Every experiment uses one recipe: a small ViT student with a class token and a distillation token, trained against a frozen teacher. The loss weights the true label and the teacher's softened prediction equally (α = 0.5, temperature T = 2):

```latex
\mathcal{L} = (1-\alpha)\,\mathrm{CE}(z_{cls},\,y) + \alpha\,T^{2}\,\mathrm{KL}\!\left(\sigma(z_{teacher}/T)\,\|\,\sigma(z_{dist}/T)\right)
```

The plain-ViT baseline drops the distillation token and the teacher and trains on cross-entropy alone. The three experiments differ as follows.

| Setting | MNIST | CIFAR-10 | CIFAR-100 |
| --- | --- | --- | --- |
| Images, classes | 28×28 grey (copied to 3 channels), 10 | 32×32 RGB, 10 | 32×32 RGB, 100 |
| Patch size, patches | 7, 16 | 4, 64 | 4, 64 |
| Student | 4 layers, dim 32, 4 heads | 6 layers, dim 128, 4 heads | 6 layers, dim 128, 4 heads (about 1.2M parameters) |
| Feed-forward width | 8 (a bug) | 4 × dim | 4 × dim |
| Epochs, learning rate | 5, 3e-4 | 30, 5e-4 | 60, 5e-4 with cosine decay |
| Optimiser | AdamW | AdamW | AdamW, weight decay 0.05, fp16 |
| Augmentation | none | random crop, flip | random crop, flip |
| Teacher | ResNet50 with a new, untrained head | ResNet50 fine-tuned 3 epochs (64 px input) | ResNet50 (64 px) and ViT-B/16 (224 px), each fine-tuned 8 epochs |

## Experiment 1: MNIST

The distilled student reached 90.45% and the student-only prediction reached 90.44%, so the teacher added nothing. The cause was the setup, not distillation itself:

- **Untrained teacher.** The ResNet50's last layer was replaced by a new 10-class layer that was never trained, so its soft labels were noise.
- **Tiny feed-forward layers.** The third argument of `nn.TransformerEncoderLayer` is the feed-forward width. The head dimension (8) was passed instead of 4 × the embedding size.
- **Very small model.** An embedding size of 32 and 5 epochs capped both models near 90%, which is low for MNIST.

The CIFAR experiments fix all three.

## Experiment 2: CIFAR-10

On CIFAR-10 the distilled student beat the plain ViT by 1.8 points (80.04% against 78.24%). Compared with MNIST, the changes were:

- **Data:** CIFAR-10 images with random crop and flip.
- **Student:** patch size 4 (64 patches), embedding size 128, 6 layers, 30 epochs, feed-forward width fixed to 4 × the embedding size.
- **Teacher:** ResNet50 fine-tuned for 3 epochs before distillation, with images upsampled to 64 px.
- **Baseline:** the same ViT with no distillation token, trained on hard labels only.

| Model | Test accuracy |
| --- | --- |
| ResNet50 teacher | 94.26% |
| DeiT student, average of both heads | 80.04% |
| DeiT student, class-token head | 80.29% |
| DeiT student, distillation-token head | 79.91% |
| Plain ViT (no teacher) | 78.24% |

The fairest comparison is the class-token head against the plain ViT (+2.05 points), because both are trained on the same cross-entropy objective. The teacher's signal reaches the class token through the shared transformer layers.

## Experiment 3: CIFAR-100

On CIFAR-100 the ResNet50 teacher lifted the student by 1.45 points (61.17% against 59.72%), while the stronger ViT-B/16 teacher gave 59.75%, no gain over the plain ViT. Both teachers were fine-tuned for 8 epochs; all three students are the same 1.2M-parameter ViT, trained for 60 epochs with the same seed.

&#91;embedded content: CIFAR-100 notebook run · five test accuracies, one seed each\]

The ResNet50 teacher is 7.4 points less accurate than the ViT-B/16 teacher, yet it produced the better student.

## Discussion

The gains are small because the student is small and the training is short, and the stronger teacher turned out not to be the better one.

- **Capacity and budget.** The student has about 1.2M parameters and trains for 30 to 60 epochs with only crop and flip augmentation. The DeiT paper trains for 300 epochs with heavy augmentation, so there is little room for the student to absorb extra knowledge.
- **The CNN teacher helped more than the ViT teacher.** On CIFAR-100 the ResNet50 teacher (81.52%) lifted the student by 1.45 points, while the more accurate ViT-B/16 teacher (88.96%) lifted it by 0.03. The DeiT paper also reports that a convnet teacher gives a better student than a transformer teacher. Two plausible reasons, neither tested here: the student picks up some of the CNN's locality bias through its predictions, and a 86M-parameter teacher is far larger than a 1.2M-parameter student can imitate.
- **Noise.** Each result is one seed. Differences near one point can come from run-to-run variation, so +1.45 is suggestive and +0.03 is no difference.
- **Untuned teachers.** Both were fine-tuned for a few epochs on 32 px images upsampled to 64 or 224 px, so their soft labels are imperfect.

## Limitations and future work

The results are suggestive, not conclusive, for four reasons:

- Each model was trained once; three seeds with mean and spread are needed.
- The experiments are not directly comparable: student size, epochs and optimiser changed between MNIST, CIFAR-10 and CIFAR-100, and the MNIST teacher was untrained.
- The loss weight α and temperature T were never tuned, and only soft-label distillation was tried.
- Training was short with light augmentation, and teacher fine-tuning was short on upsampled images.

Next steps:

1. Compare hard-label and soft-label distillation, with a ResNet and with a ViT as the pretrained teacher.
2. Check whether the CNN-teacher advantage holds when the two teachers have matched accuracy.
3. Train longer with stronger augmentation (RandAugment, Mixup, CutMix) and three seeds.

## Conclusion

DeiT-style distillation helped a small ViT modestly when the teacher was a well-trained CNN (+1.8 points on CIFAR-10, +1.45 on CIFAR-100). It did nothing with an untrained teacher (MNIST) or with the larger ViT-B/16 teacher (CIFAR-100, +0.03). The class-token and distillation-token heads ended almost equal (80.29% against 79.91% on CIFAR-10). Teacher type mattered more than teacher accuracy here, but with one run per model that needs confirming.
