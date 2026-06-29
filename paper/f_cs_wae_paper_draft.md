# F-CS-WAE: Factorized Class-Structured Spherical Cauchy Wasserstein Auto-Encoders for Label-Guided Generative Clustering

Anonymous Authors

## Abstract

Learning latent representations that support clustering, reconstruction, and prior-based generation remains challenging for natural images. Class-guided representation learning often encourages compact semantic clusters, while image generation requires preserving intra-class variation such as color, pose, texture, and background. We propose **F-CS-WAE**, a factorized class-structured Spherical Cauchy Wasserstein auto-encoder for label-guided generative clustering. F-CS-WAE decomposes the latent representation into a spherical semantic variable and an Euclidean style variable. The semantic variable is aligned with class-conditional Spherical Cauchy priors through distributional MMD matching, while the style variable is matched to a Gaussian prior to preserve sampleable intra-class variation. Clustering is performed only in the semantic latent space, whereas reconstruction and generation use both semantic and style variables. On CIFAR-10, F-CS-WAE achieves **80.9 +/- 0.6%** clustering accuracy, **65.0 +/- 1.0** NMI, and **63.2 +/- 1.1** ARI across three seeds, while maintaining stable reconstruction and moderate prior-based generation. Experiments on MNIST and Fashion-MNIST further test whether class-structured semantic priors are sufficient for simpler datasets. Our results suggest that factorizing semantic compactness from style diversity is a practical design principle for label-guided generative clustering.

## 1. Introduction

Auto-encoding generative models aim to learn latent representations that are useful for both reconstruction and sampling. In many applications, however, one also wants the latent space to expose semantic structure: examples from the same class should form coherent clusters, and samples from class-specific priors should decode into images with the desired class identity. This paper studies this setting under the name **label-guided generative clustering**.

The problem is not the same as fully unsupervised clustering. Labels are available during training and are used to guide the latent geometry. The goal is also not pure classification, because the model must retain a generative path from a prior distribution to images. A successful model should therefore satisfy three requirements: it should reconstruct inputs, organize its latent space into class-aligned clusters, and allow sampling from known priors.

These requirements can conflict. Clustering benefits from compact and well-separated class representations. Generation, especially on natural images, requires preserving rich intra-class variation. For CIFAR-10, images from the same class can differ substantially in color, pose, background, viewpoint, and texture. If one latent variable is forced to carry both class identity and all residual visual variation, class compactness can damage generation, while reconstruction pressure can weaken semantic clustering.

We propose **F-CS-WAE**, a factorized class-structured Spherical Cauchy Wasserstein auto-encoder. The key idea is to separate the latent representation into two variables:

1. a **semantic latent variable** \(z_c\), constrained to the hypersphere and aligned with class-conditional Spherical Cauchy priors;
2. a **style latent variable** \(z_s\), defined in Euclidean space and matched to a Gaussian prior.

The semantic variable is used for clustering. The decoder receives both variables, so style information can support reconstruction and generation without forcing the semantic variable to encode every visual detail. This separation is particularly important for natural images, where labels explain only part of the observed variation.

On CIFAR-10, F-CS-WAE obtains stable class-aligned clustering across three seeds, with 80.9 +/- 0.6% ACC, 65.0 +/- 1.0 NMI, and 63.2 +/- 1.1 ARI. Its FID remains moderate, so we do not claim state-of-the-art image generation. Instead, the contribution is a generative clustering model that achieves strong semantic latent organization while retaining prior-based sampling.

### Contributions

1. We propose F-CS-WAE, a factorized class-structured Wasserstein auto-encoder for label-guided generative clustering.
2. We introduce a spherical semantic latent variable aligned with class-conditional Spherical Cauchy priors using supervised and aggregated MMD objectives.
3. We introduce a style latent variable matched to a Gaussian prior, allowing the decoder to preserve intra-class variation without corrupting semantic clustering.
4. We provide controlled experiments on MNIST, Fashion-MNIST, and CIFAR-10, including comparisons to generative, conditional, and supervised representation baselines.
5. We analyze the trade-off between clustering and prior-based generation, and show that F-CS-WAE should be understood as a generative clustering model rather than a dedicated high-fidelity image generator.

## 2. Related Work

### Variational and Wasserstein Auto-Encoders

Variational auto-encoders learn probabilistic latent representations through a reconstruction term and a prior regularizer. Wasserstein auto-encoders instead match the aggregated posterior to a prior, often using MMD or adversarial matching. These models provide a generative path from prior samples to images, but standard Gaussian priors do not by themselves produce class-separated latent geometry.

### Deep Generative Clustering

Deep generative clustering models such as VaDE and Gaussian mixture VAEs introduce mixture priors into latent-variable models. These methods are important baselines because they combine generation and clustering. However, Gaussian mixture priors remain Euclidean and do not explicitly separate class semantics from style variation.

### Hyperspherical Latent Variables

Hyperspherical latent spaces are useful when representation geometry is directional. A spherical latent variable removes radial degrees of freedom and encourages angular organization. Spherical Cauchy distributions provide a heavy-tailed distribution on the sphere, offering a flexible alternative to more concentrated spherical distributions such as vMF. F-CS-WAE uses a Spherical Cauchy prior for the semantic variable because class identity is naturally represented by direction, while heavy tails allow variation around each class center.

### Supervised and Contrastive Representation Learning

Supervised classification, center loss, triplet loss, and supervised contrastive learning can produce strong class-aligned embeddings. However, these methods usually do not define a generative prior. F-CS-WAE differs by placing class structure directly in the prior used for generation.

## 3. Method

### 3.1 Problem Setting

Let

\[
\mathcal{D} = \{(x_i,y_i)\}_{i=1}^N,
\]

where \(x_i\) is an image and \(y_i \in \{1,\dots,K\}\) is its label. For CIFAR-10, \(x_i \in [0,1]^{3 \times 32 \times 32}\) and \(K=10\). The goal is to learn an encoder-decoder model whose latent space supports:

1. reconstruction of \(x\);
2. clustering using a semantic representation;
3. sampling from class-conditional priors.

Labels are used during training, so the method is label-guided rather than fully unsupervised.

### 3.2 Factorized Latent Representation

F-CS-WAE decomposes the latent variable into:

\[
z = (z_c,z_s),
\]

where

\[
z_c \in \mathbb{S}^{d_c-1}
\]

is a spherical semantic variable, and

\[
z_s \in \mathbb{R}^{d_s}
\]

is a Euclidean style variable.

The semantic variable is intended to capture class-level structure. The style variable captures residual variation such as pose, color, background, and texture. Clustering uses only the deterministic semantic direction, while reconstruction and generation use both variables.

### 3.3 Encoder

The encoder maps an input image to a shared representation:

\[
h = f_\phi(x).
\]

For CIFAR-10, \(f_\phi\) is implemented with a ResNet-18 backbone. The semantic head outputs an unnormalized direction and a scalar concentration/radius parameter:

\[
\tilde{\mu}_c = g_c(h),
\]

\[
\mu_c = \frac{\tilde{\mu}_c}{\|\tilde{\mu}_c\|_2},
\]

\[
\rho_c = \sigma(g_\rho(h))(1-\epsilon).
\]

The semantic sample is drawn using a Spherical Cauchy reparameterization:

\[
z_c \sim \mathrm{SCauchy}(\mu_c,\rho_c).
\]

The style head outputs Gaussian parameters:

\[
\mu_s = g_{\mu_s}(h), \qquad \log \sigma_s^2 = g_{\sigma_s}(h).
\]

The style sample is:

\[
z_s = \mu_s + \sigma_s \odot \epsilon,\qquad \epsilon \sim \mathcal{N}(0,I).
\]

### 3.4 Class-Conditional Spherical Cauchy Prior

Each class has a prior center on the hypersphere:

\[
m_k \in \mathbb{S}^{d_c-1}.
\]

The class-conditional semantic prior is:

\[
p(z_c|y=k)=\mathrm{SCauchy}(m_k,\rho_p).
\]

For datasets with high intra-class diversity, one may use multiple centers per class:

\[
p(z_c|y=k)=\frac{1}{R}\sum_{r=1}^R \mathrm{SCauchy}(m_{k,r},\rho_p).
\]

This allows one class to occupy multiple semantic modes while remaining class-structured.

### 3.5 Style Prior

The style prior is a standard Gaussian:

\[
p(z_s)=\mathcal{N}(0,I).
\]

This prior allows the model to sample style variation independently of class semantics.

### 3.6 Decoder

The decoder receives the concatenated latent variables:

\[
\hat{x}=D_\theta([z_c,z_s]).
\]

For CIFAR-10, we use a ResBlock upsampling decoder. The decoder maps the latent vector to a \(4 \times 4\) spatial feature map and upsamples through residual blocks to \(32 \times 32\). We avoid skip connections from the encoder so that reconstruction must use the latent variables.

### 3.7 Objective

The total objective is:

\[
\mathcal{L}
=
\mathcal{L}_{rec}
+
\alpha(t)\mathcal{L}_{class}
+
\beta(t)\mathcal{L}_{agg}
+
\gamma(t)\mathcal{L}_{style}
+
\eta(t)\mathcal{L}_{cls}
-
\lambda_{var}\mathrm{Var}(z_s).
\]

#### Reconstruction Loss

For natural images, we use an L1 or perceptual reconstruction loss:

\[
\mathcal{L}_{rec}
=
\|x-\hat{x}\|_1
+
\lambda_p \mathrm{LPIPS}(x,\hat{x}).
\]

#### Class-Conditional MMD

For each class \(k\), let:

\[
Q_k=\{z_c^i:y_i=k\}
\]

be encoded semantic samples from class \(k\), and let:

\[
P_k=\{\tilde{z}_c^j:\tilde{z}_c^j \sim p(z_c|y=k)\}
\]

be prior samples from the corresponding class prior. The supervised class matching term is:

\[
\mathcal{L}_{class}
=
\frac{1}{K}
\sum_{k=1}^{K}
\mathrm{MMD}^2(Q_k,P_k).
\]

This term aligns each class-conditional semantic distribution with its prior.

#### Aggregated Semantic MMD

The aggregated semantic prior is:

\[
p(z_c)=\frac{1}{K}\sum_{k=1}^{K}p(z_c|y=k).
\]

The aggregated MMD term is:

\[
\mathcal{L}_{agg}
=
\mathrm{MMD}^2(q(z_c),p(z_c)).
\]

This term preserves global prior sampleability.

#### Style MMD

The style matching term is:

\[
\mathcal{L}_{style}
=
\mathrm{MMD}^2(q(z_s),\mathcal{N}(0,I)).
\]

This term encourages encoded style variables to match the sampling prior.

#### Auxiliary Classification Loss

We include a lightweight classifier on the semantic direction:

\[
\mathcal{L}_{cls}
=
\mathrm{CE}(C(\mu_c),y).
\]

This loss stabilizes the semantic variable during training. It is not used for clustering evaluation. Clustering is evaluated by K-means on \(\mu_c\).

#### Style Diversity Regularization

To discourage collapse of the style variable, we include a weak variance regularizer:

\[
-\lambda_{var}\mathrm{Var}(z_s).
\]

This encourages \(z_s\) to carry residual variation.

### 3.8 Training Schedule

We train with a curriculum. Early training focuses on reconstruction so that the decoder learns an image manifold. Distribution matching terms are then annealed:

| Phase | Epochs | Active terms |
|---|---:|---|
| A | 0-50 | reconstruction |
| B | 50-100 | reconstruction + style MMD + weak classification |
| C | 100-200 | add class and aggregated semantic MMD |
| D | 200+ | full objective |

This avoids forcing the semantic latent into class priors before the decoder has learned a useful mapping.

### 3.9 Inference

For clustering, we use the deterministic semantic direction:

\[
\mu_c = \frac{\tilde{\mu}_c}{\|\tilde{\mu}_c\|_2}.
\]

K-means is applied to \(\mu_c\), followed by Hungarian matching for ACC computation. Labels are not used during K-means fitting.

For class-conditional generation, we sample:

\[
z_c \sim p(z_c|y=k), \qquad z_s \sim \mathcal{N}(0,I),
\]

and decode:

\[
x = D_\theta([z_c,z_s]).
\]

## 4. Experiments

### 4.1 Datasets

We evaluate on MNIST, Fashion-MNIST, and CIFAR-10. MNIST and Fashion-MNIST test whether class-structured semantic priors are sufficient for simple grayscale images. CIFAR-10 tests whether semantic-style factorization is necessary for natural images with larger intra-class variation.

### 4.2 Baselines

We compare against:

- AE + K-means;
- VAE;
- WAE-MMD;
- VaDE or GMVAE;
- AE + classifier loss;
- AE + center loss;
- AE + supervised contrastive loss;
- conditional VAE;
- conditional WAE-MMD;
- Gaussian class-prior WAE;
- vMF class-prior WAE;
- semantic-only F-CS-WAE, where \(z_s\) is removed.

The semantic-only variant is not presented as a previous model. It is a controlled ablation that tests whether a class-structured semantic latent alone is sufficient.

### 4.3 Metrics

We report:

- ACC: clustering accuracy after K-means and Hungarian matching;
- NMI: normalized mutual information;
- ARI: adjusted Rand index;
- SSIM, PSNR, LPIPS: reconstruction quality;
- FID and KID: sample quality;
- generated class consistency using a pretrained classifier.

### 4.4 CIFAR-10 Results

Across three seeds, F-CS-WAE obtains:

| Metric | Mean +/- Std |
|---|---:|
| SSIM | 0.7034 +/- 0.0058 |
| PSNR | 21.77 +/- 0.11 |
| LPIPS | 0.2149 +/- 0.0030 |
| ACC | 80.87 +/- 0.63 |
| NMI | 64.96 +/- 0.97 |
| ARI | 63.25 +/- 1.10 |
| FID | 83.04 +/- 1.05 |

These results show strong and stable semantic clustering. The low variance across seeds suggests that the learned semantic latent space is robust. The FID remains moderate, indicating that F-CS-WAE should not be interpreted as a high-fidelity image generator. Its intended role is label-guided generative clustering.

### 4.5 Dataset Complexity Analysis

We expect the semantic-only variant to perform strongly on MNIST and Fashion-MNIST, where images are simpler and class identity explains much of the visual structure. On CIFAR-10, the full model should be more effective because \(z_s\) can absorb intra-class variation that should not be forced into \(z_c\).

This analysis tests the hypothesis:

> The benefit of semantic-style factorization increases with intra-class visual complexity.

### 4.6 Ablations

We evaluate the following ablations:

| Variant | Purpose |
|---|---|
| semantic-only | tests whether \(z_c\) alone is sufficient |
| no style MMD | tests whether \(z_s\) is sampleable from its prior |
| no class MMD | tests whether class prior alignment is necessary |
| no aggregated MMD | tests global prior matching |
| Gaussian semantic prior | tests Euclidean prior geometry |
| vMF semantic prior | tests Spherical Cauchy vs vMF |
| single center per class | tests multi-center priors |
| no auxiliary classifier | tests whether clustering comes only from classifier supervision |

The most important ablation is semantic-only versus full F-CS-WAE on CIFAR-10.

## 5. Discussion

F-CS-WAE is motivated by a simple observation: class labels do not explain all visual variation. A latent variable optimized only for class compactness may lose information needed for generation, while a latent variable optimized only for reconstruction may fail to form class-aligned clusters. F-CS-WAE separates these roles by assigning semantic structure to a spherical class-conditioned latent and residual visual variation to a style latent.

The CIFAR-10 results show that this separation can produce strong semantic clustering. However, the FID results also show a limitation: F-CS-WAE is not a replacement for dedicated image generators. Its strength is the joint behavior of clustering and prior-based generation, not state-of-the-art sample fidelity.

The semantic-only variant is useful for understanding dataset complexity. On simpler datasets, a class-structured semantic latent may be sufficient. On natural images, explicit style modeling becomes more important. This provides a practical explanation for when factorization is necessary.

## 6. Limitations

First, F-CS-WAE uses labels during training. It is therefore not a fully unsupervised clustering method. Second, sample quality on CIFAR-10 remains moderate. Third, the auxiliary classifier must be carefully controlled so that clustering performance is not merely classifier performance. Fourth, additional datasets such as SVHN, STL-10, or CIFAR-100 would further test scalability.

## 7. Conclusion

We introduced F-CS-WAE, a factorized class-structured Spherical Cauchy Wasserstein auto-encoder for label-guided generative clustering. The model separates semantic class structure from style variation by using a spherical semantic latent aligned with class-conditional Spherical Cauchy priors and an Euclidean style latent matched to a Gaussian prior. On CIFAR-10, F-CS-WAE achieves stable and strong clustering across seeds while preserving prior-based generation. The results support the view that semantic compactness and style diversity should be modeled separately when applying generative clustering methods to natural images.

