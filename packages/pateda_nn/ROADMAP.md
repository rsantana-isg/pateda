# pateda-nn ROADMAP

## Completed (v0.1.0)

### PyTorch learning components
- [x] VAE-EDA: basic, extended (E-VAE), conditioned extended (CE-VAE)
- [x] GAN-EDA (continuous)
- [x] DBD-EDA (alpha-deblending diffusion)
- [x] DAE-EDA (denoising autoencoder)
- [x] Dendiff-EDA (denoising diffusion, Gaussian noise)
- [x] Dendiff-ReLU variant
- [x] Backdrive-EDA (backpropagation-guided)
- [x] RBM-EDA (restricted Boltzmann machine)
- [x] Generic NN-EDA framework (`LearnNNEDA`)

### PyTorch discrete / binary learning components
- [x] Discrete VAE-EDA (Gumbel-Softmax, Bernoulli)
- [x] Discrete Extended VAE-EDA
- [x] Discrete GAN-EDA
- [x] Discrete DBD (CS and CD variants)
- [x] Discrete Backdrive (standard, Huber loss, ranking loss, weighted MSE, descriptors)
- [x] Discrete Dendiff — five relaxation variants:
  - Gumbel-Softmax (basic + enhanced)
  - Corruption (basic + enhanced)
  - STE (basic + enhanced)
  - Deterministic (basic + enhanced)
  - Hard Concrete (basic + enhanced)

### Legacy (TensorFlow)
- [x] `pateda_nn.legacy` — TF-based VAE, GAN, DBD, Diffusion EDAs
- [x] GNBG benchmark class + instances
- [x] Gaussian, copula, and selection utilities (TF context)

---

## Planned / In progress (v0.2.0)

### Code quality
- [ ] Unify `learn_*(...)` function signatures — currently keyword-arg sets differ between algorithms
- [ ] Add type annotations and docstrings to all public functions
- [ ] Replace `sys.path` hacks remaining in any legacy sampling utilities
- [ ] 80 %+ test coverage for learning/sampling pairs
- [ ] GPU-aware test fixtures (skip CUDA tests if no GPU available)
- [ ] Sphinx API docs with RTD deployment

### New algorithms
- [ ] Normalising Flows EDA (RealNVP / Glow) for continuous problems
- [ ] Score-based diffusion EDA (DDPM on continuous variables)
- [ ] Transformer-based EDA (attention model over variable interactions)
- [ ] Energy-based model EDA

### Discrete improvements
- [ ] Unify discrete sampling under a single dispatcher (`sample_discrete_nn`)
- [ ] Integer-valued (multi-cardinality) VAE / GAN EDAs
- [ ] Discrete Backdrive with ranking loss for integers

### Continuous improvements
- [ ] Adaptive temperature scheduling in Dendiff
- [ ] Restarts / diversity maintenance integrated with pateda.replacement
- [ ] Multi-objective variant: Pareto-front aware VAE sampling

### Legacy migration
- [ ] Port TF VAE models to PyTorch (makes `pateda_nn.legacy` obsolete)
- [ ] Port TF GAN models to PyTorch
- [ ] Port TF diffusion (EfficientDiffusion, EfficientBackdrive) to PyTorch
- [ ] Deprecate `pateda_nn.legacy` once porting is complete

---

## Difficult to incorporate (tracked here)

- **Discrete Dendiff — Hard Concrete variant**: the relaxation introduces numerical instability for tight constraints; the current implementation may diverge on some problems with > 50 variables.
- **Multi-cardinality GAN**: mode collapse is severe for variables with cardinality > 5; needs specialised training regime.
- **Backdrive with descriptor features**: `discrete_backdrive_descriptors.py` depends on custom feature extraction that is problem-specific; not yet generalised.
- **RBM for discrete EDAs**: the current RBM uses continuous relaxation via sigmoid; a proper binary RBM sampler (Gibbs) is unfinished.
- **Dendiff-ReLU (continuous)**: the ReLU variant sometimes fails to converge because the denoising network architecture is architecture-sensitive; training hyperparameters need automated tuning.

---

## v0.3.0 and beyond

- [ ] Benchmark CLI: `pateda-nn-bench --algo VAE --problem onemax --n 50 --seeds 20`
- [ ] Integration with Weights & Biases for automatic experiment tracking
- [ ] Pre-trained model checkpoints for standard problems (OneMax, Deceptive3, TSP-20)
- [ ] ONNX export for trained generative models
