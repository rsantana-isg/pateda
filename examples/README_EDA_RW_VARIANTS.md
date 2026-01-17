# EDA Real-World Problem Variants

This directory contains 5 programs that combine EDA variants with real-world combinatorial problem support (SAT, Ising, UBQP).

## Completed Programs ✅

### 1. discrete_GAN_EDA_RW.py
**Purpose**: Combines GAN-EDA variants with real-world problems

**Supported Variants**: 
- GAN, WGAN-GP, Cond-Fit-GAN, Aux-GAN
- Repulsion-GAN, Weighted-D-GAN, Statistic-Match, Hybrid-GAN-VAE

**Usage**:
```bash
python discrete_GAN_EDA_RW.py <seed> <problem_type> <instance_name> <pop_size> <n_gen> <trunc> \
    <variant> <activation_g> <activation_d> <activation_e> <dropout> <temperature> <use_surrogate> <alpha>
```

**Example**:
```bash
python discrete_GAN_EDA_RW.py 0 SAT uf20-01 80 20 0.5 WGAN-GP relu leaky_relu relu 0.5 1.0 0 0.0
```

### 2. discrete_Backdrive_EDA_RW.py
**Purpose**: Combines Backdrive-EDA variants with real-world problems

**Supported Variants**:
- backdrive, backdrive_adaptive, backdrive_descriptors

**Usage**:
```bash
python discrete_Backdrive_EDA_RW.py <seed> <problem_type> <instance_name> <pop_size> <n_gen> <trunc> \
    <variant> <init> <loss> <activation> <weight_transfer> <early_stopping> <surrogate_filtering> <alpha>
```

**Example**:
```bash
python discrete_Backdrive_EDA_RW.py 0 Ising SG_16_1 100 30 0.5 backdrive random mse relu 0 0 0 0.95
```

## In Progress Programs 🚧

### 3. discrete_VAE_EDA_RW.py
Template created. Needs final integration of VAEEDA class and parameters.

### 4. discrete_DbD_EDA_RW.py  
Template created. Needs final integration of DbDEDA class and parameters.

### 5. discrete_Dendiff_EDA_RW.py
Template created. Needs final integration of DendiffEDA class and parameters.

## Common Features

All programs support:
- **Real-world problems**: SAT, Ising, UBQP
- **Frequency balance mutation**: Controlled via alpha parameter  
- **Reproducibility**: Seeding with set_seed()
- **Consistent interface**: Same problem loading and evaluation logic

## Problem Types

### SAT (Boolean Satisfiability)
- Instances: `uf20-01.cnf`, `uf50-01.cnf`, etc.
- Location: `functions/SAT_instances/`
- Goal: Maximize number of satisfied clauses

### Ising (Spin Glass Model)
- Instances: `SG_16_1.txt`, `SG_100_1.txt`, etc.
- Location: `functions/Ising_Instances/`
- Goal: Minimize energy (maximize -energy)

### UBQP (Unconstrained Binary Quadratic Programming)
- Instances: `bqp50.txt`, `bqp100.txt`, etc.
- Location: `functions/UBQP_Instances/`
- Goal: Maximize objective value

## Implementation Pattern

Each *_RW.py file follows this structure:

1. **Header**: Docstring describing the combined functionality
2. **Imports**: EDA-specific + real-world problem modules
3. **Constants**: SUCCESS_THRESHOLD, UNKNOWN_OPTIMAL, etc.
4. **Seeding**: set_seed() for reproducibility
5. **Problem Loading**: load_sat_instance(), load_ising_instance(), load_ubqp_instance()
6. **Problem Evaluation**: evaluate_sat(), evaluate_ising(), evaluate_ubqp()
7. **Problem Parsing**: parse_rw_problem() dispatcher
8. **EDA Class**: Variant-specific EDA with alpha parameter for mutation
9. **Main**: Argument parsing and EDA execution

## Frequency Balance Mutation

All variants include frequency balance mutation controlled by the `alpha` parameter:
- `alpha = 0.0`: No mutation (default)
- `alpha > 0.0`: Apply mutation to limit frequency imbalance (e.g., 0.95)

The mutation is applied AFTER sampling and BEFORE final evaluation, with elitism to preserve the best solution.

