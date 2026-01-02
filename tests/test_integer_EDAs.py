from benchmarks.integer_functions_benchmark import run_integer_experiment

results = run_integer_experiment(
    eda_name='tree_eda',
    function_name='gen_k_decep_int',
    n_vars=30,
    cardinality=2,
    pop_size=300,
    max_gen=100,
    seed=45
)
