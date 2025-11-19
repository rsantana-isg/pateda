"""
Quick test of MN-FDA, MN-FDAG, and MOA implementations
"""

import numpy as np
import sys

# Test imports
try:
    from pateda.core.eda import EDA, EDAComponents
    from pateda.stop_conditions import MaxGenerations
    from pateda.seeding import RandomInit
    from pateda.selection import TruncationSelection
    from pateda.replacement import GenerationalReplacement
    from pateda.learning import LearnMNFDA, LearnMNFDAG, LearnMOA
    from pateda.sampling import SampleFDA, SampleGibbs
    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def onemax(x):
    """Simple OneMax function"""
    return float(np.sum(x))


def test_mnfda():
    """Test MN-FDA with PLS sampling"""
    print("\n" + "=" * 60)
    print("Testing MN-FDA with PLS Sampling")
    print("=" * 60)

    try:
        components = EDAComponents(
            seeding=RandomInit(),
            selection=TruncationSelection(proportion=0.5),
            learning=LearnMNFDA(
                max_clique_size=3,
                threshold=0.05,
                return_factorized=True
            ),
            sampling=SampleFDA(n_samples=50),
            replacement=GenerationalReplacement(),
            stop_condition=MaxGenerations(10),
        )

        eda = EDA(
            pop_size=50,
            n_vars=20,
            fitness_func=onemax,
            cardinality=np.full(20, 2),
            components=components,
        )

        stats, _ = eda.run(verbose=False)

        print(f"✓ MN-FDA test passed!")
        print(f"  Best fitness: {stats.best_fitness_overall:.1f}/20.0")
        print(f"  Generations: {stats.generation_found + 1}/10")

        return True

    except Exception as e:
        print(f"✗ MN-FDA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mnfdag():
    """Test MN-FDAG with PLS sampling"""
    print("\n" + "=" * 60)
    print("Testing MN-FDAG with PLS Sampling")
    print("=" * 60)

    try:
        components = EDAComponents(
            seeding=RandomInit(),
            selection=TruncationSelection(proportion=0.5),
            learning=LearnMNFDAG(
                max_clique_size=3,
                alpha=0.05,
                return_factorized=True
            ),
            sampling=SampleFDA(n_samples=50),
            replacement=GenerationalReplacement(),
            stop_condition=MaxGenerations(10),
        )

        eda = EDA(
            pop_size=50,
            n_vars=20,
            fitness_func=onemax,
            cardinality=np.full(20, 2),
            components=components,
        )

        stats, _ = eda.run(verbose=False)

        print(f"✓ MN-FDAG test passed!")
        print(f"  Best fitness: {stats.best_fitness_overall:.1f}/20.0")
        print(f"  Generations: {stats.generation_found + 1}/10")

        return True

    except Exception as e:
        print(f"✗ MN-FDAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_moa():
    """Test MOA with Gibbs sampling"""
    print("\n" + "=" * 60)
    print("Testing MOA with Gibbs Sampling")
    print("=" * 60)

    try:
        components = EDAComponents(
            seeding=RandomInit(),
            selection=TruncationSelection(proportion=0.5),
            learning=LearnMOA(
                k_neighbors=5,
                threshold_factor=1.5,
            ),
            sampling=SampleGibbs(
                n_samples=50,
                IT=2,  # Fewer iterations for quick test
                temperature=1.0,
                random_order=True
            ),
            replacement=GenerationalReplacement(),
            stop_condition=MaxGenerations(10),
        )

        eda = EDA(
            pop_size=50,
            n_vars=20,
            fitness_func=onemax,
            cardinality=np.full(20, 2),
            components=components,
        )

        stats, _ = eda.run(verbose=False)

        print(f"✓ MOA test passed!")
        print(f"  Best fitness: {stats.best_fitness_overall:.1f}/20.0")
        print(f"  Generations: {stats.generation_found + 1}/10")

        return True

    except Exception as e:
        print(f"✗ MOA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Markov Network EDA Implementations")
    print("=" * 60)

    results = []
    results.append(test_mnfda())
    results.append(test_mnfdag())
    results.append(test_moa())

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"MN-FDA:  {'PASS' if results[0] else 'FAIL'}")
    print(f"MN-FDAG: {'PASS' if results[1] else 'FAIL'}")
    print(f"MOA:     {'PASS' if results[2] else 'FAIL'}")

    if all(results):
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
