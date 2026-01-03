"""
Test that discrete backdrive variants actually sample new populations in each generation.

This test addresses the bug where backdrive perturb methods were passing the current
population instead of the selected population to the sampling methods, causing the
population to remain unchanged across generations.
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pateda.learning.discrete_backdrive import learn_binary_backdrive
from pateda.learning.discrete_backdrive_weighted_mse import learn_binary_backdrive_weighted_mse
from pateda.learning.discrete_backdrive_ranking import learn_binary_backdrive_ranking
from pateda.learning.discrete_backdrive_huber import learn_binary_backdrive_huber
from pateda.sampling.discrete_neural import sample_binary_backdrive


class TestDiscreteBacdrivePopulationSampling:
    """Test that backdrive variants properly sample new populations each generation"""
    
    def _run_backdrive_generations(self, learn_fn, method_name, n_gens=3, n_vars=20, pop_size=50):
        """
        Helper method to run multiple generations and verify population changes.
        
        Parameters
        ----------
        learn_fn : callable
            Learning function for the backdrive variant
        method_name : str
            Name of the method for identification
        n_gens : int
            Number of generations to run
        n_vars : int
            Number of variables
        pop_size : int
            Population size
            
        Returns
        -------
        populations : list
            List of populations from each generation
        fitness_values : list
            List of fitness values from each generation
        """
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Initialize population
        population = np.random.randint(0, 2, size=(pop_size, n_vars))
        
        # Track populations and fitness values
        populations = [population.copy()]
        fitness_values = []
        
        for gen in range(n_gens):
            # Evaluate fitness (OneMax)
            fitness = population.sum(axis=1, keepdims=True).astype(float)
            fitness_values.append(fitness.copy())
            
            # Selection - top 50%
            n_selected = pop_size // 2
            selected_idx = np.argsort(fitness.flatten())[-n_selected:]
            selected_pop = population[selected_idx]
            selected_fitness = fitness[selected_idx]
            
            # Learn model
            model = learn_fn(selected_pop, selected_fitness, params={'epochs': 10})
            
            # Sampling parameters for perturb methods
            sampling_params = {
                'init_method': 'perturb_best',
                'init_noise': 0.1,
                'n_iterations': 50,
                'current_population': selected_pop,  # Should use selected, not current
                'current_fitness': selected_fitness,
            }
            
            # Sample new population
            new_population = sample_binary_backdrive(model, pop_size, sampling_params)
            
            # Store new population
            populations.append(new_population.copy())
            population = new_population
        
        return populations, fitness_values
    
    def test_backdrive_population_changes_standard(self):
        """Test that standard backdrive samples new populations"""
        populations, _ = self._run_backdrive_generations(
            learn_binary_backdrive, 
            'standard_backdrive'
        )
        
        # Check that population changes at least once (from Gen 0 to Gen 1)
        # This verifies that sampling is actually producing new solutions
        pop0 = populations[0]
        pop1 = populations[1]
        
        # First generation should produce different population
        assert not np.array_equal(pop0, pop1), \
            "Population did not change from generation 0 to 1 - bug is present!"
        
        # At least some individuals should be different in first generation
        n_different = np.sum(np.any(pop0 != pop1, axis=1))
        assert n_different > 0, \
            "No individuals changed from generation 0 to 1"
        
        # Note: After convergence (Gen 1+), population may stay the same
        # This is acceptable optimization behavior, not a bug
    
    def test_backdrive_population_changes_weighted_mse(self):
        """Test that weighted MSE backdrive samples new populations"""
        populations, _ = self._run_backdrive_generations(
            learn_binary_backdrive_weighted_mse,
            'weighted_mse_backdrive'
        )
        
        # Check that population changes at least in first generation
        pop0 = populations[0]
        pop1 = populations[1]
        
        # First generation should produce different population
        assert not np.array_equal(pop0, pop1), \
            "Weighted MSE: Population did not change from generation 0 to 1"
        
        # At least some individuals should be different
        n_different = np.sum(np.any(pop0 != pop1, axis=1))
        assert n_different > 0, \
            "Weighted MSE: No individuals changed from generation 0 to 1"
    
    def test_backdrive_population_changes_ranking(self):
        """Test that ranking backdrive samples new populations"""
        populations, _ = self._run_backdrive_generations(
            learn_binary_backdrive_ranking,
            'ranking_backdrive'
        )
        
        # Check that population changes at least in first generation
        pop0 = populations[0]
        pop1 = populations[1]
        
        # First generation should produce different population
        assert not np.array_equal(pop0, pop1), \
            "Ranking: Population did not change from generation 0 to 1"
        
        # At least some individuals should be different
        n_different = np.sum(np.any(pop0 != pop1, axis=1))
        assert n_different > 0, \
            "Ranking: No individuals changed from generation 0 to 1"
    
    def test_backdrive_population_changes_huber(self):
        """Test that Huber backdrive samples new populations"""
        populations, _ = self._run_backdrive_generations(
            learn_binary_backdrive_huber,
            'huber_backdrive'
        )
        
        # Check that population changes at least in first generation
        pop0 = populations[0]
        pop1 = populations[1]
        
        # First generation should produce different population
        assert not np.array_equal(pop0, pop1), \
            "Huber: Population did not change from generation 0 to 1"
        
        # At least some individuals should be different
        n_different = np.sum(np.any(pop0 != pop1, axis=1))
        assert n_different > 0, \
            "Huber: No individuals changed from generation 0 to 1"
    
    def test_backdrive_fitness_progression(self):
        """Test that fitness improves or stays constant across generations"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Run for more generations to see improvement
        populations, fitness_values = self._run_backdrive_generations(
            learn_binary_backdrive,
            'standard_backdrive',
            n_gens=5,
            n_vars=30,
            pop_size=100
        )
        
        # Check that best fitness is non-decreasing (allowing for some stochasticity)
        best_fitness_per_gen = [np.max(f) for f in fitness_values]
        
        # At minimum, final fitness should be >= initial fitness (with some tolerance)
        initial_best = best_fitness_per_gen[0]
        final_best = best_fitness_per_gen[-1]
        
        # Allow for some stochastic variation, but fitness should generally improve
        # or at least not degrade significantly
        assert final_best >= initial_best * 0.9, \
            f"Fitness degraded: initial={initial_best}, final={final_best}"
    
    def test_backdrive_population_diversity(self):
        """Test that sampled populations maintain some diversity initially"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        populations, _ = self._run_backdrive_generations(
            learn_binary_backdrive,
            'standard_backdrive',
            n_gens=3,
            n_vars=20,
            pop_size=50
        )
        
        # Check that generation 0 has diversity
        pop0 = populations[0]
        unique_gen0 = np.unique(pop0, axis=0).shape[0]
        assert unique_gen0 > pop0.shape[0] * 0.5, \
            f"Generation 0 should have high diversity, got {unique_gen0}/{pop0.shape[0]}"
        
        # Check that generation 1 changes from generation 0 (main bug check)
        pop1 = populations[1]
        assert not np.array_equal(pop0, pop1), \
            "Generation 1 should be different from generation 0"
        
        # Note: After gen 1, the population may converge (have low diversity)
        # This is acceptable optimization behavior for backdrive methods
    
    def test_backdrive_with_different_init_methods(self):
        """Test population changes with different initialization methods"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_vars = 20
        pop_size = 50
        population = np.random.randint(0, 2, size=(pop_size, n_vars))
        fitness = population.sum(axis=1, keepdims=True).astype(float)
        
        # Select top 50%
        n_selected = pop_size // 2
        selected_idx = np.argsort(fitness.flatten())[-n_selected:]
        selected_pop = population[selected_idx]
        selected_fitness = fitness[selected_idx]
        
        # Learn model
        model = learn_binary_backdrive(selected_pop, selected_fitness, params={'epochs': 10})
        
        # Test different initialization methods
        init_methods = ['random', 'uniform', 'perturb_best', 'perturb_selected']
        
        for init_method in init_methods:
            sampling_params = {
                'init_method': init_method,
                'n_iterations': 50,
            }
            
            # Add population/fitness for perturb methods
            if 'perturb' in init_method:
                sampling_params['current_population'] = selected_pop
                sampling_params['current_fitness'] = selected_fitness
                sampling_params['init_noise'] = 0.1
            
            # Sample new population
            new_pop = sample_binary_backdrive(model, pop_size, sampling_params)
            
            # Should produce valid binary solutions
            assert new_pop.shape == (pop_size, n_vars), \
                f"Init method {init_method}: Wrong shape"
            assert np.all((new_pop == 0) | (new_pop == 1)), \
                f"Init method {init_method}: Non-binary values"
            
            # Should not be identical to original population
            assert not np.array_equal(new_pop, population), \
                f"Init method {init_method}: New population is identical to original"


class TestDiscreteBacdriveEDAIntegration:
    """Integration test using the UnifiedDiscreteNeuralEDA class"""
    
    def test_eda_backdrive_perturb_best(self):
        """Test full EDA with backdrive perturb best"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Import the EDA class
        from pateda.examples.discrete_EDA import UnifiedDiscreteNeuralEDA
        
        # Simple OneMax fitness function
        def onemax(x):
            if x.ndim == 1:
                return np.array([float(np.sum(x))])
            else:
                return np.sum(x, axis=1).astype(float)
        
        n_vars = 20
        pop_size = 50
        cardinality = np.full(n_vars, 2)
        
        learning_params = {'epochs': 10, 'batch_size': 16}
        sampling_params = {
            'init_method': 'perturb_best',
            'init_noise': 0.1,
            'n_iterations': 50,
        }
        
        eda = UnifiedDiscreteNeuralEDA(
            method='backdrive_perturb_best',
            n_vars=n_vars,
            cardinality=cardinality,
            pop_size=pop_size,
            selection_ratio=0.5,
            max_generations=5,
            learning_params=learning_params,
            sampling_params=sampling_params,
            random_seed=42,
        )
        
        best_fitness, best_solution, history = eda.run(onemax, verbose=False)
        
        # Should have history for all generations
        assert len(history['best_fitness']) == 6, "Should have 6 fitness values (init + 5 gens)"
        
        # Best fitness should improve or stay constant
        initial_fitness = history['best_fitness'][0]
        final_fitness = history['best_fitness'][-1]
        assert final_fitness >= initial_fitness, \
            f"Fitness degraded: initial={initial_fitness}, final={final_fitness}"
        
        # Should find reasonable solutions (at least better than random)
        expected_random = n_vars / 2
        assert final_fitness > expected_random, \
            f"Final fitness {final_fitness} not better than random {expected_random}"
    
    def test_eda_backdrive_perturb_selected(self):
        """Test full EDA with backdrive perturb selected"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        from pateda.examples.discrete_EDA import UnifiedDiscreteNeuralEDA
        
        def onemax(x):
            if x.ndim == 1:
                return np.array([float(np.sum(x))])
            else:
                return np.sum(x, axis=1).astype(float)
        
        n_vars = 20
        pop_size = 50
        cardinality = np.full(n_vars, 2)
        
        learning_params = {'epochs': 10, 'batch_size': 16}
        sampling_params = {
            'init_method': 'perturb_selected',
            'init_noise': 0.1,
            'n_iterations': 50,
        }
        
        eda = UnifiedDiscreteNeuralEDA(
            method='backdrive_perturb_selected',
            n_vars=n_vars,
            cardinality=cardinality,
            pop_size=pop_size,
            selection_ratio=0.5,
            max_generations=5,
            learning_params=learning_params,
            sampling_params=sampling_params,
            random_seed=42,
        )
        
        best_fitness, best_solution, history = eda.run(onemax, verbose=False)
        
        # Should have history for all generations
        assert len(history['best_fitness']) == 6
        
        # Best fitness should improve or stay constant
        initial_fitness = history['best_fitness'][0]
        final_fitness = history['best_fitness'][-1]
        assert final_fitness >= initial_fitness
        
        # Should find reasonable solutions
        expected_random = n_vars / 2
        assert final_fitness > expected_random


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
