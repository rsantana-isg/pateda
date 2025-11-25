import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Define the Rastrigin function (non-differentiable in PyTorch graph)
def rastrigin(x):
    """
    Rastrigin function - minimization problem
    x: numpy array or tensor of shape (batch_size, n_dims)
    returns: numpy array of shape (batch_size,)
    """
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    
    # Handle both batch and single samples
    if x_np.ndim == 1:
        x_np = x_np.reshape(1, -1)
    
    A = 10
    n = x_np.shape[1]
    result = A * n + np.sum(x_np**2 - A * np.cos(2 * np.pi * x_np), axis=1)
    return result

# Autoencoder neural network
class OptimizationAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[16, 8, 16]):
        super(OptimizationAutoencoder, self).__init__()
        self.input_dim = input_dim  # n + 1 (for f(x))
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class OptimizationTrainer:
    def __init__(self, n_dims, hidden_dims=[16, 8, 16], lr=0.001):
        self.n_dims = n_dims
        self.input_dim = n_dims + 1  # x + f(x)
        self.model = OptimizationAutoencoder(self.input_dim, hidden_dims)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Storage for solutions and evaluations
        self.solutions_history = deque(maxlen=10000)  # Store last 10k solutions
        self.loss_history = []
        
        # Store initial training data for reference
        self.initial_data = None
        
    def create_training_data(self, num_samples=1000, bounds=(-5.12, 5.12)):
        """Create training data with random x and their f(x) values"""
        x = np.random.uniform(bounds[0], bounds[1], (num_samples, self.n_dims))
        fx = rastrigin(x).reshape(-1, 1)
        
        # Combine x and f(x)
        data = np.hstack([x, fx])
        self.initial_data = data.copy()
        return torch.tensor(data, dtype=torch.float32)
    
    def update_training_data(self, current_data, new_solutions, keep_ratio=0.7):
        """
        Update training data by selecting best solutions from current data and new predictions
        
        Args:
            current_data: current training data tensor
            new_solutions: list of new solutions from predictions
            keep_ratio: ratio of current data to keep vs replace with new solutions
        """
        if not new_solutions:
            return current_data
            
        # Convert current data to numpy for processing
        current_np = current_data.numpy()
        
        # Convert new solutions to numpy array format [x, f(x)]
        new_solutions_np = []
        for solution in new_solutions:
            x = solution['x']
            fx = solution['f(x)']
            new_solutions_np.append(np.concatenate([x, [fx]]))
        
        new_solutions_np = np.array(new_solutions_np)
        
        # Combine current data and new solutions
        all_solutions = np.vstack([current_np, new_solutions_np])
        
        # Sort by objective value (f(x)) - lower is better for minimization
        sorted_indices = np.argsort(all_solutions[:, -1])  # Sort by f(x) column
        best_solutions = all_solutions[sorted_indices]
        
        # Keep the best solutions (maintain dataset size)
        num_keep = len(current_np)
        updated_data = best_solutions[:num_keep]
        
        print(f"  Data update: kept {int(keep_ratio * 100)}% best solutions "
              f"(best f(x): {updated_data[0, -1]:.4f}, "
              f"worst f(x): {updated_data[-1, -1]:.4f})")
        
        return torch.tensor(updated_data, dtype=torch.float32)
    
    def custom_loss(self, inputs, outputs, objective_func, store_predictions=True):
        """
        Custom loss function: MSE([x,f(x)], [x',f'(x')]) * (f(x') - f(x))
        
        Args:
            inputs: original input [x, f(x)] of shape (batch_size, n_dims + 1)
            outputs: autoencoder output [x', f'(x')] of shape (batch_size, n_dims + 1)
            objective_func: the objective function to evaluate (e.g., Rastrigin)
            store_predictions: whether to store predictions for data updating
        """
        batch_size = inputs.shape[0]
        
        # Split inputs and outputs
        x_original = inputs[:, :-1]  # Original x
        fx_original = inputs[:, -1]  # Original f(x)
        
        x_predicted = outputs[:, :-1]  # Predicted x'
        fx_predicted_nn = outputs[:, -1]  # Neural network's prediction of f(x')
        
        # 1. Compute reconstruction MSE
        mse_loss = nn.MSELoss()(inputs, outputs)
        
        # 2. Evaluate the actual f(x') using the objective function
        # Detach to prevent gradient flow through objective function evaluation
        with torch.no_grad():
            fx_predicted_actual = torch.tensor(
                objective_func(x_predicted), 
                dtype=torch.float32, 
                device=inputs.device
            )
        
        # 3. Store solutions and evaluations
        if store_predictions:
            self._store_solutions(x_predicted.detach(), fx_predicted_actual.detach())
        
        # 4. Compute improvement term: f(x') - f(x)
        # Since we're minimizing, we want f(x') < f(x), so (f(x') - f(x)) should be negative
        improvement_term = fx_predicted_actual - fx_original
        
        # 5. Combine losses: MSE * improvement
        # We use torch.exp() to ensure positive scaling and avoid negative loss values
        # This formulation encourages both good reconstruction AND improvement in objective
        combined_loss = mse_loss * torch.exp(improvement_term)
        
        return combined_loss.mean(), mse_loss.mean(), improvement_term.mean()
    
    def _store_solutions(self, x_predicted, fx_predicted_actual):
        """Store predicted solutions and their objective values"""
        x_np = x_predicted.cpu().numpy()
        fx_np = fx_predicted_actual.cpu().numpy()
        
        for i in range(x_np.shape[0]):
            self.solutions_history.append({
                'x': x_np[i].copy(),
                'f(x)': fx_np[i].copy()
            })
    
    def train_epoch(self, dataloader, objective_func, store_predictions=True):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_improvement = 0
        batch_predictions = []
        
        for batch_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch)
            
            # Compute custom loss
            loss, mse_loss, improvement = self.custom_loss(
                batch, outputs, objective_func, store_predictions=store_predictions
            )
            
            # Store predictions for data updating
            if store_predictions:
                with torch.no_grad():
                    x_predicted = outputs[:, :-1]
                    fx_predicted_actual = torch.tensor(
                        objective_func(x_predicted), 
                        dtype=torch.float32, 
                        device=batch.device
                    )
                    for i in range(x_predicted.shape[0]):
                        batch_predictions.append({
                            'x': x_predicted[i].detach().cpu().numpy(),
                            'f(x)': fx_predicted_actual[i].item()
                        })
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_improvement += improvement.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        avg_improvement = total_improvement / len(dataloader)
        
        self.loss_history.append({
            'total_loss': avg_loss,
            'mse_loss': avg_mse,
            'improvement': avg_improvement
        })
        
        return avg_loss, avg_mse, avg_improvement, batch_predictions
    
    def get_best_solution(self):
        """Get the best solution found during training"""
        if not self.solutions_history:
            return None, float('inf')
        
        best_solution = min(self.solutions_history, key=lambda x: x['f(x)'])
        return best_solution['x'], best_solution['f(x)']

# Enhanced training loop with data updating
def main():
    # Configuration
    n_dims = 2
    batch_size = 25
    num_epochs = 500
    hidden_dims = [16, 8, 16]
    dataset_size = 500  # Smaller dataset for more frequent updates
    
    # Create trainer
    trainer = OptimizationTrainer(n_dims, hidden_dims)
    
    # Create initial training data
    train_data = trainer.create_training_data(num_samples=dataset_size)
    
    print("Starting training with dynamic data updates...")
    
    for epoch in range(num_epochs):
        # Create dataloader with current data
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        
        # Train for one epoch and get predictions
        loss, mse, improvement, epoch_predictions = trainer.train_epoch(
            train_loader, rastrigin, store_predictions=True
        )
        
        # Update training data with best solutions every few epochs
        k = 20
        if epoch % k == 0 and epoch > 0:  # Update every k epochs after first
            print(f"Updating training data at epoch {epoch}...")
            train_data = trainer.update_training_data(
                train_data, epoch_predictions, keep_ratio=1.0
            )
        
        if epoch % 10 == 0:
            best_x, best_fx = trainer.get_best_solution()
            print(f'Epoch {epoch:3d}: Loss = {loss:.4f}, MSE = {mse:.4f}, '
                  f'Improvement = {improvement:.4f}, Best f(x) = {best_fx:.4f}')
            
            # Also show current training data quality
            current_data_np = train_data.numpy()
            current_best_fx = np.min(current_data_np[:, -1])
            current_avg_fx = np.mean(current_data_np[:, -1])
            print(f'          Current data - Best: {current_best_fx:.4f}, Avg: {current_avg_fx:.4f}')
    
    # Final results
    best_x, best_fx = trainer.get_best_solution()
    print(f"\nTraining completed!")
    print(f"Best solution found: x = {best_x}, f(x) = {best_fx:.6f}")
    print(f"Global optimum: f([0,0,...,0]) = 0.0")
    
    # Plot training history
    plot_training_history(trainer.loss_history)
    
    # Plot solution quality progression
    plot_solution_quality(trainer)

def plot_training_history(loss_history):
    """Plot the training history"""
    epochs = range(len(loss_history))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, [h['total_loss'] for h in loss_history])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, [h['mse_loss'] for h in loss_history])
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, [h['improvement'] for h in loss_history])
    plt.title('Average Improvement (f(x\') - f(x))')
    plt.xlabel('Epoch')
    plt.ylabel('Improvement')
    
    plt.tight_layout()
    plt.show()

def plot_solution_quality(trainer):
    """Plot the progression of solution quality over epochs"""
    if not trainer.solutions_history:
        return
    
    # Extract best solution at different points in training
    checkpoints = np.linspace(0, len(trainer.solutions_history)-1, 20, dtype=int)
    best_values = []
    
    for checkpoint in checkpoints:
        recent_solutions = list(trainer.solutions_history)[:checkpoint+1]
        if recent_solutions:
            best_fx = min(sol['f(x)'] for sol in recent_solutions)
            best_values.append(best_fx)
    
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoints, best_values, 'b-o', linewidth=2, markersize=4)
    plt.axhline(y=0, color='r', linestyle='--', label='Global Optimum (0.0)')
    plt.xlabel('Number of Solutions Evaluated')
    plt.ylabel('Best f(x) Found')
    plt.title('Progression of Solution Quality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(115)
    main()
    
