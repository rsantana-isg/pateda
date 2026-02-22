#!/bin/bash
# Demonstration of all DbD variants

echo "=========================================="
echo "DbD Variants Demonstration"
echo "=========================================="
echo ""
echo "Problem: OneMax (n=20)"
echo "All variants run for 5 generations"
echo ""

echo "1. Standard DbD with MSE loss"
python3 examples/discrete_DbD_EDA.py 0 OneMax 20 80 5 0.5 dbd relu mse 10 10 0 0.1 0 0 2>&1 | grep "Best Fitness:" | tail -1

echo "2. DbD-Weighted (fitness-weighted MSE)"
python3 examples/discrete_DbD_EDA.py 0 OneMax 20 80 5 0.5 dbd relu weighted_mse 10 10 0 0.1 0 0 2>&1 | grep "Best Fitness:" | tail -1

echo "3. DbD-Ranking (ranking loss)"
python3 examples/discrete_DbD_EDA.py 0 OneMax 20 80 5 0.5 dbd relu ranking 10 10 0 0.1 0 0 2>&1 | grep "Best Fitness:" | tail -1

echo "4. DbD-Huber (robust to outliers)"
python3 examples/discrete_DbD_EDA.py 0 OneMax 20 80 5 0.5 dbd relu huber 10 10 0 0.1 0 0 2>&1 | grep "Best Fitness:" | tail -1

echo "5. C-DbD (conditional with fitness guidance)"
python3 examples/discrete_DbD_EDA.py 0 OneMax 20 80 5 0.5 dbd relu mse 10 10 0 0.1 1 0 2>&1 | grep "Best Fitness:" | tail -1

echo "6. M-DbD (Markov initialization)"
python3 examples/discrete_DbD_EDA.py 0 OneMax 20 80 5 0.5 dbd relu mse 10 10 0 0.1 0 1 2>&1 | grep "Best Fitness:" | tail -1

echo ""
echo "=========================================="
echo "All variants completed successfully! ✅"
echo "=========================================="
