# Paradigms of Multi-Objective Optimization Algorithms

Multi-objective optimization algorithms are primarily categorized into three major paradigms based on how they evaluate solutions, maintain diversity, and guide the search process toward the Pareto optimal set. 

---

## 1. Pareto-Based Approaches

### Core Principle
This paradigm relies directly on the mathematical concept of **Pareto dominance**. A solution $x_1$ is said to dominate another solution $x_2$ if $x_1$ is no worse than $x_2$ in all objectives and is strictly better than $x_2$ in at least one objective.

### Mechanism
* **Ranking & Sorting:** Solutions within a population are sorted into hierarchical non-dominated layers (fronts). 
* **Selection Pressure:** Individuals in the first front (undominated by any others) receive the highest fitness values, followed by subsequent fronts.
* **Diversity Maintenance:** Secondary mechanisms like crowding distance, niching, or clustering are employed to ensure a well-distributed spreadsheet of solutions across the front.

### Strengths
* Highly intuitive and maps directly onto the definition of optimality in multi-objective spaces.
* Well-established mathematical foundations.
* Works exceptionally well for low-dimensional objective spaces (typically $\le 3$ objectives).

### Limitations
* **Scalability:** Computational complexity scales poorly as the number of objectives increases.
* **Loss of Selection Pressure:** In high-dimensional spaces (many-objective optimization), almost all solutions become mutually non-dominated, causing the selection pressure toward the true Pareto front to degrade significantly.

### Prominent Examples
* **NSGA-II** (Non-dominated Sorting Genetic Algorithm II)
* **SPEA2** (Strength Pareto Evolutionary Algorithm 2)

---

## 2. Metric / Indicator-Based Approaches

### Core Principle
This paradigm shifts away from direct qualitative comparison (dominance) to quantitative metrics. It utilizes **scalar performance indicators** to evaluate and assign a single quality score to an entire population or individual solutions within a set.

### Mechanism
* **Performance Metrics:** The search is guided by metrics that inherently measure both convergence (closeness to the true Pareto front) and diversity (uniform spread of solutions).
* **Selection:** Solutions are selected based on their contribution to maximizing or minimizing the chosen indicator.

### Strengths
* Provides a single, clear scalar value to compare populations or individuals.
* Inherently balances convergence and diversity without requiring explicit, secondary diversity preservation steps.
* Can overcome the loss of selection pressure encountered by Pareto-based methods in many-objective problems.

### Limitations
* **Computational Cost:** Calculating certain comprehensive indicators—most notably the exact **Hypervolume (HV)**—becomes exponentially expensive ($O(n^k)$ where $k$ is the number of objectives) as dimensionality grows.

### Prominent Examples
* **IBEA** (Indicator-Based Evolutionary Algorithm)
* **SMS-EMOA** (S-Metric Selection Evolutionary Multi-Objective Algorithm)
* **IGD-based Algorithms** (Inverted Generational Distance guided searches)

---

## 3. Decomposition-Based Approaches

### Core Principle
This paradigm divides and conquers by translating a complex multi-objective optimization problem into a set of multiple, simpler **single-objective optimization subproblems** using mathematical scalarizing or aggregation functions.

### Mechanism
* **Weight Vectors:** Employs a predefined, uniformly distributed set of weight vectors. Each weight vector defines a specific single-objective subproblem.
* **Cooperative Optimization:** All subproblems are optimized simultaneously and cooperatively. Each subproblem leverages information from its designated neighborhood (adjacent weight vectors) to update solutions.
* **Common Scalarization Functions:**
  * *Weighted Sum Approach:* Combines objectives linearly (struggles with non-convex Pareto fronts).
  * *Tchebycheff Decomposition:* Minimizes the distance to a reference point (handles non-convex fronts effectively).
  * *Penalty-based Boundary Intersection (PBI):* Balances distance to the reference point and direction perpendicular to the weight vector.

### Strengths
* Highly effective for both multi-objective and many-objective optimization problems because it circumvents the failure of Pareto dominance completely.
* Lower computational complexity per generation compared to high-dimensional metric or sorting-based methods.
* Provides an elegant structure to plug in classical mathematical programming single-objective solvers.

### Limitations
* **Geometry Dependency:** Performance heavily relies on the underlying geometry and curvature of the true Pareto Front (e.g., highly non-convex, disconnected, or degenerate fronts can degrade efficiency).
* **Parameter Sensitivity:** Requires a carefully tuned set of weight vectors and neighborhood sizes to fit the problem scale accurately.

### Prominent Examples
* **MOEA/D** (Multi-Objective Evolutionary Algorithm based on Decomposition)
