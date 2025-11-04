# Neural Network Boids - Implementation Guide

## Overview

This project now supports **two modes** of boid flocking simulation:

1. **Rule-based (Original)**: Classic Reynolds boids with manually tuned separation, alignment, and cohesion weights
2. **Neural Network (New)**: AI-controlled boids using evolved neural networks for steering decisions

## What's New?

### Neural Network Architecture

- **Input Layer**: 10 neurons
  - Separation vector (x, y)
  - Alignment vector (x, y)
  - Cohesion vector (x, y)
  - Current velocity (x, y)
  - Neighbor count (normalized)
  - Average distance to neighbors (normalized)

- **Hidden Layer**: 12 neurons with Tanh activation

- **Output Layer**: 2 neurons (steering force x, y) with Tanh activation

- **Total Parameters**: 158 weights + biases to evolve

### Performance Optimizations

- **Rayon parallelization**: Ready for multi-threaded NN evaluation
- **Quadtree spatial partitioning**: Efficient neighbor queries (O(n log n))
- **Optimized matrix operations**: Cache-friendly NN forward propagation
- **Pre-allocated vectors**: Reduced memory allocations in hot paths

## How to Use

### Running Neural Network Mode

#### Visual Mode (with graphics)
```bash
# Run with neural networks
cargo run --release -- --neural

# The window will show "NEURAL EVOLUTION" in the top-left
```

#### Headless Mode (fast training)
```bash
# Train neural networks for 20 generations
cargo run --release -- --neural --headless --generations 20

# With custom population size
cargo run --release -- --neural --headless --generations 50 --population 100
```

### Running Original Rule-Based Mode

#### Visual Mode
```bash
# Run without --neural flag
cargo run --release
```

#### Headless Mode
```bash
cargo run --release -- --headless --generations 10
```

## Command-Line Arguments

| Flag | Description | Example |
|------|-------------|---------|
| `--neural` | Use neural network mode instead of rule-based | `--neural` |
| `--headless` | Run without graphics (faster training) | `--headless` |
| `--generations N` | Number of generations to evolve | `--generations 50` |
| `--population N` | Population size for genetic algorithm | `--population 100` |

## Files Generated

### Neural Network Mode
- **best_neural_genes.json**: Best evolved neural network weights
  - Contains 158 network weights
  - Physical parameters (max_speed, max_force, perception_radius)
  - Automatically saved after each generation

### Rule-Based Mode
- **best_genes.json**: Best evolved rule-based parameters
  - Contains 6 behavioral weights
  - Compatible with original implementation

## Architecture Details

### New Modules

#### `src/neural_network.rs`
- `NeuralNetwork` struct: Feedforward network with forward propagation
- Weight serialization/deserialization for evolution
- Input preparation and normalization
- Tanh activation functions

#### `src/neural_evolution.rs`
- `NeuralEvolutionManager`: GA for neural network weights
- Uniform crossover for network weights
- Gaussian mutation with clamping
- Supports both visual and headless modes

#### `src/boid.rs` (updated)
- `NeuralBoidGenes`: Genome containing NN weights + physical parameters
- `BoidGenes`: Original rule-based genes (kept for compatibility)

#### `src/simulation.rs` (updated)
- `NeuralGameState`: Simulation with NN-based steering
- `GameState`: Original rule-based simulation (unchanged)
- Both use the same quadtree optimization

## Genetic Algorithm Parameters

### Neural Network GA
```rust
POPULATION_SIZE: 50
SIMULATION_TIME: 10 seconds per individual
MUTATION_RATE: 10% (higher for NNs)
MUTATION_STRENGTH: 15% (±0.15 for weights)
ELITE_SELECTION: Top 20%
```

### Crossover Strategy
- **Network weights**: Uniform crossover (50% chance from each parent)
- **Physical parameters**: Uniform crossover for max_speed, max_force, perception_radius

### Mutation Strategy
- **Weights**: Add Gaussian noise, clamp to [-5.0, 5.0]
- **Physical params**: ±10% variation with range clamping

## Expected Results

### Performance Targets (on Intel i7 + Iris Xe)

| Boid Count | Expected FPS | Mode |
|------------|-------------|------|
| 100 | 60 FPS | Visual |
| 500 | 60 FPS | Visual (optimized) |
| 1000 | 30-60 FPS | Visual |
| 100 | Very fast | Headless |

### Evolution Progress

- **Generations 1-5**: Random exploration, fitness improves rapidly
- **Generations 5-20**: Convergence, discovers basic flocking
- **Generations 20-50**: Fine-tuning, emergent behaviors may appear
- **Generations 50+**: Diminishing returns, but can find interesting behaviors

### Fitness Metric
```
Fitness = -(avg_distance_from_center + collision_count * 10.0)
```
- **More negative = Better** (tighter flock, fewer collisions)
- Typical best fitness: -50 to -150 (depends on parameters)

## Comparing Rule-Based vs Neural

### Rule-Based Advantages
- ✅ Faster to train (6 parameters vs 158)
- ✅ More interpretable results
- ✅ Predictable behavior
- ✅ Lower computational cost

### Neural Network Advantages
- ✅ Can discover novel behaviors
- ✅ More expressive (158 parameters)
- ✅ Potentially better optimization
- ✅ Can learn complex patterns
- ✅ More realistic "learning" simulation

## Next Steps & Future Improvements

### Phase 1 (Completed)
- [x] Neural network implementation
- [x] Integration with GA
- [x] Rayon dependency for parallelization
- [x] Dual-mode support (rule-based + neural)

### Phase 2 (Future)
- [ ] True parallel NN evaluation (need thread-safe quadtree)
- [ ] SIMD optimizations for NN forward pass
- [ ] Benchmark suite for performance testing
- [ ] Visualization of NN decisions

### Phase 3 (Advanced)
- [ ] GPU compute shaders for NN inference (wgpu)
- [ ] Larger networks (experiment with architecture)
- [ ] NEAT algorithm (evolve network topology)
- [ ] Multi-objective fitness (speed + cohesion + collision avoidance)

## Troubleshooting

### Issue: Low FPS with Neural Networks
**Solution**:
- Reduce boid count
- Use headless mode for training
- Wait for future parallel optimizations

### Issue: Neural networks not learning
**Solution**:
- Increase generations (try 50+)
- Check that fitness is improving each generation
- Try adjusting mutation rate/strength in `neural_evolution.rs`

### Issue: Boids behaving strangely
**Solution**:
- This is expected early in evolution!
- Random weights produce random behavior
- After 10-20 generations, should see coordination
- Delete `best_neural_genes.json` to start fresh

## Example Training Session

```bash
# Start fresh neural evolution
rm best_neural_genes.json

# Train for 30 generations in headless mode (fast)
cargo run --release -- --neural --headless --generations 30

# Output shows:
# Generation 1: Best Fitness = -850.42
# Generation 10: Best Fitness = -245.67
# Generation 20: Best Fitness = -156.23
# Generation 30: Best Fitness = -142.89

# Now visualize the best evolved network
cargo run --release -- --neural

# Watch the boids flock using learned behavior!
```

## Technical Notes

### Why Not Fully Parallel Yet?
The current implementation does neighbor queries sequentially because the `Quadtree` struct is not thread-safe (uses `&mut self`). Future versions could:
1. Use a read-only spatial data structure (R-tree, KD-tree)
2. Build one quadtree per thread
3. Use GPU compute for all-pairs neighbor finding

### Network Size Trade-offs
- **Smaller networks** (current: 10→12→2): Faster, easier to train, less expressive
- **Larger networks** (e.g., 10→24→24→2): More expressive, slower, harder to train
- Current size is optimized for your Intel i7 CPU

### Weight Initialization
Uses Xavier initialization: `sqrt(2 / n_inputs)` for stable training from random start.

## Credits

Neural network implementation based on standard feedforward architecture with genetic algorithm training. Inspired by NEAT, genetic programming, and neuroevolution research.

## License

Same as the main project.
