# Quick Start Guide - Neural Network Boids

## 🚀 Run Commands

### Neural Network Mode
```bash
# Visual evolution (watch learning happen)
cargo run --release -- --neural

# Fast training (no graphics)
cargo run --release -- --neural --headless --generations 20
```

### Rule-Based Mode (Original)
```bash
# Visual evolution
cargo run --release

# Fast training
cargo run --release -- --headless --generations 10
```

## 📊 What You'll See

### Neural Mode
- Window title: "NEURAL EVOLUTION"
- Status shows: Generation, Individual, Best Fitness
- Boids learn flocking from scratch
- Saves to: `best_neural_genes.json`

### Rule-Based Mode
- Window title: "Boids Evolution"
- Status shows: Generation, Individual, Best Fitness
- Boids use classic Reynolds rules
- Saves to: `best_genes.json`

## 🎯 Training Workflow

```bash
# Step 1: Train overnight (50 generations)
cargo run --release -- --neural --headless --generations 50

# Step 2: Watch the result
cargo run --release -- --neural

# Step 3: Compare with rule-based
cargo run --release
```

## 🔧 Command Flags

| Flag | What it does |
|------|-------------|
| `--neural` | Use neural networks instead of rules |
| `--headless` | No graphics (faster training) |
| `--generations N` | How many generations to evolve |
| `--population N` | Population size (default: 50) |

## 📈 Fitness Values

- **Higher (less negative) = Better flocking**
- Good fitness: -100 to -200
- Excellent fitness: -50 to -100
- Watch it improve each generation!

## 🐛 Troubleshooting

**Nothing happens after running command:**
- Program might be compiling (wait 3 minutes first time)
- Check that you're in the project directory

**Boids flying everywhere randomly:**
- This is normal at generation 1!
- Evolution needs 10-20 generations to learn
- Be patient - learning takes time

**Low FPS:**
- Normal with neural networks (more computation)
- Use `--headless` mode for faster training
- Reduce boid count if needed (edit `simulation.rs` line 445)

## 📚 More Info

- Detailed guide: [NEURAL_NETWORK_GUIDE.md](NEURAL_NETWORK_GUIDE.md)
- Implementation details: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## 🎓 Understanding the Output

```
NEURAL EVOLUTION
Generation: 1
Individual: 25/50
Best Fitness (last gen): -450.23
```

- **Generation**: Current evolution cycle
- **Individual**: Which genome is being tested (1-50)
- **Best Fitness**: How good the best boid from last generation was

**Fitness improves over time:**
- Gen 1: -800 (random behavior)
- Gen 10: -300 (basic coordination)
- Gen 20: -150 (good flocking)
- Gen 50: -100 (excellent flocking)

## 💡 Pro Tips

1. **Start small**: Try 10 generations first to see how it works
2. **Use headless**: Much faster for serious training
3. **Compare modes**: Run both neural and rule-based to see the difference
4. **Experiment**: Change population size, mutation rates in code
5. **Be patient**: Neural learning takes longer but can find better solutions

## ⚡ Performance Expectations

Your hardware: Intel i7 + Iris Xe Graphics

| Boid Count | FPS (Expected) | Recommendation |
|-----------|----------------|----------------|
| 100 | 60 FPS | ✅ Perfect |
| 500 | 30-60 FPS | ✅ Good |
| 1000 | 15-30 FPS | ⚠️ Usable |
| 2000+ | <15 FPS | ❌ Too slow |

**For best experience**: Stick with 100-500 boids in visual mode.

---

**Ready to watch AI learn to flock? Run the first command and enjoy! 🐦🧠**
