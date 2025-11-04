use crate::boid::NeuralBoidGenes;
use crate::neural_network::NeuralNetwork;
use crate::simulation::NeuralGameState;
use ggez::glam::Vec2;
use ggez::{Context, GameResult, event, graphics};
use rand::Rng;

const POPULATION_SIZE: usize = 50;
const SIMULATION_TIME: f32 = 10.0;
const MUTATION_RATE: f32 = 0.10; // Higher mutation rate for NNs
const MUTATION_STRENGTH: f32 = 0.15; // Strength of mutations

/// Evolution manager for neural network-based boids.
pub struct NeuralEvolutionManager {
    pub population: Vec<(NeuralBoidGenes, f32)>, // genes and fitness
    pub active_simulation: NeuralGameState,
    pub current_generation: usize,
    pub current_individual_index: usize,
    pub timer: f32,
    pub best_genes: NeuralBoidGenes,
    pub best_fitness_last_gen: f32,
}

impl NeuralEvolutionManager {
    /// Sets up the initial population for the neural GA.
    pub fn new(ctx: &mut Context) -> Self {
        // Attempt to load the best genes from the last run
        let initial_genes = match std::fs::read_to_string("best_neural_genes.json") {
            Ok(json) => {
                println!("Loaded best neural genes from best_neural_genes.json");
                serde_json::from_str(&json).unwrap_or_else(|e| {
                    println!("Could not parse best_neural_genes.json: {}, starting fresh.", e);
                    NeuralBoidGenes::new_random()
                })
            }
            Err(_) => {
                println!("No saved neural genes file found. Starting with random genes.");
                NeuralBoidGenes::new_random()
            }
        };

        // Create the initial population
        let mut population = Vec::new();
        population.push((initial_genes.clone(), 0.0));
        for _ in 1..POPULATION_SIZE {
            population.push((NeuralBoidGenes::new_random(), 0.0));
        }

        NeuralEvolutionManager {
            population,
            active_simulation: NeuralGameState::new(ctx, initial_genes.clone()),
            current_generation: 1,
            current_individual_index: 0,
            timer: 0.0,
            best_genes: initial_genes,
            best_fitness_last_gen: 0.0,
        }
    }

    /// Creates the next generation through selection, crossover, and mutation.
    pub fn evolve_new_generation(&mut self, ctx: &mut Context) {
        // Sort by fitness in descending order (most negative = best)
        self.population
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        self.best_genes = self.population[0].0.clone();
        self.best_fitness_last_gen = self.population[0].1;

        println!(
            "Generation {}: Best Fitness = {:.2}",
            self.current_generation, self.best_fitness_last_gen
        );

        // Save best genes
        let json = serde_json::to_string_pretty(&self.best_genes).unwrap();
        std::fs::write("best_neural_genes.json", json).expect("Failed to save neural genes.");
        println!("Saved best neural genes to best_neural_genes.json");

        // Elite selection (top 20%)
        let elite_count = ((POPULATION_SIZE as f32 * 0.2) as usize).max(1);
        let elites: Vec<NeuralBoidGenes> = self
            .population
            .iter()
            .take(elite_count)
            .map(|(genes, _)| genes.clone())
            .collect();

        // Create new population
        let mut new_population = Vec::new();
        let mut rng = rand::rng();

        for _ in 0..POPULATION_SIZE {
            let parent1 = &elites[rng.random_range(0..elites.len())];
            let parent2 = &elites[rng.random_range(0..elites.len())];

            // Crossover for network weights (uniform crossover)
            let mut child_weights = Vec::new();
            for i in 0..NeuralNetwork::weight_count() {
                if rng.random_bool(0.5) {
                    child_weights.push(parent1.network_weights[i]);
                } else {
                    child_weights.push(parent2.network_weights[i]);
                }
            }

            // Crossover for physical parameters
            let child_max_speed = if rng.random_bool(0.5) {
                parent1.max_speed
            } else {
                parent2.max_speed
            };

            let child_max_force = if rng.random_bool(0.5) {
                parent1.max_force
            } else {
                parent2.max_force
            };

            let child_perception = if rng.random_bool(0.5) {
                parent1.perception_radius
            } else {
                parent2.perception_radius
            };

            let mut child_genes = NeuralBoidGenes {
                network_weights: child_weights,
                max_speed: child_max_speed,
                max_force: child_max_force,
                perception_radius: child_perception,
            };

            // Mutate network weights
            for weight in &mut child_genes.network_weights {
                if rng.random::<f32>() < MUTATION_RATE {
                    *weight += rng.random_range(-MUTATION_STRENGTH..MUTATION_STRENGTH);
                    // Clamp to reasonable range
                    *weight = weight.clamp(-5.0, 5.0);
                }
            }

            // Mutate physical parameters
            if rng.random::<f32>() < MUTATION_RATE {
                let mutation = child_genes.max_speed * rng.random_range(-0.1..0.1);
                child_genes.max_speed = (child_genes.max_speed + mutation).clamp(100.0, 400.0);
            }
            if rng.random::<f32>() < MUTATION_RATE {
                let mutation = child_genes.max_force * rng.random_range(-0.1..0.1);
                child_genes.max_force = (child_genes.max_force + mutation).clamp(100.0, 500.0);
            }
            if rng.random::<f32>() < MUTATION_RATE {
                let mutation = child_genes.perception_radius * rng.random_range(-0.1..0.1);
                child_genes.perception_radius = (child_genes.perception_radius + mutation).clamp(50.0, 200.0);
            }

            new_population.push((child_genes, 0.0));
        }

        self.population = new_population;
        self.current_generation += 1;
        self.current_individual_index = 0;
        self.active_simulation = NeuralGameState::new(ctx, self.population[0].0.clone());
    }
}

impl event::EventHandler for NeuralEvolutionManager {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        self.timer += ctx.time.delta().as_secs_f32();

        if self.timer >= SIMULATION_TIME {
            let fitness = self.active_simulation.calculate_fitness();
            self.population[self.current_individual_index].1 = fitness;
            self.timer = 0.0;

            self.current_individual_index += 1;

            if self.current_individual_index >= POPULATION_SIZE {
                self.evolve_new_generation(ctx);
            }

            let next_genes = self.population[self.current_individual_index].0.clone();
            self.active_simulation = NeuralGameState::new(ctx, next_genes);
        }

        self.active_simulation.update(ctx)
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, graphics::Color::BLACK);
        self.active_simulation.draw_to_canvas(ctx, &mut canvas)?;

        let fitness_text = if self.current_generation == 1 && self.best_fitness_last_gen == 0.0 {
            "Best Fitness: (evaluating...)".to_string()
        } else {
            format!("Best Fitness: {:.2}", self.best_fitness_last_gen)
        };

        let text = graphics::Text::new(format!(
            "NEURAL EVOLUTION\nGeneration: {}\nIndividual: {}/{}\n{}",
            self.current_generation,
            self.current_individual_index + 1,
            POPULATION_SIZE,
            fitness_text
        ));

        canvas.draw(
            &text,
            graphics::DrawParam::new()
                .dest(Vec2::new(10.0, 10.0))
                .color(graphics::Color::WHITE),
        );

        canvas.finish(ctx)
    }
}

/// Run a headless neural GA loop for `generations` iterations.
pub fn neural_headless_main(generations: usize, population_size: usize) {
    let width = 800.0f32;
    let height = 600.0f32;
    let boids_per_sim = 100usize;
    let dt = 1.0f32 / 60.0f32;

    println!("Starting headless neural evolution...");
    println!("Generations: {}, Population: {}", generations, population_size);

    let initial_genes = match std::fs::read_to_string("best_neural_genes.json") {
        Ok(json) => {
            println!("Loaded initial genes from best_neural_genes.json");
            serde_json::from_str(&json).unwrap_or_else(|_| NeuralBoidGenes::new_random())
        }
        Err(_) => {
            println!("No saved genes found, starting fresh");
            NeuralBoidGenes::new_random()
        }
    };

    let mut population: Vec<(NeuralBoidGenes, f32)> = Vec::new();
    population.push((initial_genes, 0.0));
    for _ in 1..population_size {
        population.push((NeuralBoidGenes::new_random(), 0.0));
    }

    for generation in 0..generations {
        println!("\n=== Generation {}/{} ===", generation + 1, generations);

        // Evaluate each individual
        for i in 0..population.len() {
            let genes = population[i].0.clone();
            let fitness =
                evaluate_neural_genes_headless(genes, width, height, boids_per_sim, dt, SIMULATION_TIME);
            population[i].1 = fitness;
            println!(
                "  Individual {}/{}: fitness = {:.2}",
                i + 1,
                population.len(),
                fitness
            );
        }

        // Sort by fitness (most negative = best)
        population.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let best = population[0].0.clone();
        let best_score = population[0].1;
        println!("Generation {} BEST fitness = {:.2}", generation + 1, best_score);

        // Save best genes
        let json = serde_json::to_string_pretty(&best).unwrap();
        let _ = std::fs::write("best_neural_genes.json", json);

        // Elite selection and breeding
        let elite_count = ((population_size as f32 * 0.2) as usize).max(1);
        let elites: Vec<NeuralBoidGenes> = population
            .iter()
            .take(elite_count)
            .map(|(g, _)| g.clone())
            .collect();

        let mut new_population: Vec<(NeuralBoidGenes, f32)> = Vec::new();
        let mut rng = rand::rng();

        for _ in 0..population_size {
            let parent1 = &elites[rng.random_range(0..elites.len())];
            let parent2 = &elites[rng.random_range(0..elites.len())];

            // Crossover
            let mut child_weights = Vec::new();
            for i in 0..NeuralNetwork::weight_count() {
                if rng.random_bool(0.5) {
                    child_weights.push(parent1.network_weights[i]);
                } else {
                    child_weights.push(parent2.network_weights[i]);
                }
            }

            let child_max_speed = if rng.random_bool(0.5) {
                parent1.max_speed
            } else {
                parent2.max_speed
            };
            let child_max_force = if rng.random_bool(0.5) {
                parent1.max_force
            } else {
                parent2.max_force
            };
            let child_perception = if rng.random_bool(0.5) {
                parent1.perception_radius
            } else {
                parent2.perception_radius
            };

            let mut child = NeuralBoidGenes {
                network_weights: child_weights,
                max_speed: child_max_speed,
                max_force: child_max_force,
                perception_radius: child_perception,
            };

            // Mutation
            for weight in &mut child.network_weights {
                if rng.random::<f32>() < MUTATION_RATE {
                    *weight += rng.random_range(-MUTATION_STRENGTH..MUTATION_STRENGTH);
                    *weight = weight.clamp(-5.0, 5.0);
                }
            }

            if rng.random::<f32>() < MUTATION_RATE {
                let mutation = child.max_speed * rng.random_range(-0.1..0.1);
                child.max_speed = (child.max_speed + mutation).clamp(100.0, 400.0);
            }
            if rng.random::<f32>() < MUTATION_RATE {
                let mutation = child.max_force * rng.random_range(-0.1..0.1);
                child.max_force = (child.max_force + mutation).clamp(100.0, 500.0);
            }
            if rng.random::<f32>() < MUTATION_RATE {
                let mutation = child.perception_radius * rng.random_range(-0.1..0.1);
                child.perception_radius = (child.perception_radius + mutation).clamp(50.0, 200.0);
            }

            new_population.push((child, 0.0));
        }

        population = new_population;
    }

    println!("\n=== Evolution Complete ===");
    println!("Best genes saved to best_neural_genes.json");
}

/// Evaluate a single neural gene set headless.
fn evaluate_neural_genes_headless(
    genes: NeuralBoidGenes,
    width: f32,
    height: f32,
    boid_count: usize,
    dt: f32,
    sim_time: f32,
) -> f32 {
    let mut state = NeuralGameState::new_headless(width, height, genes, boid_count);
    let steps = (sim_time / dt).ceil() as usize;
    for _ in 0..steps {
        state.update_headless(dt, width, height);
    }
    state.calculate_fitness()
}
