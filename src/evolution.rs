use crate::boid::BoidGenes;
use crate::simulation::GameState;
use ggez::glam::Vec2;
use ggez::{Context, GameResult, event, graphics};
use rand::Rng;

const POPULATION_SIZE: usize = 50;
const SIMULATION_TIME: f32 = 10.0;
const MUTATION_RATE: f32 = 0.05;

/// The top-level manager for the entire application, including the evolutionary process.
pub struct EvolutionManager {
    pub population: Vec<(BoidGenes, f32)>, // genes and fitness
    pub active_simulation: GameState,
    pub current_generation: usize,
    pub current_individual_index: usize,
    pub timer: f32,
    pub best_genes: BoidGenes,
    pub best_fitness_last_gen: f32,
}

impl EvolutionManager {
    /// Sets up the initial population for the GA, loading from a file if possible.
    pub fn new(ctx: &mut Context) -> Self {
        // Attempt to load the best genes from the last run.
        let initial_genes = match std::fs::read_to_string("best_genes.json") {
            Ok(json) => {
                println!("Loaded best genes from best_genes.json");
                serde_json::from_str(&json).unwrap_or_else(|_| {
                    println!("Could not parse best_genes.json, starting fresh.");
                    crate::boid::BoidGenes::new_random()
                })
            }
            Err(_) => {
                println!("No saved genes file found. Starting with random genes.");
                crate::boid::BoidGenes::new_random()
            }
        };

        // Create the initial population.
        let mut population = Vec::new();
        population.push((initial_genes, 0.0));
        for _ in 1..POPULATION_SIZE {
            population.push((crate::boid::BoidGenes::new_random(), 0.0));
        }

        EvolutionManager {
            population,
            active_simulation: GameState::new(ctx, initial_genes),
            current_generation: 1,
            current_individual_index: 0,
            timer: 0.0,
            best_genes: initial_genes,
            best_fitness_last_gen: 0.0,
        }
    }

    /// Creates the next generation of boids through selection, crossover, and mutation.
    pub fn evolve_new_generation(&mut self, ctx: &mut Context) {
        self.population
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        self.best_genes = self.population[0].0;
        self.best_fitness_last_gen = self.population[0].1;

        println!(
            "Generation {}: Best Fitness = {}",
            self.current_generation, self.best_fitness_last_gen
        );

        let json = serde_json::to_string_pretty(&self.best_genes).unwrap();
        std::fs::write("best_genes.json", json).expect("Failed to save genes.");
        println!("Saved best genes to best_genes.json");

        let elite_count = ((POPULATION_SIZE as f32 * 0.2) as usize).max(1);
        let elites: Vec<BoidGenes> = self
            .population
            .iter()
            .take(elite_count)
            .map(|(genes, _)| *genes)
            .collect();

        let mut new_population = Vec::new();
        let mut rng = rand::rng();

        for _ in 0..POPULATION_SIZE {
            let parent1 = elites[rng.random_range(0..elites.len())];
            let parent2 = elites[rng.random_range(0..elites.len())];

            let mut child_genes = BoidGenes {
                perception_radius: if rng.random_bool(0.5) {
                    parent1.perception_radius
                } else {
                    parent2.perception_radius
                },
                max_speed: if rng.random_bool(0.5) {
                    parent1.max_speed
                } else {
                    parent2.max_speed
                },
                max_force: if rng.random_bool(0.5) {
                    parent1.max_force
                } else {
                    parent2.max_force
                },
                separation_weight: if rng.random_bool(0.5) {
                    parent1.separation_weight
                } else {
                    parent2.separation_weight
                },
                alignment_weight: if rng.random_bool(0.5) {
                    parent1.alignment_weight
                } else {
                    parent2.alignment_weight
                },
                cohesion_weight: if rng.random_bool(0.5) {
                    parent1.cohesion_weight
                } else {
                    parent2.cohesion_weight
                },
            };

            if rng.random::<f32>() < MUTATION_RATE {
                let mutation_amount = child_genes.separation_weight * rng.random_range(-0.1..0.1);
                child_genes.separation_weight += mutation_amount;
            }

            new_population.push((child_genes, 0.0));
        }

        self.population = new_population;
        self.current_generation += 1;
        self.current_individual_index = 0;
        self.active_simulation = GameState::new(ctx, self.population[0].0);
    }
}

impl event::EventHandler for EvolutionManager {
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

            let next_genes = self.population[self.current_individual_index].0;
            self.active_simulation = GameState::new(ctx, next_genes);
        }

        self.active_simulation.update(ctx)
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, graphics::Color::BLACK);
        self.active_simulation.draw_to_canvas(ctx, &mut canvas)?;

        let text = graphics::Text::new(format!(
            "Generation: {}\nIndividual: {}/{}\nBest Fitness (last gen): {:.2}",
            self.current_generation,
            self.current_individual_index + 1,
            POPULATION_SIZE,
            self.best_genes.cohesion_weight
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

/// Run a simple headless GA loop for `generations` iterations.
pub fn headless_main(generations: usize, population_size: usize) {
    let width = 800.0f32;
    let height = 600.0f32;
    let boids_per_sim = 100usize;
    let dt = 1.0f32 / 60.0f32;

    let initial_genes = match std::fs::read_to_string("best_genes.json") {
        Ok(json) => {
            serde_json::from_str(&json).unwrap_or_else(|_| crate::boid::BoidGenes::new_random())
        }
        Err(_) => crate::boid::BoidGenes::new_random(),
    };

    let mut population: Vec<(BoidGenes, f32)> = Vec::new();
    population.push((initial_genes, 0.0));
    for _ in 1..population_size {
        population.push((crate::boid::BoidGenes::new_random(), 0.0));
    }

    for generation in 0..generations {
        println!("Evaluating generation {}/{}", generation + 1, generations);

        for i in 0..population.len() {
            let genes = population[i].0;
            let fitness =
                evaluate_genes_headless(genes, width, height, boids_per_sim, dt, SIMULATION_TIME);
            population[i].1 = fitness;
            println!(
                "  Individual {}/{} fitness = {}",
                i + 1,
                population.len(),
                fitness
            );
        }

        population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let best = population[0].0;
        let best_score = population[0].1;
        println!("Generation {} best = {}", generation + 1, best_score);
        let json = serde_json::to_string_pretty(&best).unwrap();
        let _ = std::fs::write("best_genes.json", json);

        let elite_count = ((population_size as f32 * 0.2) as usize).max(1);
        let elites: Vec<BoidGenes> = population
            .iter()
            .take(elite_count)
            .map(|(g, _)| *g)
            .collect();
        let mut new_population: Vec<(BoidGenes, f32)> = Vec::new();
        let mut rng = rand::rng();
        for _ in 0..population_size {
            let p1 = elites[rng.random_range(0..elites.len())];
            let p2 = elites[rng.random_range(0..elites.len())];
            let mut child = BoidGenes {
                perception_radius: if rng.random_bool(0.5) {
                    p1.perception_radius
                } else {
                    p2.perception_radius
                },
                max_speed: if rng.random_bool(0.5) {
                    p1.max_speed
                } else {
                    p2.max_speed
                },
                max_force: if rng.random_bool(0.5) {
                    p1.max_force
                } else {
                    p2.max_force
                },
                separation_weight: if rng.random_bool(0.5) {
                    p1.separation_weight
                } else {
                    p2.separation_weight
                },
                alignment_weight: if rng.random_bool(0.5) {
                    p1.alignment_weight
                } else {
                    p2.alignment_weight
                },
                cohesion_weight: if rng.random_bool(0.5) {
                    p1.cohesion_weight
                } else {
                    p2.cohesion_weight
                },
            };
            if rng.random::<f32>() < MUTATION_RATE {
                child.separation_weight += child.separation_weight * rng.random_range(-0.1..0.1);
            }
            new_population.push((child, 0.0));
        }
        population = new_population;
    }
}

/// Evaluate a single genes set headless by simulating boids for sim_time and returning fitness.
pub fn evaluate_genes_headless(
    genes: BoidGenes,
    width: f32,
    height: f32,
    boid_count: usize,
    dt: f32,
    sim_time: f32,
) -> f32 {
    let mut state = GameState::new_headless(width, height, genes, boid_count);
    let steps = (sim_time / dt).ceil() as usize;
    for _ in 0..steps {
        state.update_headless(dt, width, height);
    }
    state.calculate_fitness()
}
