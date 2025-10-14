use ggez::conf::WindowSetup;
use ggez::event::{self, EventHandler};
use ggez::glam::Vec2;
use ggez::graphics::{self, Color, DrawMode};
use ggez::{Context, ContextBuilder, GameResult};
use rand::Rng;

// --- Constants ---
const TRAIL_LIFESPAN: f32 = 1.0;
const POPULATION_SIZE: usize = 50;
const SIMULATION_TIME: f32 = 10.0; // Seconds to test each generation member
const MUTATION_RATE: f32 = 0.05;
// --- Struct Definitions ---

/// Represents the "DNA" of a flock, holding all tunable parameters.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct BoidGenes {
    perception_radius: f32,
    max_speed: f32,
    max_force: f32,
    separation_weight: f32,
    alignment_weight: f32,
    cohesion_weight: f32,
}

/// A simplified point used by the Quadtree for efficiency.
#[derive(Debug, Clone, Copy)]
struct Point {
    position: Vec2,
    velocity: Vec2,
}

/// A single point in a boid's trail.
struct TrailPoint {
    position: Vec2,
    lifespan: f32,
}

/// A rectangle defined by its center and half-dimensions (w, h).
#[derive(Debug, Clone, Copy)]
struct Rectangle {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

/// The Quadtree structure for spatial partitioning.
struct Quadtree {
    boundary: Rectangle,
    capacity: usize,
    points: Vec<Point>,
    divided: bool,
    
    northwest: Option<Box<Quadtree>>,
    northeast: Option<Box<Quadtree>>,
    southwest: Option<Box<Quadtree>>,
    southeast: Option<Box<Quadtree>>,
}

/// An autonomous agent in the flock.
#[derive(Clone)]
struct Boid {
    position: Vec2,
    velocity: Vec2,
    acceleration: Vec2,
}

/// Manages the state of a single simulation instance.
struct GameState {
    boids: Vec<Boid>,
    trails: Vec<TrailPoint>,
    genes: BoidGenes,
}

/// The top-level manager for the entire application, including the evolutionary process.
struct EvolutionManager {
    population: Vec<(BoidGenes, f32)>, // Holds the genes and their fitness score
    active_simulation: GameState,
    current_generation: usize,
    current_individual_index: usize,
    timer: f32,
    best_genes: BoidGenes,
    best_fitness_last_gen: f32,
}
// --- Implementations ---

impl BoidGenes {
    /// Creates a new set of genes with random values within reasonable ranges.
    fn new_random() -> Self {
        let mut rng = rand::rng();
        BoidGenes {
            perception_radius: rng.random_range(25.0..150.0),
            max_speed: rng.random_range(100.0..300.0),
            max_force: rng.random_range(100.0..400.0),
            separation_weight: rng.random_range(0.5..2.5),
            alignment_weight: rng.random_range(0.5..2.5),
            cohesion_weight: rng.random_range(0.5..2.5),
        }
    }
}

impl Rectangle {
    /// Checks if a point is within the rectangle's bounds.
    fn contains(&self, point: &Point) -> bool {
        (point.position.x >= self.x - self.w)
            && (point.position.x < self.x + self.w)
            && (point.position.y >= self.y - self.h)
            && (point.position.y < self.y + self.h)
    }

    /// Checks if another rectangle intersects with this one.
    fn intersects(&self, range: &Rectangle) -> bool {
        !(range.x - range.w > self.x + self.w
            || range.x + range.w < self.x - self.w
            || range.y - range.h > self.y + self.h
            || range.y + range.h < self.y - self.h)
    }
}

impl Quadtree {
    fn new(boundary: Rectangle, capacity: usize) -> Self {
        Quadtree {
            boundary,
            capacity,
            points: Vec::new(),
            divided: false,
            northwest: None,
            northeast: None,
            southwest: None,
            southeast: None,
        }
    }

    fn subdivide(&mut self) {
        let x = self.boundary.x;
        let y = self.boundary.y;
        let w = self.boundary.w / 2.0;
        let h = self.boundary.h / 2.0;

        let nw = Rectangle {
            x: x - w,
            y: y - h,
            w,
            h,
        };
        self.northwest = Some(Box::new(Quadtree::new(nw, self.capacity)));
        let ne = Rectangle {
            x: x + w,
            y: y - h,
            w,
            h,
        };
        self.northeast = Some(Box::new(Quadtree::new(ne, self.capacity)));
        let sw = Rectangle {
            x: x - w,
            y: y + h,
            w,
            h,
        };
        self.southwest = Some(Box::new(Quadtree::new(sw, self.capacity)));
        let se = Rectangle {
            x: x + w,
            y: y + h,
            w,
            h,
        };
        self.southeast = Some(Box::new(Quadtree::new(se, self.capacity)));

        self.divided = true;
    }

    fn insert(&mut self, point: Point) -> bool {
        if !self.boundary.contains(&point) {
            return false;
        }

        if self.points.len() < self.capacity {
            self.points.push(point);
            return true;
        } else {
            if !self.divided {
                self.subdivide();
            }
            if self.northeast.as_mut().unwrap().insert(point) {
                return true;
            }
            if self.northwest.as_mut().unwrap().insert(point) {
                return true;
            }
            if self.southeast.as_mut().unwrap().insert(point) {
                return true;
            }
            if self.southwest.as_mut().unwrap().insert(point) {
                return true;
            }
        }
        false
    }

    /// Finds all points within a given rectangular range.
    fn query(&self, range: &Rectangle, found: &mut Vec<Point>) {
        if !self.boundary.intersects(range) {
            return;
        }

        for p in &self.points {
            if range.contains(p) {
                found.push(*p);
            }
        }

        if self.divided {
            self.northwest.as_ref().unwrap().query(range, found);
            self.northeast.as_ref().unwrap().query(range, found);
            self.southwest.as_ref().unwrap().query(range, found);
            self.southeast.as_ref().unwrap().query(range, found);
        }
    }
}

impl Boid {
    fn new(screen_width: f32, screen_height: f32) -> Self {
        let mut rng = rand::rng();
        Boid {
            position: Vec2::new(
                rng.random_range(0.0..screen_width),
                rng.random_range(0.0..screen_height),
            ),
            velocity: Vec2::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
                .normalize()
                * 100.0,
            acceleration: Vec2::ZERO,
        }
    }
}

impl GameState {
    /// Creates a new simulation instance with a given set of genes.
    fn new(ctx: &mut Context, genes: BoidGenes) -> Self {
        let (width, height) = ctx.gfx.drawable_size();
        let mut boids = Vec::new();
        for _ in 0..100 {
            boids.push(Boid::new(width, height));
        }
        GameState {
            boids,
            trails: Vec::new(),
            genes,
        }
    }

    /// Runs the core simulation logic for one frame.
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        let dt = ctx.time.delta().as_secs_f32();
        let (width, height) = ctx.gfx.drawable_size();

        // Update trails
        for boid in &self.boids {
            self.trails.push(TrailPoint {
                position: boid.position,
                lifespan: TRAIL_LIFESPAN,
            });
        }
        for trail in &mut self.trails {
            trail.lifespan -= dt;
        }
        self.trails.retain(|trail| trail.lifespan > 0.0);

        
        let boundary = Rectangle {
            x: width / 2.0,
            y: height / 2.0,
            w: width / 2.0,
            h: height / 2.0,
        };
        let mut qtree = Quadtree::new(boundary, 4);
        for boid in &self.boids {
            qtree.insert(Point {
                position: boid.position,
                velocity: boid.velocity,
            });
        }

        for boid in &mut self.boids {
            let range = Rectangle {
                x: boid.position.x,
                y: boid.position.y,
                w: self.genes.perception_radius,
                h: self.genes.perception_radius,
            };

            let mut potential_neighbors = Vec::new();
            qtree.query(&range, &mut potential_neighbors);

            // Filter out the boid itself from the neighbor list using distance check.
            let neighbors: Vec<Point> = potential_neighbors
                .into_iter()
                .filter(|p| boid.position.distance(p.position) > 0.0001)
                .collect();

            
            let mut separation = Vec2::ZERO;
            let mut alignment = Vec2::ZERO;
            let mut cohesion = Vec2::ZERO;
            if !neighbors.is_empty() {
                let mut separation = Vec2::ZERO;
                let mut alignment = Vec2::ZERO;
                let mut cohesion = Vec2::ZERO;
                let neighbor_count = neighbors.len() as f32;

                for other in &neighbors {
                    let diff = boid.position - other.position;
                    let distance = diff.length();
                    if distance > 0.0 {
                        separation += diff.normalize() / distance;
                    }
                    alignment += other.velocity;
                    cohesion += other.position;
                }

                // Calculate and apply the steering forces.
                alignment /= neighbor_count;
                alignment = alignment.normalize_or_zero() * self.genes.max_speed;
                alignment -= boid.velocity;
                alignment = alignment.clamp_length_max(self.genes.max_force);

                cohesion /= neighbor_count;
                cohesion -= boid.position;
                cohesion = cohesion.normalize_or_zero() * self.genes.max_speed;
                cohesion -= boid.velocity;
                cohesion = cohesion.clamp_length_max(self.genes.max_force);

                separation /= neighbor_count;
                if separation.length() > 0.0 {
                    separation = separation.normalize_or_zero() * self.genes.max_speed;
                    separation -= boid.velocity;
                    separation = separation.clamp_length_max(self.genes.max_force);
                }

                boid.acceleration += separation * self.genes.separation_weight;
                boid.acceleration += alignment * self.genes.alignment_weight;
                boid.acceleration += cohesion * self.genes.cohesion_weight;
            }
            
            boid.velocity += boid.acceleration * dt;
            boid.velocity = boid.velocity.clamp_length_max(self.genes.max_speed);
            boid.position += boid.velocity * dt;
            boid.acceleration = Vec2::ZERO;

            if boid.position.x < 0.0 {
                boid.position.x += width;
            }
            if boid.position.x >= width {
                boid.position.x -= width;
            }
            if boid.position.y < 0.0 {
                boid.position.y += height;
            }
            if boid.position.y >= height {
                boid.position.y -= height;
            }
        }
        Ok(())
    }

    /// Draws the simulation to the screen for one frame.
    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, Color::BLACK);

        
        let trail_mesh =
            graphics::Mesh::new_circle(ctx, DrawMode::fill(), Vec2::ZERO, 2.0, 0.1, Color::WHITE)?;
        for trail in &self.trails {
            
            let alpha = trail.lifespan / TRAIL_LIFESPAN;
            let color = Color::new(1.0, 1.0, 1.0, alpha * 0.5);
            canvas.draw(
                &trail_mesh,
                graphics::DrawParam::new().dest(trail.position).color(color),
            );
        }

        let boid_mesh = graphics::Mesh::new_polygon(
            ctx,
            DrawMode::fill(),
            &[
                Vec2::new(0.0, -10.0),
                Vec2::new(5.0, 10.0),
                Vec2::new(-5.0, 10.0),
            ],
            Color::WHITE,
        )?;
        for boid in &self.boids {
            let angle = boid.velocity.y.atan2(boid.velocity.x) + 90.0_f32.to_radians();
            canvas.draw(
                &boid_mesh,
                graphics::DrawParam::new()
                    .dest(boid.position)
                    .rotation(angle),
            );
        }

        canvas.finish(ctx)
    }
    /// Calculates a fitness score based on the flock's cohesion and separation.
    fn calculate_fitness(&self) -> f32 {
        if self.boids.is_empty() {
            return 0.0;
        }

        let mut total_distance_from_center = 0.0;
        let mut collision_count = 0;
        let mut center_of_mass = Vec2::ZERO;

        for boid in &self.boids {
            center_of_mass += boid.position;
        }
        center_of_mass /= self.boids.len() as f32;

        for boid in &self.boids {
            total_distance_from_center += boid.position.distance(center_of_mass);
            for other in &self.boids {
                if boid.position != other.position && boid.position.distance(other.position) < 10.0
                {
                    collision_count += 1;
                }
            }
        }

        let avg_distance = total_distance_from_center / self.boids.len() as f32;

        // We want to minimize distance and collisions, so a higher score is worse.
        // We will select for the lowest score.
        avg_distance + (collision_count as f32 * 10.0) // Penalize collisions heavily
    }
}

impl EvolutionManager {
    /// Creates the next generation of boids through selection, crossover, and mutation.
    fn evolve_new_generation(&mut self, ctx: &mut Context) {
        // Sort the population by fitness to find the best performers.
        self.population
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Identify the champion of the completed generation.
        self.best_genes = self.population[0].0;
        self.best_fitness_last_gen = self.population[0].1;

        println!(
            "Generation {}: Best Fitness = {}",
            self.current_generation, self.best_fitness_last_gen
        );

        let json = serde_json::to_string_pretty(&self.best_genes).unwrap();
        std::fs::write("best_genes.json", json).expect("Failed to save genes.");
        println!("Saved best genes to best_genes.json");

        // Keep the top 20% of the population as "elites".
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
            // Create a new child by breeding two random elite parents.
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

            // Apply a small random mutation with a given probability.
            if rng.random::<f32>() < MUTATION_RATE {
                let mutation_amount = child_genes.separation_weight * rng.random_range(-0.1..0.1);
                child_genes.separation_weight += mutation_amount;
            }

            new_population.push((child_genes, 0.0));
        }

        // Replace the old population with the new one.
        self.population = new_population;
        self.current_generation += 1;
        self.current_individual_index = 0;
        self.active_simulation = GameState::new(ctx, self.population[0].0);
    }
/// Sets up the initial population for the GA, loading from a file if possible.
fn new(ctx: &mut Context) -> Self {
    // Attempt to load the best genes from the last run.
    let initial_genes = match std::fs::read_to_string("best_genes.json") {
        Ok(json) => {
            println!("Loaded best genes from best_genes.json");
            // If the file is found but parsing fails, fall back to random.
            serde_json::from_str(&json).unwrap_or_else(|_| {
                println!("Could not parse best_genes.json, starting fresh.");
                BoidGenes::new_random()
            })
        }
        Err(_) => {
            // If the file is not found, start with random genes.
            println!("No saved genes file found. Starting with random genes.");
            BoidGenes::new_random()
        }
    };

    // Create the initial population.
    let mut population = Vec::new();
    // The first individual is the best one we know (either loaded or newly random).
    population.push((initial_genes, 0.0));
    // Fill the rest of the population with new random individuals.
    for _ in 1..POPULATION_SIZE {
        population.push((BoidGenes::new_random(), 0.0));
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
}

impl EventHandler for EvolutionManager {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        self.timer += ctx.time.delta().as_secs_f32();

        // If the timer is up, the simulation for this individual is over.
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

        // Always run the update logic for the currently active simulation.
        self.active_simulation.update(ctx)
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        // First, draw the active simulation as usual.
        self.active_simulation.draw(ctx)?;

        // Now, draw the GA's status text on top.
        let text = graphics::Text::new(format!(
            "Generation: {}\nIndividual: {}/{}\nBest Fitness (last gen): {:.2}",
            self.current_generation,
            self.current_individual_index + 1,
            POPULATION_SIZE,
            self.best_genes.cohesion_weight // Just an example, replace with a real fitness display if you want
        ));

        let mut canvas = graphics::Canvas::from_frame(ctx, None);

        // Draw the text in the top-left corner.
        canvas.draw(
            &text,
            graphics::DrawParam::new()
                .dest(Vec2::new(10.0, 10.0))
                .color(Color::WHITE),
        );

        canvas.finish(ctx)
    }
}

// --- Main Function ---
// In src/main.rs

pub fn main() -> GameResult {
    // 1. Build the context and event loop as usual.
    //    We make `ctx` mutable so we can pass a mutable reference to it.
    let (mut ctx, event_loop) = ContextBuilder::new("boids_evolution", "Gemini")
        .window_setup(WindowSetup::default().title("Boids Evolution"))
        .build()?;

    // 2. Create the state manager, passing it the context.
    let state = EvolutionManager::new(&mut ctx);

    // 3. Run the game loop with the context, event loop, and state.
    event::run(ctx, event_loop, state)
}
