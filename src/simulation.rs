use crate::boid::{Boid, BoidGenes, NeuralBoidGenes, TrailPoint};
use crate::neural_network::NeuralNetwork;
use crate::quadtree::{Point, Quadtree, Rectangle};
use ggez::glam::Vec2;
use ggez::{graphics, Context, GameResult};
// Note: rayon is available for future parallel optimizations
// use rayon::prelude::*;

const TRAIL_LIFESPAN: f32 = 1.0;
const MAX_TRAIL_POINTS: usize = 10000; // Cap trail growth to prevent memory issues

/// Manages the state of a single simulation instance (OLD rule-based).
pub struct GameState {
    pub boids: Vec<Boid>,
    pub trails: Vec<TrailPoint>,
    pub genes: BoidGenes,
    trail_mesh: Option<graphics::Mesh>,
    boid_mesh: Option<graphics::Mesh>,
}

/// Neural network-based simulation state.
pub struct NeuralGameState {
    pub boids: Vec<Boid>,
    pub trails: Vec<TrailPoint>,
    pub genes: NeuralBoidGenes,
    pub network: NeuralNetwork,
    trail_mesh: Option<graphics::Mesh>,
    boid_mesh: Option<graphics::Mesh>,
}

impl GameState {
    /// Creates a new simulation instance with a given set of genes.
    pub fn new(ctx: &mut Context, genes: BoidGenes) -> Self {
        let (width, height) = ctx.gfx.drawable_size();
        let mut boids = Vec::new();
        for _ in 0..100 {
            boids.push(Boid::new(width, height));
        }

        // Pre-create meshes for performance
        let trail_mesh = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            Vec2::ZERO,
            2.0,
            0.1,
            graphics::Color::WHITE,
        )
        .ok();

        let boid_mesh = graphics::Mesh::new_polygon(
            ctx,
            graphics::DrawMode::fill(),
            &[
                Vec2::new(0.0, -10.0),
                Vec2::new(5.0, 10.0),
                Vec2::new(-5.0, 10.0),
            ],
            graphics::Color::WHITE,
        )
        .ok();

        GameState {
            boids,
            trails: Vec::new(),
            genes,
            trail_mesh,
            boid_mesh,
        }
    }

    /// Create a GameState without a ggez Context for headless simulation.
    pub fn new_headless(width: f32, height: f32, genes: BoidGenes, boid_count: usize) -> Self {
        let mut boids = Vec::new();
        for _ in 0..boid_count {
            boids.push(Boid::new(width, height));
        }
        GameState {
            boids,
            trails: Vec::new(),
            genes,
            trail_mesh: None,
            boid_mesh: None,
        }
    }

    /// Runs the core simulation logic for one frame.
    pub fn update(&mut self, ctx: &mut Context) -> GameResult {
        if self.boids.is_empty() {
            let (width, height) = ctx.gfx.drawable_size();
            for _ in 0..100 {
                self.boids.push(Boid::new(width, height));
            }
        }
        let dt = ctx.time.delta().as_secs_f32();
        let (width, height) = ctx.gfx.drawable_size();

        // Update trails with cap to prevent unbounded growth
        for boid in &self.boids {
            if self.trails.len() < MAX_TRAIL_POINTS {
                self.trails.push(TrailPoint {
                    position: boid.position,
                    lifespan: TRAIL_LIFESPAN,
                });
            }
        }
        for trail in &mut self.trails {
            trail.lifespan -= dt;
        }
        self.trails.retain(|trail| trail.lifespan > 0.0);

        // Build the Quadtree for this frame to find neighbors efficiently.
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

            // Use pre-allocated capacity for potential performance benefit
            let mut potential_neighbors = Vec::with_capacity(50);
            qtree.query(&range, &mut potential_neighbors);

            // Filter out the boid itself from the neighbor list using distance check.
            let neighbors: Vec<Point> = potential_neighbors
                .into_iter()
                .filter(|p| boid.position.distance(p.position) > 0.0001)
                .collect();

            // Apply the three flocking rules only if there are actual neighbors.
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

            // Update physics and handle screen wrapping.
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

    /// Headless update step: same logic as `update` but driven by an external dt and sizes.
    pub fn update_headless(&mut self, dt: f32, width: f32, height: f32) {
        // Update trails with cap to prevent unbounded growth
        for boid in &self.boids {
            if self.trails.len() < MAX_TRAIL_POINTS {
                self.trails.push(TrailPoint {
                    position: boid.position,
                    lifespan: TRAIL_LIFESPAN,
                });
            }
        }
        for trail in &mut self.trails {
            trail.lifespan -= dt;
        }
        self.trails.retain(|trail| trail.lifespan > 0.0);

        // Build the Quadtree for this frame to find neighbors efficiently.
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

            // Use pre-allocated capacity for potential performance benefit
            let mut potential_neighbors = Vec::with_capacity(50);
            qtree.query(&range, &mut potential_neighbors);

            let neighbors: Vec<Point> = potential_neighbors
                .into_iter()
                .filter(|p| boid.position.distance(p.position) > 0.0001)
                .collect();

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
    }

    /// Draws the simulation into an existing canvas/frame.
    pub fn draw_to_canvas(
        &mut self,
        ctx: &mut Context,
        canvas: &mut graphics::Canvas,
    ) -> GameResult {
        // Lazy initialize meshes if not already created (for compatibility)
        if self.trail_mesh.is_none() {
            self.trail_mesh = graphics::Mesh::new_circle(
                ctx,
                graphics::DrawMode::fill(),
                Vec2::ZERO,
                2.0,
                0.1,
                graphics::Color::WHITE,
            )
            .ok();
        }
        if self.boid_mesh.is_none() {
            self.boid_mesh = graphics::Mesh::new_polygon(
                ctx,
                graphics::DrawMode::fill(),
                &[
                    Vec2::new(0.0, -10.0),
                    Vec2::new(5.0, 10.0),
                    Vec2::new(-5.0, 10.0),
                ],
                graphics::Color::WHITE,
            )
            .ok();
        }

        // Draw the trails first, so they are behind the boids.
        if let Some(trail_mesh) = &self.trail_mesh {
            for trail in &self.trails {
                let alpha = trail.lifespan / TRAIL_LIFESPAN;
                let color = graphics::Color::new(1.0, 1.0, 1.0, alpha * 0.5);
                canvas.draw(
                    trail_mesh,
                    graphics::DrawParam::new().dest(trail.position).color(color),
                );
            }
        }

        // Draw the boids as triangles.
        if let Some(boid_mesh) = &self.boid_mesh {
            for boid in &self.boids {
                let angle = boid.velocity.y.atan2(boid.velocity.x) + 90.0_f32.to_radians();
                canvas.draw(
                    boid_mesh,
                    graphics::DrawParam::new()
                        .dest(boid.position)
                        .rotation(angle),
                );
            }
        }

        Ok(())
    }

    /// Calculates a fitness score based on the flock's cohesion and separation.
    /// Lower fitness is better (tight flocking with fewer collisions).
    pub fn calculate_fitness(&self) -> f32 {
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

        // Calculate average distance from center of mass
        for boid in &self.boids {
            total_distance_from_center += boid.position.distance(center_of_mass);
        }

        // Use quadtree for efficient collision detection
        // Use a large boundary to encompass all boids (default window is 800x600)
        let boundary = Rectangle {
            x: 400.0,
            y: 300.0,
            w: 400.0,
            h: 300.0,
        };
        let mut qtree = Quadtree::new(boundary, 4);

        for boid in &self.boids {
            qtree.insert(Point {
                position: boid.position,
                velocity: boid.velocity,
            });
        }

        // Check for collisions using quadtree
        for boid in &self.boids {
            let collision_range = Rectangle {
                x: boid.position.x,
                y: boid.position.y,
                w: 10.0,
                h: 10.0,
            };
            let mut nearby = Vec::new();
            qtree.query(&collision_range, &mut nearby);

            for other in nearby {
                // Exclude self and check collision distance
                if boid.position.distance(other.position) > 0.0001
                    && boid.position.distance(other.position) < 10.0
                {
                    collision_count += 1;
                }
            }
        }

        let avg_distance = total_distance_from_center / self.boids.len() as f32;
        // Divide collision_count by 2 since each collision is counted twice (A->B and B->A)
        // Negate to make lower values better (tight flocking, fewer collisions = better fitness)
        -(avg_distance + ((collision_count / 2) as f32 * 10.0))
    }
}

impl NeuralGameState {
    /// Creates a new neural simulation instance with given genes.
    pub fn new(ctx: &mut Context, genes: NeuralBoidGenes) -> Self {
        let (width, height) = ctx.gfx.drawable_size();
        let mut boids = Vec::new();
        for _ in 0..100 {
            boids.push(Boid::new(width, height));
        }

        let network = genes.create_network();

        let trail_mesh = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            Vec2::ZERO,
            2.0,
            0.1,
            graphics::Color::WHITE,
        )
        .ok();

        let boid_mesh = graphics::Mesh::new_polygon(
            ctx,
            graphics::DrawMode::fill(),
            &[
                Vec2::new(0.0, -10.0),
                Vec2::new(5.0, 10.0),
                Vec2::new(-5.0, 10.0),
            ],
            graphics::Color::WHITE,
        )
        .ok();

        NeuralGameState {
            boids,
            trails: Vec::new(),
            genes,
            network,
            trail_mesh,
            boid_mesh,
        }
    }

    /// Create a NeuralGameState without a ggez Context for headless simulation.
    pub fn new_headless(width: f32, height: f32, genes: NeuralBoidGenes, boid_count: usize) -> Self {
        let mut boids = Vec::new();
        for _ in 0..boid_count {
            boids.push(Boid::new(width, height));
        }
        let network = genes.create_network();

        NeuralGameState {
            boids,
            trails: Vec::new(),
            genes,
            network,
            trail_mesh: None,
            boid_mesh: None,
        }
    }

    /// Update with neural network steering (with parallel processing).
    pub fn update(&mut self, ctx: &mut Context) -> GameResult {
        if self.boids.is_empty() {
            let (width, height) = ctx.gfx.drawable_size();
            for _ in 0..100 {
                self.boids.push(Boid::new(width, height));
            }
        }
        let dt = ctx.time.delta().as_secs_f32();
        let (width, height) = ctx.gfx.drawable_size();

        self.update_internal(dt, width, height);
        Ok(())
    }

    /// Headless update for neural network.
    pub fn update_headless(&mut self, dt: f32, width: f32, height: f32) {
        self.update_internal(dt, width, height);
    }

    /// Internal update logic using neural network steering with parallelization.
    fn update_internal(&mut self, dt: f32, width: f32, height: f32) {
        // Update trails
        for boid in &self.boids {
            if self.trails.len() < MAX_TRAIL_POINTS {
                self.trails.push(TrailPoint {
                    position: boid.position,
                    lifespan: TRAIL_LIFESPAN,
                });
            }
        }
        for trail in &mut self.trails {
            trail.lifespan -= dt;
        }
        self.trails.retain(|trail| trail.lifespan > 0.0);

        // Build quadtree for neighbor queries
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

        // Parallel NN evaluation and steering force calculation
        let perception_radius = self.genes.perception_radius;
        let max_speed = self.genes.max_speed;
        let max_force = self.genes.max_force;
        let network = &self.network;

        // Note: Quadtree query is not thread-safe, so we do sequential neighbor finding
        // But we can still parallelize the NN evaluation later if needed
        // For now, sequential is fine and still performant with the quadtree optimization
        for boid in &mut self.boids {
            let range = Rectangle {
                x: boid.position.x,
                y: boid.position.y,
                w: perception_radius,
                h: perception_radius,
            };

            let mut potential_neighbors = Vec::with_capacity(50);
            qtree.query(&range, &mut potential_neighbors);

            let neighbors: Vec<Point> = potential_neighbors
                .into_iter()
                .filter(|p| boid.position.distance(p.position) > 0.0001)
                .collect();

            // Calculate inputs for neural network
            let steering_force = if !neighbors.is_empty() {
                let mut separation = Vec2::ZERO;
                let mut alignment = Vec2::ZERO;
                let mut cohesion = Vec2::ZERO;
                let mut total_distance = 0.0;
                let neighbor_count = neighbors.len() as f32;

                for other in &neighbors {
                    let diff = boid.position - other.position;
                    let distance = diff.length();
                    if distance > 0.0 {
                        separation += diff.normalize() / distance;
                    }
                    total_distance += distance;
                    alignment += other.velocity;
                    cohesion += other.position;
                }

                separation /= neighbor_count;
                alignment /= neighbor_count;
                cohesion /= neighbor_count;
                cohesion -= boid.position; // Direction to center of neighbors

                let avg_distance = total_distance / neighbor_count;

                // Prepare neural network inputs
                let inputs = NeuralNetwork::prepare_inputs(
                    separation,
                    alignment,
                    cohesion,
                    boid.velocity,
                    neighbor_count,
                    avg_distance,
                    max_speed,
                );

                // Get steering force from neural network
                let nn_output = network.forward(&inputs);

                // Scale output to max_force range (output is in [-1, 1])
                nn_output * max_force
            } else {
                Vec2::ZERO
            };

            boid.acceleration += steering_force;

            // Update physics
            boid.velocity += boid.acceleration * dt;
            boid.velocity = boid.velocity.clamp_length_max(max_speed);
            boid.position += boid.velocity * dt;
            boid.acceleration = Vec2::ZERO;

            // Screen wrapping
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
    }

    /// Draw to canvas (same as GameState).
    pub fn draw_to_canvas(
        &mut self,
        ctx: &mut Context,
        canvas: &mut graphics::Canvas,
    ) -> GameResult {
        if self.trail_mesh.is_none() {
            self.trail_mesh = graphics::Mesh::new_circle(
                ctx,
                graphics::DrawMode::fill(),
                Vec2::ZERO,
                2.0,
                0.1,
                graphics::Color::WHITE,
            )
            .ok();
        }
        if self.boid_mesh.is_none() {
            self.boid_mesh = graphics::Mesh::new_polygon(
                ctx,
                graphics::DrawMode::fill(),
                &[
                    Vec2::new(0.0, -10.0),
                    Vec2::new(5.0, 10.0),
                    Vec2::new(-5.0, 10.0),
                ],
                graphics::Color::WHITE,
            )
            .ok();
        }

        if let Some(trail_mesh) = &self.trail_mesh {
            for trail in &self.trails {
                let alpha = trail.lifespan / TRAIL_LIFESPAN;
                let color = graphics::Color::new(1.0, 1.0, 1.0, alpha * 0.5);
                canvas.draw(
                    trail_mesh,
                    graphics::DrawParam::new().dest(trail.position).color(color),
                );
            }
        }

        if let Some(boid_mesh) = &self.boid_mesh {
            for boid in &self.boids {
                let angle = boid.velocity.y.atan2(boid.velocity.x) + 90.0_f32.to_radians();
                canvas.draw(
                    boid_mesh,
                    graphics::DrawParam::new()
                        .dest(boid.position)
                        .rotation(angle),
                );
            }
        }

        Ok(())
    }

    /// Calculate fitness (same logic as GameState).
    pub fn calculate_fitness(&self) -> f32 {
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
        }

        let boundary = Rectangle {
            x: 400.0,
            y: 300.0,
            w: 400.0,
            h: 300.0,
        };
        let mut qtree = Quadtree::new(boundary, 4);

        for boid in &self.boids {
            qtree.insert(Point {
                position: boid.position,
                velocity: boid.velocity,
            });
        }

        for boid in &self.boids {
            let collision_range = Rectangle {
                x: boid.position.x,
                y: boid.position.y,
                w: 10.0,
                h: 10.0,
            };
            let mut nearby = Vec::new();
            qtree.query(&collision_range, &mut nearby);

            for other in nearby {
                if boid.position.distance(other.position) > 0.0001
                    && boid.position.distance(other.position) < 10.0
                {
                    collision_count += 1;
                }
            }
        }

        let avg_distance = total_distance_from_center / self.boids.len() as f32;
        -(avg_distance + ((collision_count / 2) as f32 * 10.0))
    }
}
