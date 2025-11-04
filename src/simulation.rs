use crate::boid::{Boid, BoidGenes, TrailPoint};
use crate::quadtree::{Point, Quadtree, Rectangle};
use ggez::glam::Vec2;
use ggez::{Context, GameResult, graphics};

const TRAIL_LIFESPAN: f32 = 1.0;

/// Manages the state of a single simulation instance.
pub struct GameState {
    pub boids: Vec<Boid>,
    pub trails: Vec<TrailPoint>,
    pub genes: BoidGenes,
}

impl GameState {
    /// Creates a new simulation instance with a given set of genes.
    pub fn new(ctx: &mut Context, genes: BoidGenes) -> Self {
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

            let mut potential_neighbors = Vec::new();
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

            let mut potential_neighbors = Vec::new();
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
        // Draw the trails first, so they are behind the boids.
        let trail_mesh = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            Vec2::ZERO,
            2.0,
            0.1,
            graphics::Color::WHITE,
        )?;
        for trail in &self.trails {
            let alpha = trail.lifespan / TRAIL_LIFESPAN;
            let color = graphics::Color::new(1.0, 1.0, 1.0, alpha * 0.5);
            canvas.draw(
                &trail_mesh,
                graphics::DrawParam::new().dest(trail.position).color(color),
            );
        }

        // Draw the boids as triangles.
        let boid_mesh = graphics::Mesh::new_polygon(
            ctx,
            graphics::DrawMode::fill(),
            &[
                Vec2::new(0.0, -10.0),
                Vec2::new(5.0, 10.0),
                Vec2::new(-5.0, 10.0),
            ],
            graphics::Color::WHITE,
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

        Ok(())
    }

    /// Calculates a fitness score based on the flock's cohesion and separation.
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
            for other in &self.boids {
                if boid.position != other.position && boid.position.distance(other.position) < 10.0
                {
                    collision_count += 1;
                }
            }
        }

        let avg_distance = total_distance_from_center / self.boids.len() as f32;
        avg_distance + (collision_count as f32 * 10.0)
    }
}
