use ggez::conf::WindowSetup;
use ggez::event::{self, EventHandler};
use ggez::glam::Vec2;
use ggez::graphics::{self, Color, DrawMode};
use ggez::{Context, ContextBuilder, GameResult};
use rand::{Rng, thread_rng};

const PERCEPTION_RADIUS: f32 = 75.0;
const MAX_SPEED: f32 = 200.0;
const MAX_FORCE: f32 = 150.0;

const SEPARATION_WEIGHT: f32 = 1.5;
const ALIGNMENT_WEIGHT: f32 = 1.0;
const COHESION_WEIGHT: f32 = 1.0;
const TRAIL_LIFESPAN: f32 = 1.0;
#[derive(Debug, Clone, Copy)]
struct Point {
    position: Vec2,
    velocity: Vec2,
}

struct TrailPoint {
    position: Vec2,
    lifespan: f32,
}

#[derive(Debug, Clone, Copy)]
struct Rectangle {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}
struct Quadtree {
    boundary: Rectangle,
    capacity: usize,
    points: Vec<Point>,
    divided: bool,
    // `Box` allocates the children on the heap, preventing infinite size issues.
    northwest: Option<Box<Quadtree>>,
    northeast: Option<Box<Quadtree>>,
    southwest: Option<Box<Quadtree>>,
    southeast: Option<Box<Quadtree>>,
}
impl Rectangle {
    fn contains(&self, point: &Point) -> bool {
        (point.position.x >= self.x - self.w)
            && (point.position.x < self.x + self.w)
            && (point.position.y >= self.y - self.h)
            && (point.position.y < self.y + self.h)
    }
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
    fn query(&self, range: &Rectangle, found: &mut Vec<Point>) {
        if !self.boundary.intersects(range) {
            return;
        } else {
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
}

#[derive(Clone)]
struct Boid {
    position: Vec2,
    velocity: Vec2,
    acceleration: Vec2,
}

impl Boid {
    fn new(screen_width: f32, screen_height: f32) -> Self {
        let mut rng = rand::thread_rng();
        Boid {
            position: Vec2::new(
                rng.gen_range(0.0..screen_width),
                rng.gen_range(0.0..screen_height),
            ),
            velocity: Vec2::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)).normalize()
                * 100.0,
            acceleration: Vec2::ZERO,
        }
    }
}

struct GameState {
    boids: Vec<Boid>,
    trails: Vec<TrailPoint>,
}

impl GameState {
    fn new(ctx: &mut Context) -> Self {
        let (width, height) = ctx.gfx.drawable_size();
        let mut boids = Vec::new();
        for _ in 0..150 {
            boids.push(Boid::new(width, height));
        }
        GameState {
            boids,
            trails: Vec::new(),
        }
    }
}

impl EventHandler for GameState {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        let dt = ctx.time.delta().as_secs_f32();
        let (width, height) = ctx.gfx.drawable_size();

        // This loop adds a new trail point at each boid's current position.
        for boid in &self.boids {
            self.trails.push(TrailPoint {
                position: boid.position,
                lifespan: TRAIL_LIFESPAN,
            });
        }

        // Decrease lifespan for each point.
        for trail in &mut self.trails {
            trail.lifespan -= dt;
        }
        // Remove points whose lifespan has run out.
        self.trails.retain(|trail| trail.lifespan > 0.0);

        // --- Boids logic (this is the same as before) ---
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
                w: PERCEPTION_RADIUS,
                h: PERCEPTION_RADIUS,
            };
            let mut potential_neighbors = Vec::new();
            qtree.query(&range, &mut potential_neighbors);
            let neighbors: Vec<Point> = potential_neighbors
                .into_iter()
                .filter(|p| boid.position.distance(p.position) > 0.0001)
                .collect();

            // ... all the separation, alignment, cohesion logic remains here ...
            let mut separation = Vec2::ZERO;
            let mut alignment = Vec2::ZERO;
            let mut cohesion = Vec2::ZERO;
            if !neighbors.is_empty() {
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
                alignment = alignment.normalize_or_zero() * MAX_SPEED;
                alignment -= boid.velocity;
                alignment = alignment.clamp_length_max(MAX_FORCE);
                cohesion /= neighbor_count;
                cohesion -= boid.position;
                cohesion = cohesion.normalize_or_zero() * MAX_SPEED;
                cohesion -= boid.velocity;
                cohesion = cohesion.clamp_length_max(MAX_FORCE);
                separation /= neighbor_count;
                if separation.length() > 0.0 {
                    separation = separation.normalize_or_zero() * MAX_SPEED;
                    separation -= boid.velocity;
                    separation = separation.clamp_length_max(MAX_FORCE);
                }
                boid.acceleration += separation * SEPARATION_WEIGHT;
                boid.acceleration += alignment * ALIGNMENT_WEIGHT;
                boid.acceleration += cohesion * COHESION_WEIGHT;
            }
            // ... the physics update logic remains here ...
            boid.velocity += boid.acceleration * dt;
            boid.velocity = boid.velocity.clamp_length_max(MAX_SPEED);
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

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, Color::BLACK);

        // --- Draw the trails ---
        let trail_mesh =
            graphics::Mesh::new_circle(ctx, DrawMode::fill(), Vec2::ZERO, 2.0, 0.1, Color::WHITE)?;
        for trail in &self.trails {
            // Calculate alpha (transparency) based on remaining lifespan
            let alpha = trail.lifespan / TRAIL_LIFESPAN;
            let color = Color::new(1.0, 1.0, 1.0, alpha * 0.5); // Make it faint
            canvas.draw(
                &trail_mesh,
                graphics::DrawParam::new().dest(trail.position).color(color),
            );
        }

        // --- Draw the boids (same as before) ---
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
}

pub fn main() -> GameResult {
    let (mut ctx, event_loop) = ContextBuilder::new("Boids", "Bishesh")
        .window_setup(WindowSetup::default().title("Boid Simulator"))
        .build()?;
    let state = GameState::new(&mut ctx);

    event::run(ctx, event_loop, state)
}
