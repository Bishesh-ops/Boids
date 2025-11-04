use ggez::glam::Vec2;
use rand::Rng;
use crate::neural_network::NeuralNetwork;

/// Represents the "DNA" of a flock, holding all tunable parameters.
/// This is the OLD rule-based genes, kept for backward compatibility.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct BoidGenes {
    pub perception_radius: f32,
    pub max_speed: f32,
    pub max_force: f32,
    pub separation_weight: f32,
    pub alignment_weight: f32,
    pub cohesion_weight: f32,
}

/// Neural network-based genes for boid behavior.
/// Contains the neural network weights and physical parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NeuralBoidGenes {
    /// The flattened neural network weights (158 values)
    pub network_weights: Vec<f32>,
    /// Maximum speed the boid can move
    pub max_speed: f32,
    /// Maximum steering force that can be applied
    pub max_force: f32,
    /// How far the boid can perceive neighbors
    pub perception_radius: f32,
}

impl BoidGenes {
    /// Creates a new set of genes with random values within reasonable ranges.
    pub fn new_random() -> Self {
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

impl NeuralBoidGenes {
    /// Creates a new set of neural genes with random values.
    pub fn new_random() -> Self {
        let mut rng = rand::rng();
        let nn = NeuralNetwork::new_random();

        NeuralBoidGenes {
            network_weights: nn.to_weights(),
            max_speed: rng.random_range(100.0..300.0),
            max_force: rng.random_range(100.0..400.0),
            perception_radius: rng.random_range(50.0..150.0),
        }
    }

    /// Creates a neural network from these genes.
    pub fn create_network(&self) -> NeuralNetwork {
        NeuralNetwork::from_weights(&self.network_weights)
    }
}

/// A single point in a boid's trail.
pub struct TrailPoint {
    pub position: Vec2,
    pub lifespan: f32,
}

/// An autonomous agent in the flock.
#[derive(Clone)]
pub struct Boid {
    pub position: Vec2,
    pub velocity: Vec2,
    pub acceleration: Vec2,
}

impl Boid {
    pub fn new(screen_width: f32, screen_height: f32) -> Self {
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
