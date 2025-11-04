mod boid;
mod evolution;
mod neural_evolution;
mod neural_network;
mod quadtree;
mod simulation;

use ggez::conf::WindowSetup;
use ggez::{ContextBuilder, GameResult, event};
use std::env;

use crate::evolution::{EvolutionManager, headless_main};
use crate::neural_evolution::{NeuralEvolutionManager, neural_headless_main};

pub fn main() -> GameResult {
    // Check CLI flags
    let args: Vec<String> = env::args().collect();
    let use_neural = args.iter().any(|a| a == "--neural");
    let is_headless = args.iter().any(|a| a == "--headless");

    // Parse optional parameters
    let mut generations: usize = 10;
    let mut pop_size: usize = 50;
    for i in 0..args.len() {
        if args[i] == "--generations" {
            if let Some(val) = args.get(i + 1) {
                if let Ok(n) = val.parse::<usize>() {
                    generations = n;
                }
            }
        }
        if args[i] == "--population" {
            if let Some(val) = args.get(i + 1) {
                if let Ok(n) = val.parse::<usize>() {
                    pop_size = n;
                }
            }
        }
    }

    if is_headless {
        if use_neural {
            println!(
                "Running NEURAL headless GA for {} generations (population {})",
                generations, pop_size
            );
            neural_headless_main(generations, pop_size);
        } else {
            println!(
                "Running rule-based headless GA for {} generations (population {})",
                generations, pop_size
            );
            headless_main(generations, pop_size);
        }
        return Ok(());
    }

    let (mut ctx, event_loop) = ContextBuilder::new("boids_evolution", "Gemini")
        .window_setup(WindowSetup::default().title("Boids Evolution"))
        .build()?;

    if use_neural {
        println!("Starting NEURAL network evolution with visual mode");
        let state = NeuralEvolutionManager::new(&mut ctx);
        event::run(ctx, event_loop, state)
    } else {
        println!("Starting rule-based evolution with visual mode");
        let state = EvolutionManager::new(&mut ctx);
        event::run(ctx, event_loop, state)
    }
}
