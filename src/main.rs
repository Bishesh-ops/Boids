mod boid;
mod evolution;
mod quadtree;
mod simulation;

use ggez::conf::WindowSetup;
use ggez::{ContextBuilder, GameResult, event};
use std::env;

use crate::evolution::{EvolutionManager, headless_main};

pub fn main() -> GameResult {
    // Check CLI flags for headless mode
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--headless") {
        // Optional: parse generations and population overrides
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

        println!(
            "Running headless GA for {} generations (population {})",
            generations, pop_size
        );
        headless_main(generations, pop_size);
        return Ok(());
    }

    let (mut ctx, event_loop) = ContextBuilder::new("boids_evolution", "Gemini")
        .window_setup(WindowSetup::default().title("Boids Evolution"))
        .build()?;

    let state = EvolutionManager::new(&mut ctx);

    event::run(ctx, event_loop, state)
}
