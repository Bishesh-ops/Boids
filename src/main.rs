use ggez::event::{self, EventHandler};
use ggez::graphics::{self, Color};
use ggez::{Context, ContextBuilder, GameResult};

struct GameState {}

impl EventHandler for GameState {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, Color::from([0.1, 0.2, 0.3, 1.0]));

        canvas.finish(ctx)
    }
}

pub fn main() -> GameResult{
    let (mut ctx, event_loop) = ContextBuilder::new("particle_system", "Bishesh").build()?;
    let state = GameState{};

    event::run(ctx, event_loop, state)
}