# Rust Boids Simulation

A 2D flocking simulation built from scratch in Rust with the `ggez` game engine. This project demonstrates emergent behavior by applying a few simple rules to a large number of autonomous agents, or "boids." The system is highly optimized using a Quadtree for efficient neighbor searching, allowing for a large number of boids to be simulated in real-time.

![Boids Simulation GIF](https://imgur.com/a/x6kugwQ)

---

## Features

* **Flocking Behavior:** Implements the three classic Boids rules coined by Craig Reynolds:
    * **Separation:** Boids steer to avoid crowding local flockmates.
    * **Alignment:** Boids steer towards the average heading of local flockmates.
    * **Cohesion:** Boids steer to move toward the average position of local flockmates.
* **Performance Optimization:** Uses a **Quadtree** data structure to drastically reduce the complexity of neighbor searching from O(n²) to O(n log n), allowing for a large number of boids to run smoothly.
* **Interactive Emitter:** The flock is attracted to and follows the user's mouse cursor.
* **Visual Trails:** Each boid leaves a beautiful, faint trail that visualizes the flowing, hypnotic patterns of the flock's movement.
* **Screen Wrapping:** Boids that exit one side of the screen seamlessly reappear on the opposite side.
* **Directional Rendering:** Boids are rendered as triangles that are always oriented in the direction of their velocity.

---

## Tech Stack

* **Language:** [Rust](https://www.rust-lang.org/)
* **Graphics & Game Loop:** [ggez](https://ggez.rs/) (A lightweight game framework for Rust)
* **Key Crates:**
    * `glam`: For high-performance vector mathematics.
    * `rand`: For random number generation.


