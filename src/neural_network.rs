use ggez::glam::Vec2;
use rand::Rng;

/// A simple feedforward neural network for boid steering decisions.
/// Architecture: 10 inputs -> 12 hidden neurons -> 2 outputs
#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    // Weights from input (10) to hidden layer (12): 10 * 12 = 120 weights
    weights_input_hidden: Vec<f32>,
    // Biases for hidden layer: 12 biases
    biases_hidden: Vec<f32>,
    // Weights from hidden (12) to output layer (2): 12 * 2 = 24 weights
    weights_hidden_output: Vec<f32>,
    // Biases for output layer: 2 biases
    biases_output: Vec<f32>,

    // Network dimensions
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
}

impl NeuralNetwork {
    /// Creates a new neural network with random weights.
    pub fn new_random() -> Self {
        let mut rng = rand::rng();
        let input_size = 10;
        let hidden_size = 12;
        let output_size = 2;

        // Xavier initialization for better training
        let input_hidden_scale = (2.0 / input_size as f32).sqrt();
        let hidden_output_scale = (2.0 / hidden_size as f32).sqrt();

        let weights_input_hidden: Vec<f32> = (0..input_size * hidden_size)
            .map(|_| rng.random_range(-input_hidden_scale..input_hidden_scale))
            .collect();

        let biases_hidden: Vec<f32> = (0..hidden_size)
            .map(|_| rng.random_range(-0.1..0.1))
            .collect();

        let weights_hidden_output: Vec<f32> = (0..hidden_size * output_size)
            .map(|_| rng.random_range(-hidden_output_scale..hidden_output_scale))
            .collect();

        let biases_output: Vec<f32> = (0..output_size)
            .map(|_| rng.random_range(-0.1..0.1))
            .collect();

        NeuralNetwork {
            weights_input_hidden,
            biases_hidden,
            weights_hidden_output,
            biases_output,
            input_size,
            hidden_size,
            output_size,
        }
    }

    /// Creates a neural network from a flat weight vector (genome).
    pub fn from_weights(weights: &[f32]) -> Self {
        let input_size = 10;
        let hidden_size = 12;
        let output_size = 2;

        let ih_weights = input_size * hidden_size; // 120
        let h_biases = hidden_size; // 12
        let ho_weights = hidden_size * output_size; // 24
        let o_biases = output_size; // 2

        let expected_size = ih_weights + h_biases + ho_weights + o_biases; // 158

        assert_eq!(
            weights.len(),
            expected_size,
            "Weight vector must be exactly {} elements",
            expected_size
        );

        let mut idx = 0;

        let weights_input_hidden = weights[idx..idx + ih_weights].to_vec();
        idx += ih_weights;

        let biases_hidden = weights[idx..idx + h_biases].to_vec();
        idx += h_biases;

        let weights_hidden_output = weights[idx..idx + ho_weights].to_vec();
        idx += ho_weights;

        let biases_output = weights[idx..idx + o_biases].to_vec();

        NeuralNetwork {
            weights_input_hidden,
            biases_hidden,
            weights_hidden_output,
            biases_output,
            input_size,
            hidden_size,
            output_size,
        }
    }

    /// Converts the network weights to a flat vector (genome) for evolution.
    pub fn to_weights(&self) -> Vec<f32> {
        let mut weights = Vec::new();
        weights.extend_from_slice(&self.weights_input_hidden);
        weights.extend_from_slice(&self.biases_hidden);
        weights.extend_from_slice(&self.weights_hidden_output);
        weights.extend_from_slice(&self.biases_output);
        weights
    }

    /// Returns the expected size of the weight vector.
    pub fn weight_count() -> usize {
        let input_size = 10;
        let hidden_size = 12;
        let output_size = 2;
        input_size * hidden_size + hidden_size + hidden_size * output_size + output_size
    }

    /// Forward propagation: input -> hidden -> output
    /// Returns a Vec2 representing the steering force.
    pub fn forward(&self, inputs: &[f32]) -> Vec2 {
        assert_eq!(
            inputs.len(),
            self.input_size,
            "Input size must be {}",
            self.input_size
        );

        // Hidden layer computation
        let mut hidden = vec![0.0; self.hidden_size];
        for h in 0..self.hidden_size {
            let mut sum = self.biases_hidden[h];
            for i in 0..self.input_size {
                sum += inputs[i] * self.weights_input_hidden[i * self.hidden_size + h];
            }
            // Tanh activation for hidden layer
            hidden[h] = sum.tanh();
        }

        // Output layer computation
        let mut output = vec![0.0; self.output_size];
        for o in 0..self.output_size {
            let mut sum = self.biases_output[o];
            for h in 0..self.hidden_size {
                sum += hidden[h] * self.weights_hidden_output[h * self.output_size + o];
            }
            // Tanh activation for output layer (produces values in [-1, 1])
            output[o] = sum.tanh();
        }

        Vec2::new(output[0], output[1])
    }

    /// Prepare input vector from neighbor data for the neural network.
    /// Input features:
    /// 0-1: average separation direction (normalized)
    /// 2-3: average alignment direction (normalized)
    /// 4-5: average cohesion direction (normalized)
    /// 6-7: current velocity (normalized)
    /// 8: neighbor count (normalized to [0, 1])
    /// 9: average distance to neighbors (normalized)
    pub fn prepare_inputs(
        separation: Vec2,
        alignment: Vec2,
        cohesion: Vec2,
        velocity: Vec2,
        neighbor_count: f32,
        avg_distance: f32,
        _max_speed: f32,
    ) -> Vec<f32> {
        // Normalize vectors
        let sep_norm = separation.normalize_or_zero();
        let ali_norm = alignment.normalize_or_zero();
        let coh_norm = cohesion.normalize_or_zero();
        let vel_norm = velocity.normalize_or_zero();

        // Normalize neighbor count (assume max ~50 neighbors)
        let neighbor_norm = (neighbor_count / 50.0).min(1.0);

        // Normalize distance (assume max perception is ~150)
        let distance_norm = (avg_distance / 150.0).min(1.0);

        vec![
            sep_norm.x,
            sep_norm.y,
            ali_norm.x,
            ali_norm.y,
            coh_norm.x,
            coh_norm.y,
            vel_norm.x,
            vel_norm.y,
            neighbor_norm,
            distance_norm,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_count() {
        assert_eq!(NeuralNetwork::weight_count(), 158);
    }

    #[test]
    fn test_forward_propagation() {
        let nn = NeuralNetwork::new_random();
        let inputs = vec![0.5; 10];
        let output = nn.forward(&inputs);

        // Output should be in range [-1, 1] due to tanh activation
        assert!(output.x >= -1.0 && output.x <= 1.0);
        assert!(output.y >= -1.0 && output.y <= 1.0);
    }

    #[test]
    fn test_weight_serialization() {
        let nn1 = NeuralNetwork::new_random();
        let weights = nn1.to_weights();
        let nn2 = NeuralNetwork::from_weights(&weights);

        assert_eq!(nn1.to_weights(), nn2.to_weights());
    }
}
