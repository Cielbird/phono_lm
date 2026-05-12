use burn::{
    prelude::*,
    record::{DefaultRecorder, Recorder},
};

use crate::{
    data::{PHONO_LOGITS, PhonoGenerationBatch, PhonoToken},
    model::{PhonoGenerationModel, PhonoGenerationModelConfig},
    training::ExperimentConfig,
};

/// Load a trained model from `artifact_dir` (the same path passed to `train`).
pub fn load_model<B: Backend>(artifact_dir: &str, device: &B::Device) -> PhonoGenerationModel<B> {
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json"))
        .expect("config.json not found");
    let model_config =
        PhonoGenerationModelConfig::new(config.transformer.clone(), config.max_seq_length);
    let record = DefaultRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("model record not found");
    model_config.init(device).load_record(record)
}

/// Autoregressively generates phonological tokens from a trained model.
pub struct PhonologicalGenerate<'a, B: Backend> {
    model: &'a PhonoGenerationModel<B>,
    device: &'a B::Device,
    seed: Vec<PhonoToken>,
    n_tokens: usize,
}

impl<'a, B: Backend> PhonologicalGenerate<'a, B> {
    pub fn new(model: &'a PhonoGenerationModel<B>, device: &'a B::Device, n_tokens: usize) -> Self {
        Self {
            model,
            device,
            seed: vec![PhonoToken::WordBoundary],
            n_tokens,
        }
    }

    /// Set a conditioning IPA seed sequence.
    pub fn with_seed(mut self, seed: Vec<PhonoToken>) -> Self {
        if !seed.is_empty() {
            self.seed = seed;
        }
        self
    }

    /// Run generation and return the newly produced tokens (seed not included).
    pub fn run(self) -> Vec<PhonoToken> {
        let mut context = self.seed;
        let mut generated = Vec::with_capacity(self.n_tokens);
        let max_seq = self.model.max_seq_length();

        for _ in 0..self.n_tokens {
            let seq_len = context.len().min(max_seq - 1);
            let slice = &context[context.len() - seq_len..];

            let mut data = vec![0.; seq_len * PHONO_LOGITS];
            let mut mask_pad = vec![true; seq_len];
            for (t, tok) in slice.iter().enumerate() {
                let arr = tok.to_probs();
                data[t * PHONO_LOGITS..(t + 1) * PHONO_LOGITS].copy_from_slice(&arr);
                mask_pad[t] = false;
            }

            let tokens = Tensor::<B, 1>::from_floats(data.as_slice(), self.device).reshape([
                1,
                seq_len,
                PHONO_LOGITS,
            ]);
            let mask_pad = Tensor::<B, 1, Bool>::from_bool(mask_pad.as_slice(), self.device)
                .reshape([1, seq_len]);
            let input = PhonoGenerationBatch::new(tokens, mask_pad);

            let output = self.model.infer(input); // [1, seq_len, TOKEN_CLASSES+TOKEN_FEATURES]

            let last = output
                .slice([0..1, (seq_len - 1)..seq_len, 0..PHONO_LOGITS])
                .reshape([PHONO_LOGITS]);

            let vals: Vec<f32> = last.into_data().to_vec::<f32>().unwrap();
            let arr: [f32; PHONO_LOGITS] = vals.try_into().expect("wrong feature count");

            let next = PhonoToken::from_probs(arr);
            context.push(next.clone());
            generated.push(next);
        }

        generated
    }
}
