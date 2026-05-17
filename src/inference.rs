use burn::{
    prelude::*,
    record::{DefaultRecorder, Recorder},
};

use crate::{
    data::{PAD_TOKEN, PhonoGenerationBatch, PhonoToken, PhonoVocab},
    model::{PhonoGenerationModel, PhonoGenerationModelConfig},
    training::ExperimentConfig,
};

pub fn load_model(artifact_dir: &str, device: &Device) -> (PhonoGenerationModel, PhonoVocab) {
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json"))
        .expect("config.json not found");
    let vocab =
        PhonoVocab::load(&format!("{artifact_dir}/vocab.json")).expect("vocab.json not found");
    let model_config = PhonoGenerationModelConfig::new(
        config.transformer.clone(),
        vocab.vocab_size(),
        PAD_TOKEN,
        config.max_seq_length,
    );
    let record = DefaultRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("model record not found");
    let model = model_config.init(device).load_record(record);
    (model, vocab)
}

pub struct PhonologicalGenerate<'a> {
    model: &'a PhonoGenerationModel,
    vocab: &'a PhonoVocab,
    device: &'a Device,
    seed: Vec<PhonoToken>,
    n_tokens: usize,
}

impl<'a> PhonologicalGenerate<'a> {
    pub fn new(
        model: &'a PhonoGenerationModel,
        vocab: &'a PhonoVocab,
        device: &'a Device,
        n_tokens: usize,
    ) -> Self {
        Self {
            model,
            vocab,
            device,
            seed: vec![PhonoToken::WordBoundary],
            n_tokens,
        }
    }

    pub fn with_seed(mut self, seed: Vec<PhonoToken>) -> Self {
        if !seed.is_empty() {
            self.seed = seed;
        }
        self
    }

    pub fn run(self) -> Vec<PhonoToken> {
        let mut context = self.seed;
        let mut generated = Vec::with_capacity(self.n_tokens);
        let max_seq = self.model.max_seq_length();

        for _ in 0..self.n_tokens {
            let seq_len = context.len().min(max_seq - 1);
            let slice = &context[context.len() - seq_len..];

            let token_ids: Vec<i32> = slice.iter().map(|t| self.vocab.to_id(t) as i32).collect();
            let mask_pad: Vec<bool> = vec![false; seq_len];

            let tokens = Tensor::<1, Int>::from_ints(token_ids.as_slice(), self.device)
                .reshape([1, seq_len]);
            let mask_pad = Tensor::<1, Bool>::from_bool(mask_pad.as_slice(), self.device)
                .reshape([1, seq_len]);

            let probs = self
                .model
                .infer(PhonoGenerationBatch::new(tokens, mask_pad));
            let last_probs = probs
                .slice([0..1, (seq_len - 1)..seq_len])
                .squeeze::<1>(); // [VOCAB_SIZE]

            let next_id = last_probs.argmax(0).into_scalar() as usize;
            let next = if next_id == PAD_TOKEN {
                PhonoToken::WordBoundary
            } else {
                self.vocab.from_id(next_id)
            };

            context.push(next.clone());
            generated.push(next);
        }

        generated
    }
}
