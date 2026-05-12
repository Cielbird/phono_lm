use burn::{prelude::*, record::{DefaultRecorder, Recorder}};
use getheode::phonology::{
    feature::FeatureState,
    segment::{SEG_FEATURE_COUNT, SegmentFeatures},
};

use crate::{
    data::{FEATURES, PhonoToken},
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
        Self { model, device, seed: vec![PhonoToken::WordBoundary], n_tokens }
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
            let seq_len = context.len().min(max_seq);
            let slice = &context[context.len() - seq_len..];

            let mut data = vec![0.0f32; seq_len * FEATURES];
            for (t, tok) in slice.iter().enumerate() {
                let arr: [f32; FEATURES] = tok.to_arr();
                data[t * FEATURES..(t + 1) * FEATURES].copy_from_slice(&arr);
            }

            let input = Tensor::<B, 1>::from_floats(data.as_slice(), self.device)
                .reshape([1, seq_len, FEATURES]);

            let output = self.model.forward(input); // [1, seq_len, FEATURES]

            let last = output
                .slice([0..1, (seq_len - 1)..seq_len, 0..FEATURES])
                .reshape([FEATURES]);

            let vals: Vec<f32> = last.into_data().to_vec::<f32>().unwrap();
            let arr: [f32; FEATURES] = vals.try_into().expect("wrong feature count");

            let next = snap_to_token(&arr);
            context.push(next);
            generated.push(next);
        }

        generated
    }
}

/// Decode a continuous model output into a `PhonoToken` by taking the argmax
/// of each feature triplet (POS, NEG, NA), then enforcing phonological invariants.
fn snap_to_token(output: &[f32; FEATURES]) -> PhonoToken {
    if output[0] > 0.5 {
        return PhonoToken::WordBoundary;
    }
    if output[1] > 0.5 {
        return PhonoToken::SylBoundary;
    }

    let seg_count = SEG_FEATURE_COUNT as usize;
    let feat_offset = FEATURES - seg_count * 3; // skip boundary bits and syl triplets
    let mut seg_features = [FeatureState::NA; SEG_FEATURE_COUNT as usize];
    for i in 0..seg_count {
        let base = feat_offset + i * 3;
        let (pos, neg, na) = (output[base], output[base + 1], output[base + 2]);
        seg_features[i] = if pos >= neg && pos >= na {
            FeatureState::POS
        } else if neg >= na {
            FeatureState::NEG
        } else {
            FeatureState::NA
        };
    }

    let mut s = SegmentFeatures::from_features(seg_features);
    s.enforce_invariants();
    PhonoToken::from_seg_features(s.features())
}
