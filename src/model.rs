use crate::data::{FEATURES, TrainingPhonoGenerationBatch};
use burn::{
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        attention::generate_autoregressive_mask,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::{activation::sigmoid, backend::AutodiffBackend},
    train::{InferenceStep, RegressionOutput, TrainOutput, TrainStep},
};

#[derive(Config, Debug)]
pub struct PhonoGenerationModelConfig {
    transformer: TransformerEncoderConfig,
    #[config(default = "FEATURES")]
    n_features: usize,
    max_seq_length: usize,
}

#[derive(Module, Debug)]
pub struct PhonoGenerationModel<B: Backend> {
    transformer: TransformerEncoder<B>,
    input_proj: Linear<B>,
    embedding_pos: Embedding<B>,
    output_proj: Linear<B>,
    max_seq_length: usize,
}

impl PhonoGenerationModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PhonoGenerationModel<B> {
        let input_proj = LinearConfig::new(self.n_features, self.transformer.d_model).init(device);
        let embedding_pos =
            EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model).init(device);
        let transformer = self.transformer.init(device);
        let output_proj = LinearConfig::new(self.transformer.d_model, self.n_features).init(device);

        PhonoGenerationModel {
            transformer,
            input_proj,
            embedding_pos,
            output_proj,
            max_seq_length: self.max_seq_length,
        }
    }
}

impl<B: Backend> PhonoGenerationModel<B> {
    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }

    /// Raw forward pass: tokens `[batch, seq, FEATURES]` → predictions `[batch, seq, FEATURES]`.
    pub fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_length, _] = tokens.dims();
        let device = &self.devices()[0];

        let tokens = tokens.to_device(device);

        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let projected = self.input_proj.forward(tokens);
        let embedding = (embedding_positions + projected) / 2;

        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding).mask_attn(mask_attn),
        );

        sigmoid(self.output_proj.forward(encoded))
    }

    pub fn forward_training(&self, item: TrainingPhonoGenerationBatch<B>) -> RegressionOutput<B> {
        let [batch_size, seq_length, n_feats] = item.tokens_inputs.dims();
        let device = &self.devices()[0];

        let inputs = item.tokens_inputs.to_device(device);
        let targets = item.targets.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let projected = self.input_proj.forward(inputs);
        let embedding = (embedding_positions + projected) / 2;

        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding)
                .mask_pad(mask_pad.clone())
                .mask_attn(mask_attn),
        );

        let output = sigmoid(self.output_proj.forward(encoded));
        let output_flat = output.reshape([batch_size * seq_length, n_feats]);
        let targets_flat = targets.reshape([batch_size * seq_length, n_feats]);

        // 1.0 for real (non-padding) token positions, broadcast over feature dim
        let real_mask = mask_pad
            .reshape([batch_size * seq_length])
            .float()         // true(pad)→1.0, false(real)→0.0
            .neg()
            .add_scalar(1.0) // 0.0(pad), 1.0(real)
            .reshape([batch_size * seq_length, 1])
            .repeat_dim(1, n_feats);

        // 0.0 for NA feature triplets (structurally determined, not learned),
        // 1.0 for POS/NEG features and boundary bits — prevents NA domination
        let na_mask = na_feature_mask(&targets_flat, batch_size * seq_length, n_feats);

        let combined_mask = real_mask * na_mask;

        let diff = output_flat.clone() - targets_flat.clone();
        let loss = (diff.clone() * diff * combined_mask.clone()).sum()
            / combined_mask.sum().clamp_min(1.0);

        RegressionOutput::new(loss, output_flat, targets_flat)
    }
}

/// Build a loss mask that zeros out feature triplets where the target class is NA.
///
/// Each segment feature is encoded as a 3-wide one-hot (POS, NEG, NA). When the
/// NA slot is 1.0, the feature is structurally inapplicable (e.g. [ant] on a
/// non-coronal segment). We exclude these from the loss so the model only learns
/// to distinguish POS from NEG for features that actually matter.
///
/// The first two positions (WordBoundary / SylBoundary flags) are always included.
fn na_feature_mask<B: Backend>(targets: &Tensor<B, 2>, bs_sl: usize, n_feats: usize) -> Tensor<B, 2> {
    let seg_count = (n_feats - 2) / 3;
    let device = &targets.device();

    // Feature triplets: [bs_sl, seg_count * 3]
    let feat = targets.clone().slice([0..bs_sl, 2..n_feats]);
    // [bs_sl, seg_count, 3]
    let feat_3d = feat.reshape([bs_sl, seg_count, 3]);
    // NA indicator (last slot of each triplet): [bs_sl, seg_count, 1]
    let na = feat_3d.slice([0..bs_sl, 0..seg_count, 2..3]);
    // Defined mask: 1.0 where not NA, repeated across the triplet: [bs_sl, seg_count, 3]
    let defined_3d = na.neg().add_scalar(1.0).repeat_dim(2, 3);
    // Flatten: [bs_sl, seg_count * 3]
    let defined_flat = defined_3d.reshape([bs_sl, seg_count * 3]);

    // Always include the 2 boundary bits
    let boundary = Tensor::<B, 2>::ones([bs_sl, 2], device);
    Tensor::cat(vec![boundary, defined_flat], 1)
}

impl<B: AutodiffBackend> TrainStep for PhonoGenerationModel<B> {
    type Input = TrainingPhonoGenerationBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, item: TrainingPhonoGenerationBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let output = self.forward_training(item);
        let grads = output.loss.backward();
        TrainOutput::new(self, grads, output)
    }
}

impl<B: Backend> InferenceStep for PhonoGenerationModel<B> {
    type Input = TrainingPhonoGenerationBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, item: TrainingPhonoGenerationBatch<B>) -> RegressionOutput<B> {
        self.forward_training(item)
    }
}
