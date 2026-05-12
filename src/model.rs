use crate::{
    data::{
        PHONO_LOGITS, PhonoGenerationBatch, TOKEN_CLASSES, TOKEN_FEATURES,
        TrainingPhonoGenerationBatch,
    },
    loss::{masked_cross_entropy, masked_multilabel_bce},
    output::{PhonoClassificationOutput, PhonoGenerationOutput},
};

use burn::{
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        attention::generate_autoregressive_mask,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::{
        activation::{sigmoid, softmax},
        backend::AutodiffBackend,
    },
    train::{InferenceStep, TrainOutput, TrainStep},
};

#[derive(Config, Debug)]
pub struct PhonoGenerationModelConfig {
    transformer: TransformerEncoderConfig,
    max_seq_length: usize,
}

#[derive(Module, Debug)]
pub struct PhonoGenerationModel<B: Backend> {
    transformer: TransformerEncoder<B>,
    input_proj: Linear<B>,
    embedding_pos: Embedding<B>,
    output_class: Linear<B>,
    output_features: Linear<B>,
    max_seq_length: usize,
}

impl PhonoGenerationModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PhonoGenerationModel<B> {
        let d = self.transformer.d_model;
        let input_proj = LinearConfig::new(PHONO_LOGITS, d).init(device);
        let embedding_pos = EmbeddingConfig::new(self.max_seq_length, d).init(device);
        let transformer = self.transformer.init(device);
        let output_class = LinearConfig::new(d, TOKEN_CLASSES).init(device);
        let output_features = LinearConfig::new(d, TOKEN_FEATURES).init(device);

        PhonoGenerationModel {
            transformer,
            input_proj,
            embedding_pos,
            output_class,
            output_features,
            max_seq_length: self.max_seq_length,
        }
    }
}

impl<B: Backend> PhonoGenerationModel<B> {
    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }

    /// Raw forward pass:
    /// - tokens [batch, seq, TOKEN_CLASSES+TOKEN_FEATURES] int
    /// - mask [batch, seq] bool
    /// -> predictions
    fn raw_forward(
        &self,
        tokens_inputs: Tensor<B, 3>,
        mask_pad: Tensor<B, 2, Bool>,
    ) -> PhonoGenerationOutput<B> {
        let [batch_size, seq_length, classes_plus_feats] = tokens_inputs.dims();
        assert_eq!(classes_plus_feats, PHONO_LOGITS);
        let device = &self.devices()[0];

        let inputs = tokens_inputs.to_device(device);
        let mask_pad = mask_pad.to_device(device);

        // Forward pass
        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let projected = self.input_proj.forward(inputs);
        let embedding = embedding_positions + projected;

        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);

        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding)
                .mask_pad(mask_pad.clone())
                .mask_attn(mask_attn),
        );

        let token_class_logits = self.output_class.forward(encoded.clone()); // [B, S, 3]
        let token_features_logits = self.output_features.forward(encoded); // [B, S, F]

        PhonoGenerationOutput {
            token_class_logits,
            token_features_logits,
        }
    }

    pub fn forward(&self, item: TrainingPhonoGenerationBatch<B>) -> PhonoClassificationOutput<B> {
        let [batch_size, seq_length, n_feats] = item.tokens_inputs.dims();
        assert_eq!(n_feats, PHONO_LOGITS);
        let device = &self.devices()[0];

        let targets = item.targets.to_device(device);

        let PhonoGenerationOutput {
            token_class_logits: class_logits,
            token_features_logits: feature_logits,
        } = self.raw_forward(item.tokens_inputs, item.mask_pad.clone());

        // Split targets
        let class_targets = targets
            .clone()
            .slice([0..batch_size, 0..seq_length, 0..TOKEN_CLASSES])
            .argmax(2)
            .reshape([batch_size, seq_length]); // [B, S]

        let feature_targets = targets.clone().slice([
            0..batch_size,
            0..seq_length,
            TOKEN_CLASSES..(TOKEN_CLASSES + TOKEN_FEATURES),
        ]); // [B, S, F]

        // TOKEN CLASS LOSS
        let class_loss = self.token_class_loss(
            batch_size,
            seq_length,
            class_logits.clone(),
            class_targets.clone(),
            item.mask_pad.clone(),
        );

        // SEGMENT FEATURE LOSS (masked BCE)
        let feature_loss = self.token_feature_loss(
            feature_logits.clone(),
            feature_targets,
            class_targets,
            item.mask_pad,
        );

        // final loss
        let loss = class_loss + feature_loss;

        let output = Tensor::cat(vec![class_logits, feature_logits], 2);
        let output_flat = output.reshape([batch_size * seq_length, n_feats]);
        let targets_flat = targets.reshape([batch_size * seq_length, n_feats]);

        PhonoClassificationOutput::new(loss, output_flat, targets_flat)
    }

    fn token_class_loss(
        &self,
        batch_size: usize,
        seq_len: usize,
        class_logits: Tensor<B, 3>,
        class_targets: Tensor<B, 2, Int>,
        mask_pad: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 1> {
        let logits = class_logits.reshape([batch_size * seq_len, 3]);
        let targets = class_targets.reshape([batch_size * seq_len]);
        let mask = (!mask_pad).reshape([batch_size * seq_len]);

        masked_cross_entropy(logits, targets, mask)
    }

    fn token_feature_loss(
        &self,
        feature_logits: Tensor<B, 3>,
        feature_targets: Tensor<B, 3>,
        class_targets: Tensor<B, 2, Int>,
        mask_pad: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 1> {
        let [batch_size, seq_length, _] = feature_targets.dims();

        // padding mask -> expand
        let pad_mask = (!mask_pad)
            .reshape([batch_size, seq_length, 1])
            .repeat_dim(2, TOKEN_FEATURES);

        // condition: only when token is segment (index 0)
        let isnt_boundary = class_targets.equal_elem(0);
        let cond_mask = isnt_boundary
            .reshape([batch_size, seq_length, 1])
            .repeat_dim(2, TOKEN_FEATURES);

        // NA and NEG both encode as 0.0, so training to predict 0 for
        // inapplicable sub-features is correct; with_invariants() converts
        // them to NA at inference time.
        let mask = pad_mask & cond_mask;

        let logits = feature_logits.reshape([batch_size * seq_length, TOKEN_FEATURES]);
        let targets = feature_targets.reshape([batch_size * seq_length, TOKEN_FEATURES]);
        let mask = mask.reshape([batch_size * seq_length, TOKEN_FEATURES]);

        masked_multilabel_bce(logits, targets, mask, 1.0)
    }

    // output: [batch_size, seq_len, class_probabilities + binary feature probabilities]
    pub fn infer(&self, item: PhonoGenerationBatch<B>) -> Tensor<B, 3> {
        let PhonoGenerationOutput {
            token_class_logits,
            token_features_logits,
        } = self.raw_forward(item.tokens, item.mask_pad);

        // softmax on the token class logits, and
        // sigmoid on each binary token feature
        Tensor::cat(
            vec![
                softmax(token_class_logits, 2),
                sigmoid(token_features_logits),
            ],
            2,
        )
    }
}


impl<B: AutodiffBackend> TrainStep for PhonoGenerationModel<B> {
    type Input = TrainingPhonoGenerationBatch<B>;
    type Output = PhonoClassificationOutput<B>;

    fn step(
        &self,
        item: TrainingPhonoGenerationBatch<B>,
    ) -> TrainOutput<PhonoClassificationOutput<B>> {
        let output = self.forward(item);
        let grads = output.loss.backward();
        TrainOutput::new(self, grads, output)
    }
}

impl<B: Backend> InferenceStep for PhonoGenerationModel<B> {
    type Input = TrainingPhonoGenerationBatch<B>;
    type Output = PhonoClassificationOutput<B>;

    fn step(&self, item: TrainingPhonoGenerationBatch<B>) -> PhonoClassificationOutput<B> {
        self.forward(item)
    }
}
