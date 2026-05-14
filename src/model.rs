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
        Embedding, EmbeddingConfig, Gelu, Linear, LinearConfig,
        attention::generate_autoregressive_mask,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::activation::{sigmoid, softmax},
    train::{InferenceStep, TrainOutput, TrainStep},
};

#[derive(Config, Debug)]
pub struct PhonoGenerationModelConfig {
    transformer: TransformerEncoderConfig,
    input_proj_hidden: usize,
    output_class_hidden: usize,
    output_features_hidden: usize,
    max_seq_length: usize,
}

#[derive(Module, Debug)]
pub struct PhonoGenerationModel {
    transformer: TransformerEncoder,

    // input features MLP
    input_proj_1: Linear,
    input_proj_2: Linear,

    embedding_pos: Embedding,

    // output class MLP
    output_class_1: Linear,
    output_class_2: Linear,

    // output features MLP
    output_features_1: Linear,
    output_features_2: Linear,

    gelu: Gelu,
    max_seq_length: usize,
}

impl PhonoGenerationModelConfig {
    pub fn init(&self, device: &Device) -> PhonoGenerationModel {
        let d = self.transformer.d_model;
        let input_proj_1 = LinearConfig::new(PHONO_LOGITS, self.input_proj_hidden).init(device);
        let input_proj_2 = LinearConfig::new(self.input_proj_hidden, d).init(device);
        let embedding_pos = EmbeddingConfig::new(self.max_seq_length, d).init(device);
        let transformer = self.transformer.init(device);

        let output_class_1 = LinearConfig::new(d, self.output_class_hidden).init(device);
        let output_class_2 = LinearConfig::new(self.output_class_hidden, TOKEN_CLASSES).init(device);

        let output_features_1 = LinearConfig::new(d, self.output_features_hidden).init(device);
        let output_features_2 =
            LinearConfig::new(self.output_features_hidden, TOKEN_FEATURES).init(device);

        PhonoGenerationModel {
            transformer,
            input_proj_1,
            input_proj_2,
            embedding_pos,
            output_class_1,
            output_class_2,
            output_features_1,
            output_features_2,
            max_seq_length: self.max_seq_length,
            gelu: Gelu::new(),
        }
    }
}

impl PhonoGenerationModel {
    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }

    /// Raw forward pass:
    /// - tokens [batch, seq, TOKEN_CLASSES+TOKEN_FEATURES] int
    /// - mask [batch, seq] bool
    /// -> predictions
    fn raw_forward(
        &self,
        tokens_inputs: Tensor<3>,
        mask_pad: Tensor<2, Bool>,
    ) -> PhonoGenerationOutput {
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

        // input MLP
        let inputs = self.input_proj_1.forward(inputs);
        let inputs = self.gelu.forward(inputs);
        let projected = self.input_proj_2.forward(inputs);

        // add positional embedding
        let embedding = embedding_positions + projected;

        let mask_attn = generate_autoregressive_mask(batch_size, seq_length, device);

        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding)
                .mask_pad(mask_pad.clone())
                .mask_attn(mask_attn),
        );

        let encoded_class = self.output_class_1.forward(encoded.clone());
        let encoded_class = self.gelu.forward(encoded_class);
        let token_class_logits = self.output_class_2.forward(encoded_class); // [B, S, 3]

        let encoded_feats = self.output_features_1.forward(encoded);
        let encoded_feats = self.gelu.forward(encoded_feats);
        let token_features_logits = self.output_features_2.forward(encoded_feats); // [B, S, F]

        PhonoGenerationOutput {
            token_class_logits,
            token_features_logits,
        }
    }

    pub fn forward(&self, item: TrainingPhonoGenerationBatch) -> PhonoClassificationOutput {
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
        class_logits: Tensor<3>,
        class_targets: Tensor<2, Int>,
        mask_pad: Tensor<2, Bool>,
    ) -> Tensor<1> {
        let logits = class_logits.reshape([batch_size * seq_len, 3]);
        let targets = class_targets.reshape([batch_size * seq_len]);
        let mask = (!mask_pad).reshape([batch_size * seq_len]);

        masked_cross_entropy(logits, targets, mask)
    }

    fn token_feature_loss(
        &self,
        feature_logits: Tensor<3>,
        feature_targets: Tensor<3>,
        class_targets: Tensor<2, Int>,
        mask_pad: Tensor<2, Bool>,
    ) -> Tensor<1> {
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
    pub fn infer(&self, item: PhonoGenerationBatch) -> Tensor<3> {
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

impl TrainStep for PhonoGenerationModel {
    type Input = TrainingPhonoGenerationBatch;
    type Output = PhonoClassificationOutput;

    fn step(&self, item: TrainingPhonoGenerationBatch) -> TrainOutput<PhonoClassificationOutput> {
        let output = self.forward(item);
        let grads = output.loss.backward();
        TrainOutput::new(self, grads, output)
    }
}

impl InferenceStep for PhonoGenerationModel {
    type Input = TrainingPhonoGenerationBatch;
    type Output = PhonoClassificationOutput;

    fn step(&self, item: TrainingPhonoGenerationBatch) -> PhonoClassificationOutput {
        self.forward(item)
    }
}
