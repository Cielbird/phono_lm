use crate::data::{PhonoGenerationBatch, TrainingPhonoGenerationBatch};

use burn::{
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        attention::generate_autoregressive_mask,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::activation::softmax,
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

#[derive(Config, Debug)]
pub struct PhonoGenerationModelConfig {
    transformer: TransformerEncoderConfig,
    vocab_size: usize,
    pad_token: usize,
    max_seq_length: usize,
}

#[derive(Module, Debug)]
pub struct PhonoGenerationModel {
    transformer: TransformerEncoder,
    embedding_token: Embedding,
    embedding_pos: Embedding,
    output: Linear,
    vocab_size: usize,
    pad_token: usize,
    max_seq_length: usize,
}

impl PhonoGenerationModelConfig {
    pub fn init(&self, device: &Device) -> PhonoGenerationModel {
        let d = self.transformer.d_model;
        let transformer = self.transformer.init(device);
        let embedding_token = EmbeddingConfig::new(self.vocab_size, d).init(device);
        let embedding_pos = EmbeddingConfig::new(self.max_seq_length, d).init(device);
        let output = LinearConfig::new(d, self.vocab_size).init(device);

        PhonoGenerationModel {
            transformer,
            embedding_token,
            embedding_pos,
            output,
            vocab_size: self.vocab_size,
            pad_token: self.pad_token,
            max_seq_length: self.max_seq_length,
        }
    }
}

impl PhonoGenerationModel {
    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }

    fn raw_forward(&self, tokens: Tensor<2, Int>, mask_pad: Tensor<2, Bool>) -> Tensor<3> {
        let [batch_size, seq_length] = tokens.dims();
        let device = &self.devices()[0];

        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens.to_device(device));
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let mask_attn = generate_autoregressive_mask(batch_size, seq_length, device);
        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding)
                .mask_pad(mask_pad.to_device(device))
                .mask_attn(mask_attn),
        );

        self.output.forward(encoded) // [B, S, VOCAB_SIZE]
    }

    pub fn forward(&self, item: TrainingPhonoGenerationBatch) -> ClassificationOutput {
        let [batch_size, seq_length] = item.tokens_inputs.dims();

        let logits = self.raw_forward(item.tokens_inputs, item.mask_pad); // [B, S, V]

        let output_flat = logits.reshape([batch_size * seq_length, self.vocab_size]);
        let targets_flat = item.targets.reshape([batch_size * seq_length]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_token]))
            .init(&output_flat.device())
            .forward(output_flat.clone(), targets_flat.clone());

        ClassificationOutput {
            loss,
            output: output_flat,
            targets: targets_flat,
        }
    }

    pub fn infer(&self, item: PhonoGenerationBatch) -> Tensor<3> {
        let [batch_size, seq_length] = item.tokens.dims();
        let logits = self.raw_forward(item.tokens, item.mask_pad); // [B, S, V]
        softmax(
            logits.reshape([batch_size * seq_length, self.vocab_size]),
            1,
        )
        .reshape([batch_size, seq_length, self.vocab_size])
    }
}

impl TrainStep for PhonoGenerationModel {
    type Input = TrainingPhonoGenerationBatch;
    type Output = ClassificationOutput;

    fn step(&self, item: TrainingPhonoGenerationBatch) -> TrainOutput<ClassificationOutput> {
        let output = self.forward(item);
        let grads = output.loss.backward();
        TrainOutput::new(self, grads, output)
    }
}

impl InferenceStep for PhonoGenerationModel {
    type Input = TrainingPhonoGenerationBatch;
    type Output = ClassificationOutput;

    fn step(&self, item: TrainingPhonoGenerationBatch) -> ClassificationOutput {
        self.forward(item)
    }
}
