use super::{dataset::PhonoGenerationItem, token::FEATURES, tokenizer::PhonoTokenizer};
use burn::{data::dataloader::batcher::Batcher, prelude::*};
use std::sync::Arc;

#[derive(Clone, new)]
pub struct PhonoGenerationBatcher {
    tokenizer: Arc<PhonoTokenizer>,
    max_seq_length: usize,
}

#[derive(Debug, Clone, new)]
pub struct PhonoGenerationBatch<B: Backend> {
    pub tokens: Tensor<B, 3>,         // [batch, seq, FEATURES]
    pub mask_pad: Tensor<B, 2, Bool>, // [batch, seq] — true = pad
}

#[derive(Debug, Clone, new)]
pub struct TrainingPhonoGenerationBatch<B: Backend> {
    pub tokens_inputs: Tensor<B, 3>,  // [batch, seq-1, FEATURES]
    pub targets: Tensor<B, 3>,        // [batch, seq-1, FEATURES]
    pub mask_pad: Tensor<B, 2, Bool>, // [batch, seq-1]
}

impl<B: Backend> Batcher<B, PhonoGenerationItem, PhonoGenerationBatch<B>> for PhonoGenerationBatcher {
    fn batch(&self, items: Vec<PhonoGenerationItem>, device: &B::Device) -> PhonoGenerationBatch<B> {
        let batch_size = items.len();
        let seq_len = self.max_seq_length;

        let mut token_data = vec![0.0f32; batch_size * seq_len * FEATURES];
        let mut mask_data = vec![1i32; batch_size * seq_len]; // 1 = pad

        for (b, item) in items.iter().enumerate() {
            let tokens = self.tokenizer.encode(&item.text).unwrap_or_default();
            for (t, token) in tokens.into_iter().take(seq_len).enumerate() {
                let arr = token.to_arr();
                let offset = (b * seq_len + t) * FEATURES;
                token_data[offset..offset + FEATURES].copy_from_slice(&arr);
                mask_data[b * seq_len + t] = 0; // not pad
            }
        }

        let tokens = Tensor::<B, 1>::from_floats(token_data.as_slice(), device)
            .reshape([batch_size, seq_len, FEATURES]);

        let mask_pad = Tensor::<B, 1, Int>::from_ints(mask_data.as_slice(), device)
            .reshape([batch_size, seq_len])
            .equal_elem(1i32);

        PhonoGenerationBatch { tokens, mask_pad }
    }
}

impl<B: Backend> Batcher<B, PhonoGenerationItem, TrainingPhonoGenerationBatch<B>>
    for PhonoGenerationBatcher
{
    fn batch(
        &self,
        items: Vec<PhonoGenerationItem>,
        device: &B::Device,
    ) -> TrainingPhonoGenerationBatch<B> {
        let item: PhonoGenerationBatch<B> = self.batch(items, device);
        let [batch_size, seq_length, n_feats] = item.tokens.dims();

        let tokens_inputs = item.tokens.clone().slice([0..batch_size, 0..seq_length - 1, 0..n_feats]);
        let targets = item.tokens.slice([0..batch_size, 1..seq_length, 0..n_feats]);
        let mask_pad = item.mask_pad.slice([0..batch_size, 0..seq_length - 1]);

        TrainingPhonoGenerationBatch::new(tokens_inputs, targets, mask_pad)
    }
}
