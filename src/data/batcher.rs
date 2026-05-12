use crate::data::PHONO_LOGITS;

use super::{dataset::PhonoGenerationItem, tokenizer::PhonoTokenizer};
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
    pub mask_pad: Tensor<B, 2, Bool>, // [batch, seq], true = pad
}

#[derive(Debug, Clone, new)]
pub struct TrainingPhonoGenerationBatch<B: Backend> {
    pub tokens_inputs: Tensor<B, 3>,  // [batch, seq-1, FEATURES]
    pub targets: Tensor<B, 3>,        // [batch, seq-1, FEATURES]
    pub mask_pad: Tensor<B, 2, Bool>, // [batch, seq-1], true when pad
}

impl<B: Backend> Batcher<B, PhonoGenerationItem, PhonoGenerationBatch<B>>
    for PhonoGenerationBatcher
{
    fn batch(
        &self,
        items: Vec<PhonoGenerationItem>,
        device: &B::Device,
    ) -> PhonoGenerationBatch<B> {
        let batch_size = items.len();
        let seq_len = self.max_seq_length;

        let mut token_data = vec![0.; batch_size * seq_len * PHONO_LOGITS];
        let mut mask_data = vec![true; batch_size * seq_len]; // true = pad

        for (b, item) in items.iter().enumerate() {
            let tokens = self.tokenizer.encode(&item.text).unwrap_or_default();
            for (t, token) in tokens.into_iter().take(seq_len).enumerate() {
                let arr = token.to_probs();
                let offset = (b * seq_len + t) * PHONO_LOGITS;
                token_data[offset..offset + PHONO_LOGITS].copy_from_slice(&arr);
                mask_data[b * seq_len + t] = false; // not pad
            }
        }

        let tokens = Tensor::<B, 1>::from_floats(token_data.as_slice(), device).reshape([
            batch_size,
            seq_len,
            PHONO_LOGITS,
        ]);

        let mask_pad = Tensor::<B, 1, Bool>::from_bool(mask_data.as_slice(), device)
            .reshape([batch_size, seq_len]);

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

        let tokens_inputs =
            item.tokens
                .clone()
                .slice([0..batch_size, 0..seq_length - 1, 0..n_feats]);
        let targets = item
            .tokens
            .slice([0..batch_size, 1..seq_length, 0..n_feats]);
        let mask_pad = item.mask_pad.slice([0..batch_size, 1..seq_length]);

        TrainingPhonoGenerationBatch::new(tokens_inputs, targets, mask_pad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn make_batcher(max_seq: usize) -> PhonoGenerationBatcher {
        PhonoGenerationBatcher::new(Arc::new(super::super::PhonoTokenizer), max_seq)
    }

    fn item(text: &str) -> PhonoGenerationItem {
        PhonoGenerationItem::new(text.to_string())
    }

    // "k a" encodes to [segment(k), segment(a)], 2 real tokens.
    // With max_seq=5: full seq = [k, a, PAD, PAD, PAD]
    //   tokens_inputs = [k, a, PAD, PAD]       (positions 0..4)
    //   targets       = [a, PAD, PAD, PAD]      (positions 1..5)
    //   mask_pad      = [F, T, T, T]            (true = target is padding)
    #[test]
    fn shift_is_correct() {
        let device = Default::default();
        let batch: TrainingPhonoGenerationBatch<B> =
            make_batcher(5).batch(vec![item("k a")], &device);

        let [_, seq, feats] = batch.tokens_inputs.dims();
        assert_eq!(seq, 4);
        assert_eq!(feats, PHONO_LOGITS);

        // tokens_inputs[0] = k (segment): class one-hot [1, 0, 0]
        let inp: Vec<f32> = batch.tokens_inputs.into_data().to_vec().unwrap();
        assert_eq!(inp[0], 1.0, "input[0] class 0 should be 1 (segment)");
        assert_eq!(inp[1], 0.0, "input[0] class 1 should be 0");
        assert_eq!(inp[2], 0.0, "input[0] class 2 should be 0");

        // targets[0] = a (segment): class one-hot [1, 0, 0]
        let tgt: Vec<f32> = batch.targets.into_data().to_vec().unwrap();
        assert_eq!(tgt[0], 1.0, "target[0] class 0 should be 1 (segment)");
        assert_eq!(tgt[1], 0.0, "target[0] class 1 should be 0");
        assert_eq!(tgt[2], 0.0, "target[0] class 2 should be 0");

        // targets[1] (first PAD) should be all zeros
        let pad_offset = PHONO_LOGITS;
        for i in pad_offset..pad_offset + PHONO_LOGITS {
            assert_eq!(tgt[i], 0.0, "padding target[1][{i}] should be 0");
        }
    }

    // mask_pad must be true for every position where the TARGET is padding,
    // not where the input is padding.
    #[test]
    fn mask_keyed_to_targets() {
        let device = Default::default();
        let batch: TrainingPhonoGenerationBatch<B> =
            make_batcher(5).batch(vec![item("k a")], &device);

        let mask: Vec<bool> = batch.mask_pad.into_data().to_vec().unwrap();
        // targets = [a, PAD, PAD, PAD] → mask = [false, true, true, true]
        assert_eq!(mask, vec![false, true, true, true]);
    }

    // Word-boundary class is index 2 (one-hot [0, 0, 1]).
    #[test]
    fn word_boundary_class_encoding() {
        let device = Default::default();
        // "k WORD_BOUNDARY a" → [seg(k), word_boundary, seg(a)]
        // targets = [word_boundary, seg(a), PAD], mask = [false, false, true]
        let batch: TrainingPhonoGenerationBatch<B> =
            make_batcher(4).batch(vec![item("k WORD_BOUNDARY a")], &device);

        let mask: Vec<bool> = batch.mask_pad.into_data().to_vec().unwrap();
        assert_eq!(mask, vec![false, false, true]);

        let tgt: Vec<f32> = batch.targets.into_data().to_vec().unwrap();
        assert_eq!(tgt[0], 0.0, "target[0] class 0 (segment) should be 0");
        assert_eq!(tgt[1], 0.0, "target[0] class 1 (syl_boundary) should be 0");
        assert_eq!(tgt[2], 1.0, "target[0] class 2 (word_boundary) should be 1");
    }
}
