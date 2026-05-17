use super::{
    dataset::PhonoGenerationItem,
    tokenizer::PhonoTokenizer,
    vocab::{PAD_TOKEN, PhonoVocab},
};
use burn::{data::dataloader::batcher::Batcher, prelude::*};
use std::sync::Arc;

#[derive(Clone)]
pub struct PhonoGenerationBatcher {
    tokenizer: Arc<PhonoTokenizer>,
    vocab: Arc<PhonoVocab>,
    max_seq_length: usize,
}

impl PhonoGenerationBatcher {
    pub fn new(
        tokenizer: Arc<PhonoTokenizer>,
        vocab: Arc<PhonoVocab>,
        max_seq_length: usize,
    ) -> Self {
        Self {
            tokenizer,
            vocab,
            max_seq_length,
        }
    }
}

#[derive(Debug, Clone, new)]
pub struct PhonoGenerationBatch {
    pub tokens: Tensor<2, Int>,    // [batch, seq]
    pub mask_pad: Tensor<2, Bool>, // [batch, seq], true = pad
}

#[derive(Debug, Clone, new)]
pub struct TrainingPhonoGenerationBatch {
    pub tokens_inputs: Tensor<2, Int>, // [batch, seq-1]
    pub targets: Tensor<2, Int>,       // [batch, seq-1]
    pub mask_pad: Tensor<2, Bool>,     // [batch, seq-1], true when target is pad
}

impl Batcher<PhonoGenerationItem, PhonoGenerationBatch> for PhonoGenerationBatcher {
    fn batch(&self, items: Vec<PhonoGenerationItem>, device: &Device) -> PhonoGenerationBatch {
        let batch_size = items.len();
        let seq_len = self.max_seq_length;

        let mut token_data = vec![PAD_TOKEN as i32; batch_size * seq_len];
        let mut mask_data = vec![true; batch_size * seq_len];

        for (b, item) in items.iter().enumerate() {
            let tokens = self.tokenizer.encode(&item.text).unwrap_or_default();
            for (t, token) in tokens.into_iter().take(seq_len).enumerate() {
                token_data[b * seq_len + t] = self.vocab.to_id(&token) as i32;
                mask_data[b * seq_len + t] = false;
            }
        }

        let tokens = Tensor::<1, Int>::from_ints(token_data.as_slice(), device)
            .reshape([batch_size, seq_len]);
        let mask_pad = Tensor::<1, Bool>::from_bool(mask_data.as_slice(), device)
            .reshape([batch_size, seq_len]);

        PhonoGenerationBatch { tokens, mask_pad }
    }
}

impl Batcher<PhonoGenerationItem, TrainingPhonoGenerationBatch> for PhonoGenerationBatcher {
    fn batch(
        &self,
        items: Vec<PhonoGenerationItem>,
        device: &Device,
    ) -> TrainingPhonoGenerationBatch {
        let item: PhonoGenerationBatch = self.batch(items, device);
        let [batch_size, seq_length] = item.tokens.dims();

        let tokens_inputs = item
            .tokens
            .clone()
            .slice([0..batch_size, 0..seq_length - 1]);
        let targets = item.tokens.slice([0..batch_size, 1..seq_length]);
        let mask_pad = item.mask_pad.slice([0..batch_size, 1..seq_length]);

        TrainingPhonoGenerationBatch::new(tokens_inputs, targets, mask_pad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::WgpuDevice;

    fn make_batcher(max_seq: usize) -> PhonoGenerationBatcher {
        let tokenizer = Arc::new(PhonoTokenizer);
        let dataset = vec![
            PhonoGenerationItem::new("k a".to_string()),
            PhonoGenerationItem::new("k WORD_BOUNDARY a".to_string()),
        ];
        struct SimpleDataset(Vec<PhonoGenerationItem>);
        impl burn::data::dataset::Dataset<PhonoGenerationItem> for SimpleDataset {
            fn get(&self, i: usize) -> Option<PhonoGenerationItem> {
                self.0.get(i).cloned()
            }
            fn len(&self) -> usize {
                self.0.len()
            }
        }
        let ds = SimpleDataset(dataset);
        let vocab = Arc::new(PhonoVocab::build(&[&ds], &tokenizer));
        PhonoGenerationBatcher::new(tokenizer, vocab, max_seq)
    }

    fn item(text: &str) -> PhonoGenerationItem {
        PhonoGenerationItem::new(text.to_string())
    }

    fn device() -> Device {
        WgpuDevice::default().into()
    }

    #[test]
    fn shift_is_correct() {
        let device = device();
        let batch: TrainingPhonoGenerationBatch = make_batcher(5).batch(vec![item("k a")], &device);

        let [_, seq] = batch.tokens_inputs.dims();
        assert_eq!(seq, 4);

        let inp: Vec<i32> = batch.tokens_inputs.into_data().to_vec().unwrap();
        assert!(inp[0] >= 3, "input[0] should be a segment token (id >= 3)");

        let tgt: Vec<i32> = batch.targets.into_data().to_vec().unwrap();
        assert!(tgt[0] >= 3, "target[0] should be a segment token (id >= 3)");
        assert_eq!(
            tgt[1], PAD_TOKEN as i32,
            "first padding target should be PAD"
        );
    }

    #[test]
    fn mask_keyed_to_targets() {
        let device = device();
        let batch: TrainingPhonoGenerationBatch = make_batcher(5).batch(vec![item("k a")], &device);

        let mask: Vec<i32> = batch.mask_pad.int().into_data().to_vec().unwrap();
        assert_eq!(mask, vec![0, 1, 1, 1]);
    }

    #[test]
    fn word_boundary_class_encoding() {
        let device = device();
        let batch: TrainingPhonoGenerationBatch =
            make_batcher(4).batch(vec![item("k WORD_BOUNDARY a")], &device);

        let mask: Vec<i32> = batch.mask_pad.int().into_data().to_vec().unwrap();
        assert_eq!(mask, vec![0, 0, 1]);

        let tgt: Vec<i32> = batch.targets.into_data().to_vec().unwrap();
        assert_eq!(tgt[0], 1, "target[0] should be WORD_BOUNDARY (id=1)");
    }
}
