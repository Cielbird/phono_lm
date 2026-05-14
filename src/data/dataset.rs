use burn::data::dataset::{
    Dataset, SqliteDataset, source::huggingface::HuggingfaceDatasetLoader,
    transform::PartialDataset,
};

#[derive(new, Clone, Debug)]
pub struct PhonoGenerationItem {
    pub text: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IpaChildesItem {
    pub ipa_transcription: String,
}

pub struct IpaChildesDataset {
    dataset: PartialDataset<SqliteDataset<IpaChildesItem>, IpaChildesItem>,
}

impl Dataset<PhonoGenerationItem> for IpaChildesDataset {
    fn get(&self, index: usize) -> Option<PhonoGenerationItem> {
        self.dataset
            .get(index)
            .map(|item| PhonoGenerationItem::new(item.ipa_transcription))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl IpaChildesDataset {
    pub fn train() -> Self {
        let dataset = Self::open();
        let n = dataset.len();
        Self {
            dataset: PartialDataset::new(dataset, 0, n * 4 / 5),
        }
    }

    pub fn test() -> Self {
        let dataset = Self::open();
        let n = dataset.len();
        Self {
            dataset: PartialDataset::new(dataset, n * 4 / 5, n),
        }
    }

    fn open() -> SqliteDataset<IpaChildesItem> {
        HuggingfaceDatasetLoader::new("phonemetransformers/IPA-CHILDES")
            .with_subset("EnglishNA")
            .dataset("train")
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{
        PHONO_LOGITS, PhonoGenerationBatcher, PhonoToken, PhonoTokenizer, TOKEN_CLASSES,
        TrainingPhonoGenerationBatch,
    };
    use burn::backend::NdArray;
    use burn::data::dataloader::batcher::Batcher;
    use std::sync::Arc;

    // Run with: cargo test probe_dataset -- --ignored --nocapture
    #[test]
    #[ignore]
    fn probe_dataset() {
        let train = IpaChildesDataset::train();
        let test = IpaChildesDataset::test();

        println!("train items : {}", train.len());
        println!("test  items : {}", test.len());
        assert!(train.len() > 0, "training set is empty");
        assert!(test.len() > 0, "test set is empty");

        let tokenizer = PhonoTokenizer;

        // Sample 20 items and report token-class distribution
        let n_probe = 20.min(train.len());
        let mut n_segments = 0usize;
        let mut n_syl = 0usize;
        let mut n_wb = 0usize;
        let mut n_untokenizable = 0usize;

        for i in 0..n_probe {
            let item = train.get(i).unwrap();
            match tokenizer.encode(&item.text) {
                None => {
                    n_untokenizable += 1;
                    println!("  [{}] UNTOKENIZABLE: {:?}", i, item.text);
                }
                Some(tokens) => {
                    print!("  [{}] {:?} → {} tokens  [", i, item.text, tokens.len());
                    for tok in &tokens {
                        match tok {
                            PhonoToken::Segment { .. } => {
                                n_segments += 1;
                                print!("S");
                            }
                            PhonoToken::SylBoundary => {
                                n_syl += 1;
                                print!("|");
                            }
                            PhonoToken::WordBoundary => {
                                n_wb += 1;
                                print!("#");
                            }
                        }
                    }
                    println!("]");
                }
            }
        }

        println!("\ntoken breakdown over {} items:", n_probe);
        println!("  segments        : {n_segments}");
        println!("  syl boundaries  : {n_syl}");
        println!("  word boundaries : {n_wb}");
        println!("  untokenizable   : {n_untokenizable}");

        // Batch the first 4 items and validate shapes and mask
        let batcher = PhonoGenerationBatcher::new(Arc::new(tokenizer), 32);
        let items: Vec<_> = (0..4).filter_map(|i| train.get(i)).collect();
        let batch: TrainingPhonoGenerationBatch = batcher.batch(items, &Default::default());

        let [bs, seq, feats] = batch.tokens_inputs.dims();
        println!(
            "\nbatch shapes: inputs=[{bs},{seq},{feats}]  targets={:?}  mask={:?}",
            batch.targets.dims(),
            batch.mask_pad.dims()
        );
        assert_eq!(feats, PHONO_LOGITS);
        assert_eq!(batch.targets.dims(), [bs, seq, feats]);
        assert_eq!(batch.mask_pad.dims(), [bs, seq]);

        // Every unmasked target position must have exactly one hot class bit set
        let targets: Vec<f32> = batch.targets.clone().into_data().to_vec().unwrap();
        let mask: Vec<bool> = batch.mask_pad.into_data().to_vec().unwrap();

        let mut bad = 0usize;
        for b in 0..bs {
            for t in 0..seq {
                if mask[b * seq + t] {
                    continue;
                } // padding target, skip
                let base = (b * seq + t) * feats;
                let class_sum: f32 = targets[base..base + TOKEN_CLASSES].iter().sum();
                if (class_sum - 1.0).abs() > 1e-4 {
                    bad += 1;
                    println!("  BAD class encoding at batch={b} pos={t}: sum={class_sum}");
                }
            }
        }
        assert_eq!(bad, 0, "{bad} target positions have invalid class one-hots");
        println!("\nAll unmasked target positions have valid class one-hots. ✓");
    }
}
