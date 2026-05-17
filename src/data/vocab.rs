use std::collections::HashMap;

use getheode::phonology::{
    feature::FeatureState,
    segment::{SEG_FEATURE_COUNT, SegmentFeatures, format_segment},
};

use crate::data::{PhonoGenerationItem, PhonoToken, PhonoTokenizer};
use burn::data::dataset::Dataset;

pub const PAD_TOKEN: usize = 0;
const WORD_BOUNDARY_ID: usize = 1;
const SYL_BOUNDARY_ID: usize = 2;
const FIRST_SEGMENT_ID: usize = 3;

fn encode_feature(f: FeatureState) -> u8 {
    match f {
        FeatureState::UNDEF => 0,
        FeatureState::POS => 1,
        FeatureState::NEG => 2,
        FeatureState::NA => 3,
    }
}

fn decode_feature(v: u8) -> FeatureState {
    match v {
        1 => FeatureState::POS,
        2 => FeatureState::NEG,
        3 => FeatureState::NA,
        _ => FeatureState::UNDEF,
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct VocabData {
    // One Vec<u8> per segment (index = ID - FIRST_SEGMENT_ID), each u8 = encoded FeatureState
    segments: Vec<Vec<u8>>,
}

pub struct PhonoVocab {
    seg_to_id: HashMap<SegmentFeatures, usize>,
    id_to_seg: Vec<SegmentFeatures>,
}

impl PhonoVocab {
    pub fn build<D: Dataset<PhonoGenerationItem>>(
        datasets: &[&D],
        tokenizer: &PhonoTokenizer,
    ) -> Self {
        let mut seg_to_id = HashMap::new();
        let mut id_to_seg = Vec::new();

        for dataset in datasets {
            for i in 0..dataset.len() {
                let Some(item) = dataset.get(i) else { continue };
                let Some(tokens) = tokenizer.encode(&item.text) else {
                    continue;
                };
                for token in tokens {
                    if let PhonoToken::Segment { seg_features } = token {
                        if !seg_to_id.contains_key(&seg_features) {
                            let id = FIRST_SEGMENT_ID + id_to_seg.len();
                            seg_to_id.insert(seg_features.clone(), id);
                            id_to_seg.push(seg_features);
                        }
                    }
                }
            }
        }

        println!(
            "vocab: {} unique segments (vocab_size = {})",
            id_to_seg.len(),
            FIRST_SEGMENT_ID + id_to_seg.len()
        );
        Self {
            seg_to_id,
            id_to_seg,
        }
    }

    pub fn vocab_size(&self) -> usize {
        FIRST_SEGMENT_ID + self.id_to_seg.len()
    }

    pub fn to_id(&self, token: &PhonoToken) -> usize {
        match token {
            PhonoToken::WordBoundary => WORD_BOUNDARY_ID,
            PhonoToken::SylBoundary => SYL_BOUNDARY_ID,
            PhonoToken::Segment { seg_features } => match self.seg_to_id.get(seg_features) {
                Some(&id) => id,
                None => {
                    eprintln!(
                        "unknown segment '{}', defaulting to PAD",
                        format_segment(seg_features)
                    );
                    PAD_TOKEN
                }
            },
        }
    }

    pub fn from_id(&self, id: usize) -> PhonoToken {
        match id {
            0 | 1 => PhonoToken::WordBoundary,
            2 => PhonoToken::SylBoundary,
            n => {
                let idx = n - FIRST_SEGMENT_ID;
                if idx < self.id_to_seg.len() {
                    PhonoToken::Segment {
                        seg_features: self.id_to_seg[idx].clone(),
                    }
                } else {
                    PhonoToken::WordBoundary
                }
            }
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let segments = self
            .id_to_seg
            .iter()
            .map(|seg| {
                seg.features()
                    .iter()
                    .map(|&f| encode_feature(f))
                    .collect::<Vec<u8>>()
            })
            .collect();
        let data = VocabData { segments };
        let json = serde_json::to_string(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let data: VocabData = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut seg_to_id = HashMap::new();
        let mut id_to_seg = Vec::new();

        for (idx, encoded) in data.segments.into_iter().enumerate() {
            let mut arr = [FeatureState::UNDEF; SEG_FEATURE_COUNT];
            for (i, v) in encoded.into_iter().enumerate().take(SEG_FEATURE_COUNT) {
                arr[i] = decode_feature(v);
            }
            let seg = SegmentFeatures::from_features(arr);
            let id = FIRST_SEGMENT_ID + idx;
            seg_to_id.insert(seg.clone(), id);
            id_to_seg.push(seg);
        }

        Ok(Self {
            seg_to_id,
            id_to_seg,
        })
    }
}
