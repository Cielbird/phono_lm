use getheode::phonology::{
    feature::FeatureState,
    segment::SEG_FEATURE_COUNT,
    syllable::SYL_FEATURE_COUNT,
};

#[derive(Clone, Copy)]
pub enum PhonoToken {
    WordBoundary,
    SylBoundary,
    Segment {
        syl_features: [FeatureState; SYL_FEATURE_COUNT as usize],
        seg_features: [FeatureState; SEG_FEATURE_COUNT as usize],
    },
}

// 2 boundary bits + 3 bits per feature (one-hot: POS, NEG, NA/UNDEF)
pub const FEATURES: usize = 2 + (SYL_FEATURE_COUNT as usize + SEG_FEATURE_COUNT as usize) * 3;

impl PhonoToken {
    pub fn to_arr(&self) -> [f32; FEATURES] {
        let mut arr = [0.0f32; FEATURES];
        match self {
            PhonoToken::WordBoundary => arr[0] = 1.0,
            PhonoToken::SylBoundary => arr[1] = 1.0,
            PhonoToken::Segment { syl_features, seg_features } => {
                for (i, &f) in syl_features.iter().chain(seg_features.iter()).enumerate() {
                    let base = 2 + i * 3;
                    match f {
                        FeatureState::POS => arr[base] = 1.0,
                        FeatureState::NEG => arr[base + 1] = 1.0,
                        _ => arr[base + 2] = 1.0, // NA or UNDEF
                    }
                }
            }
        }
        arr
    }

    pub fn from_arr(arr: [f32; FEATURES]) -> Self {
        if arr[0] > 0.5 {
            return Self::WordBoundary;
        }
        if arr[1] > 0.5 {
            return Self::SylBoundary;
        }
        let syl_count = SYL_FEATURE_COUNT as usize;
        let seg_count = SEG_FEATURE_COUNT as usize;
        let mut syl_features = [FeatureState::NA; SYL_FEATURE_COUNT as usize];
        let mut seg_features = [FeatureState::NA; SEG_FEATURE_COUNT as usize];
        for i in 0..syl_count {
            syl_features[i] = argmax_triplet(&arr, 2 + i * 3);
        }
        for i in 0..seg_count {
            seg_features[i] = argmax_triplet(&arr, 2 + (syl_count + i) * 3);
        }
        Self::Segment { syl_features, seg_features }
    }

    /// Construct a Segment token from a `SegmentFeatures` array (syl_features left as NA).
    pub fn from_seg_features(seg_features: &[FeatureState; SEG_FEATURE_COUNT as usize]) -> Self {
        Self::Segment {
            syl_features: [FeatureState::NA; SYL_FEATURE_COUNT as usize],
            seg_features: *seg_features,
        }
    }
}

fn argmax_triplet(arr: &[f32], base: usize) -> FeatureState {
    let (pos, neg, na) = (arr[base], arr[base + 1], arr[base + 2]);
    if pos >= neg && pos >= na {
        FeatureState::POS
    } else if neg >= na {
        FeatureState::NEG
    } else {
        FeatureState::NA
    }
}
