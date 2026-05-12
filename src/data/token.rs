use getheode::phonology::{
    feature::FeatureState,
    segment::{SEG_FEATURE_COUNT, SegmentFeatures},
    syllable::{SYL_FEATURE_COUNT, SyllableFeatures},
};

#[derive(Clone)]
pub enum PhonoToken {
    WordBoundary,
    SylBoundary,
    Segment {
        syl_features: SyllableFeatures,
        seg_features: SegmentFeatures,
    },
}

pub const TOKEN_FEATURES: usize = SYL_FEATURE_COUNT + SEG_FEATURE_COUNT;
pub const TOKEN_CLASSES: usize = 3; // segment, syl bound, or word bound
pub const PHONO_LOGITS: usize = TOKEN_CLASSES + TOKEN_FEATURES;

impl PhonoToken {
    /// 3 token classes + syllable and segment binary features
    /// token classes :
    /// - 0 : segment with features
    /// - 1 : syllable boundary
    /// - 2 : word boundary
    ///
    /// segment (and syl) features :
    /// all binary: 1=POS, 0=NEG or NA (always ignored), and UNDEF impossible
    ///
    /// if discriminant is 1 or 2 (word or syllable boundary), the binary segment features are
    /// irrelevant (NA).
    /// some segment features are dependent on other segment features, so those are sometimes ignored
    /// too
    pub fn to_probs(&self) -> [f32; PHONO_LOGITS] {
        let mut arr = [0.0; PHONO_LOGITS];
        match self {
            PhonoToken::WordBoundary => arr[2] = 1.,
            PhonoToken::SylBoundary => arr[1] = 1.,
            PhonoToken::Segment {
                syl_features,
                seg_features,
            } => {
                arr[0] = 1.;
                for (i, &f) in syl_features
                    .features()
                    .iter()
                    .chain(seg_features.features().iter())
                    .enumerate()
                {
                    let base = TOKEN_CLASSES + i;
                    arr[base] = match f {
                        FeatureState::POS => 1.,
                        FeatureState::NEG => 0.,
                        _ => 0., // NA or UNDEF, ignored in loss func
                    }
                }
            }
        }
        arr
    }

    pub fn from_probs(arr: [f32; PHONO_LOGITS]) -> Self {
        if arr[2] > 0.5 {
            return Self::WordBoundary;
        }
        if arr[1] > 0.5 {
            return Self::SylBoundary;
        }
        let syl_count = SYL_FEATURE_COUNT;
        let seg_count = SEG_FEATURE_COUNT;
        let mut syl_features = [FeatureState::NEG; SYL_FEATURE_COUNT];
        let mut seg_features = [FeatureState::NEG; SEG_FEATURE_COUNT];
        for i in 0..syl_count {
            if arr[TOKEN_CLASSES + i] > 0.5 {
                syl_features[i] = FeatureState::POS;
            }
        }
        for i in 0..seg_count {
            if arr[TOKEN_CLASSES + SYL_FEATURE_COUNT + i] > 0.5 {
                seg_features[i] = FeatureState::POS;
            }
        }
        Self::Segment {
            syl_features: SyllableFeatures::from_features(syl_features),
            seg_features: SegmentFeatures::from_features(seg_features).with_invariants(),
        }
    }
}
