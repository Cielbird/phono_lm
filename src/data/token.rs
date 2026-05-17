use getheode::phonology::segment::SegmentFeatures;

#[derive(Clone, Debug)]
pub enum PhonoToken {
    WordBoundary,
    SylBoundary,
    Segment { seg_features: SegmentFeatures },
}
