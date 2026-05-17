use getheode::phonology::segment::{SegmentFeatures, parse_segment};

use crate::data::token::PhonoToken;

pub struct PhonoTokenizer;

impl PhonoTokenizer {
    /// Encode an IPA-CHILDES utterance into a token sequence.
    ///
    /// The input format is whitespace-separated pieces where each piece is
    /// either `WORD_BOUNDARY` or one or more IPA bases (e.g. `k`, `oʊ`, `d̠ʒ`).
    /// A single piece may contain multiple bases (diphthong, affricate), each
    /// base is emitted as its own segment token. A single segment may also span
    /// multiple consecutive pieces if the first piece fails entirely.
    pub fn encode(&self, utterance: &str) -> Option<Vec<PhonoToken>> {
        let pieces: Vec<&str> = utterance.split_whitespace().collect();
        let mut tokens = Vec::new();
        let mut i = 0;

        while i < pieces.len() {
            let piece = pieces[i];

            if piece == "WORD_BOUNDARY" {
                tokens.push(PhonoToken::WordBoundary);
                i += 1;
                continue;
            }

            // Try parsing the piece greedily: a piece like "oʊ" yields two segments.
            let piece_tokens = parse_ipa_piece(piece);
            if !piece_tokens.is_empty() {
                tokens.extend(piece_tokens);
                i += 1;
                continue;
            }

            // Piece failed entirely, try concatenating following non-boundary pieces.
            let (tok, consumed) = try_concat(&pieces[i..]);
            if let Some(t) = tok {
                tokens.push(t);
            }
            i += consumed;
        }

        if tokens.is_empty() {
            None
        } else {
            Some(tokens)
        }
    }
}

/// Greedily parse as many IPA segments as possible from a single string.
/// "oʊ" → [segment(o), segment(ʊ)], "k" → [segment(k)], "???" → [].
fn parse_ipa_piece(s: &str) -> Vec<PhonoToken> {
    let mut tokens = Vec::new();
    let mut remaining = s;
    while !remaining.is_empty() {
        match parse_segment(remaining) {
            Ok((rest, feats)) => {
                tokens.push(make_segment(feats));
                remaining = rest;
            }
            Err(_) => break,
        }
    }
    tokens
}

/// When a piece fails to parse entirely, try concatenating it with the following
/// pieces (without spaces) until we get a parse. Never crosses WORD_BOUNDARY.
/// Returns (token, number_of_pieces_consumed).
fn try_concat(pieces: &[&str]) -> (Option<PhonoToken>, usize) {
    let mut candidate = String::from(pieces[0]);

    for n in 2..=pieces.len() {
        let next = pieces[n - 1];
        if next == "WORD_BOUNDARY" {
            break;
        }
        candidate.push_str(next);

        if let Ok(("", feats)) = parse_segment(&candidate) {
            return (Some(make_segment(feats)), n);
        }
    }

    eprintln!("unrecognised IPA token: {:?}", pieces[0]);
    (None, 1)
}

fn make_segment(feats: SegmentFeatures) -> PhonoToken {
    PhonoToken::Segment {
        seg_features: feats,
    }
}

#[cfg(test)]
mod tests {
    use getheode::phonology::feature::FeatureState;

    use super::*;

    #[test]
    fn test_word_boundary() {
        let tok = PhonoTokenizer;
        let tokens = tok.encode("k WORD_BOUNDARY t").unwrap();
        assert_eq!(tokens.len(), 3);
        assert!(matches!(tokens[0], PhonoToken::Segment { .. }));
        assert!(matches!(tokens[1], PhonoToken::WordBoundary));
        assert!(matches!(tokens[2], PhonoToken::Segment { .. }));
    }

    #[test]
    fn test_diphthong_splits_into_two() {
        let tok = PhonoTokenizer;
        let tokens = tok.encode("oʊ").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0], PhonoToken::Segment { .. }));
        assert!(matches!(tokens[1], PhonoToken::Segment { .. }));
    }

    #[test]
    fn test_independent_pieces_stay_separate() {
        let tok = PhonoTokenizer;
        let tokens = tok.encode("t ʃ").unwrap();
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_multi_piece_does_not_cross_boundary() {
        let tok = PhonoTokenizer;
        let tokens = tok.encode("t WORD_BOUNDARY ʃ").unwrap();
        assert_eq!(tokens.len(), 3);
        assert!(matches!(tokens[1], PhonoToken::WordBoundary));
    }

    #[test]
    fn test_single_utterance() {
        let tok = PhonoTokenizer;
        let tokens = tok
            .encode("d̠ʒ ʌ s t WORD_BOUNDARY l aɪ k WORD_BOUNDARY")
            .unwrap();
        assert!(!tokens.is_empty());
        let wb = tokens
            .iter()
            .position(|t| matches!(t, PhonoToken::WordBoundary));
        assert!(wb.is_some());
    }

    #[test]
    fn test_seg_features_are_ternary() {
        let tok = PhonoTokenizer;
        let tokens = tok.encode("k a").unwrap();
        for token in &tokens {
            if let PhonoToken::Segment { seg_features, .. } = token {
                // Every feature must be POS, NEG, or NA, never UNDEF
                for f in seg_features.features().iter() {
                    assert!(*f != FeatureState::UNDEF);
                }
            }
        }
    }

    #[test]
    fn test_empty_returns_none() {
        let tok = PhonoTokenizer;
        assert!(tok.encode("").is_none());
    }
}
