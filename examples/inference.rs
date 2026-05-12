use getheode::phonology::segment::format_segment;
use phono_lm::{
    PhonoToken, PhonoTokenizer,
    inference::{PhonologicalGenerate, load_model},
};

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

type Backend = burn::backend::LibTorch<Elem>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let artifact_dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/tmp/text-generation");
    let n_tokens: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let seed_ipa: Option<&str> = args.get(3).map(|s| s.as_str());

    let device = if cfg!(target_os = "macos") {
        burn::tensor::Device::<Backend>::Mps
    } else {
        burn::tensor::Device::<Backend>::Cuda(0)
    };

    let model = load_model::<Backend>(artifact_dir, &device);

    let seed = seed_ipa.and_then(|ipa| {
        let tokens = PhonoTokenizer.encode(ipa);
        if tokens.is_none() {
            eprintln!("warning: seed IPA could not be parsed, using default seed");
        }
        tokens
    });

    let generator = PhonologicalGenerate::new(&model, &device, n_tokens);
    let generator = match seed {
        Some(s) => generator.with_seed(s),
        None => generator,
    };
    let generated = generator.run();

    print_tokens(&generated);
    println!("\n{} tokens generated", generated.len());
}

fn print_tokens(tokens: &[PhonoToken]) {
    for token in tokens {
        match token {
            PhonoToken::WordBoundary => print!(" | "),
            PhonoToken::SylBoundary => print!("."),
            PhonoToken::Segment { seg_features, .. } => {
                // seg_features already has invariants enforced by snap_to_token
                print!("{}", format_segment(seg_features));
            }
        }
    }
    println!();
}
