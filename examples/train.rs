use burn::optim::decay::WeightDecayConfig;
use phono_lm::training;
use phono_lm::{IpaChildesDataset, training::ExperimentConfig};

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

type Backend = burn::backend::Autodiff<burn::backend::LibTorch<Elem>>;

fn main() {
    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(128, 512, 4, 3)
            .with_norm_first(true),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    ).with_max_seq_length(128);

    training::train::<Backend, IpaChildesDataset>(
        if cfg!(target_os = "macos") {
            burn::tensor::Device::<Backend>::Mps
        } else {
            burn::tensor::Device::<Backend>::Cuda(0)
        },
        IpaChildesDataset::train(),
        IpaChildesDataset::test(),
        config,
        "/tmp/text-generation",
    );
}
