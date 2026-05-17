use crate::{
    data::{PAD_TOKEN, PhonoGenerationBatcher, PhonoGenerationItem, PhonoTokenizer, PhonoVocab},
    model::PhonoGenerationModelConfig,
};
use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{Dataset, transform::SamplerDataset},
    },
    lr_scheduler::noam::NoamLrSchedulerConfig,
    nn::transformer::TransformerEncoderConfig,
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, DefaultRecorder, Recorder},
    train::{
        Learner, SupervisedTraining,
        metric::{AccuracyMetric, CudaMetric, LearningRateMetric, LossMetric},
    },
};
use std::sync::Arc;

#[derive(Config, Debug)]
pub struct ExperimentConfig {
    pub transformer: TransformerEncoderConfig,
    pub optimizer: AdamConfig,
    #[config(default = 512)]
    pub max_seq_length: usize,
    #[config(default = 6)]
    pub batch_size: usize,
    #[config(default = 50)]
    pub num_epochs: usize,
}

pub fn train<D: Dataset<PhonoGenerationItem> + 'static>(
    device: Device,
    dataset_train: D,
    dataset_test: D,
    config: ExperimentConfig,
    artifact_dir: &str,
) {
    std::fs::create_dir_all(artifact_dir).expect("failed to create artifact dir");

    let tokenizer = Arc::new(PhonoTokenizer);

    let vocab = PhonoVocab::build(&[&dataset_train, &dataset_test], &tokenizer);
    vocab
        .save(&format!("{artifact_dir}/vocab.json"))
        .expect("failed to save vocab");
    let vocab = Arc::new(vocab);

    let batcher = PhonoGenerationBatcher::new(tokenizer, vocab.clone(), config.max_seq_length);
    let device = device.autodiff();

    let model = PhonoGenerationModelConfig::new(
        config.transformer.clone(),
        vocab.vocab_size(),
        PAD_TOKEN,
        config.max_seq_length,
    )
    .init(&device);

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_train, 10_000));

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_test, 1000));

    let optim = config.optimizer.init();
    let lr_scheduler = NoamLrSchedulerConfig::new(1.0)
        .with_warmup_steps(200)
        .with_model_size(config.transformer.d_model)
        .init()
        .unwrap();

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let result = training.launch(Learner::new(model, optim, lr_scheduler));

    config.save(format!("{artifact_dir}/config.json")).unwrap();

    DefaultRecorder::new()
        .record(
            result.model.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}
