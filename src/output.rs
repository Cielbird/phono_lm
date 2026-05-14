use burn::{
    Tensor,
    tensor::{Transaction},
    train::{
        ItemLazy,
        metric::{Adaptor, LossInput},
    },
};

// raw output of the model
pub struct PhonoGenerationOutput {
    pub token_class_logits: Tensor<3>, // [batch_size, seq_len, TOK_CLASSES]
    pub token_features_logits: Tensor<3>, // [batch_size, seq_len, SEGMENT_FEATURES]
}

/// hierarchical multi-label classification for phonological tokens
#[derive(new)]
pub struct PhonoClassificationOutput {
    /// The loss.
    pub loss: Tensor<1>,

    /// Shape: \[batch_size, TOKEN_CLASSES+SEGMENT_FEATURES\].
    pub output: Tensor<2>,

    /// The ground truth class index for each sample. Shape: \[batch_size, TOKEN_CLASSES+SEGMENT_FEATURES\].
    pub targets: Tensor<2>,
}

impl ItemLazy for PhonoClassificationOutput {
    fn sync(self) -> Self {
        let [output, loss, targets] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        PhonoClassificationOutput {
            output: Tensor::from_data(output, device),
            loss: Tensor::from_data(loss, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}

impl Adaptor<LossInput> for PhonoClassificationOutput {
    fn adapt(&self) -> LossInput {
        LossInput::new(self.loss.clone())
    }
}
