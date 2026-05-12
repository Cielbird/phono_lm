use burn::{
    Tensor,
    backend::NdArray,
    tensor::{Transaction, backend::Backend},
    train::{
        ItemLazy,
        metric::{Adaptor, LossInput},
    },
};

// raw output of the model
pub struct PhonoGenerationOutput<B: Backend> {
    pub token_class_logits: Tensor<B, 3>, // [batch_size, seq_len, TOK_CLASSES]
    pub token_features_logits: Tensor<B, 3>, // [batch_size, seq_len, SEGMENT_FEATURES]
}

/// hierarchical multi-label classification for phonological tokens
#[derive(new)]
pub struct PhonoClassificationOutput<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,

    /// Shape: \[batch_size, TOKEN_CLASSES+SEGMENT_FEATURES\].
    pub output: Tensor<B, 2>,

    /// The ground truth class index for each sample. Shape: \[batch_size, TOKEN_CLASSES+SEGMENT_FEATURES\].
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> ItemLazy for PhonoClassificationOutput<B> {
    type ItemSync = PhonoClassificationOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
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

impl<B: Backend> Adaptor<LossInput<B>> for PhonoClassificationOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
