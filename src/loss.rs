use burn::tensor::{Bool, Int, Tensor, activation::log_softmax};

/// logits: [B, C]  — raw (pre-softmax) scores
/// targets: [B]    — class indices
/// mask: [B]       — true = include this sample in the loss
pub fn masked_cross_entropy(
    logits: Tensor<2>,
    targets: Tensor<1, Int>,
    mask: Tensor<1, Bool>,
) -> Tensor<1> {
    let [batch_size, _num_classes] = logits.dims();

    // 1. Log-softmax over class dimension
    let log_probs = log_softmax(logits, 1); // [B, C]

    // 2. Gather log-prob of the correct class for each sample
    //    targets needs shape [B, 1] for gather
    let targets_2d = targets.unsqueeze_dim(1); // [B, 1]
    let nll = log_probs.gather(1, targets_2d).squeeze_dim(1); // [B]
    let nll = nll.neg(); // NLL = -log(p)

    // 3. Zero out masked samples
    //    Convert bool mask to float: 1.0 = include, 0.0 = exclude
    let mask_float = mask.float(); // [B]
    let nll_masked = nll * mask_float.clone(); // [B]

    // 4. Mean over *unmasked* samples only (avoid dividing by zero)
    let num_active = mask_float.sum().clamp(1.0, batch_size as f32);
    nll_masked.sum() / num_active
}

/// logits:     [B, L]  — raw scores (logits, pre-sigmoid)
/// targets:    [B, L]  — binary ground truth (0.0 or 1.0)
/// mask:       [B, L]  — true = this (sample, label) pair contributes to loss
/// pos_weight  — multiplier applied to positive-target (1.0) entries to compensate
///               for the NEG-heavy imbalance typical in phonological feature sets.
///               1.0 = unweighted; ~5.0 suits ~85% NEG / 15% POS distributions.
pub fn masked_multilabel_bce(
    logits: Tensor<2>,
    targets: Tensor<2>,
    mask: Tensor<2, Bool>,
    pos_weight: f32,
) -> Tensor<1> {
    // 1. Numerically stable BCE per (sample, label)
    let zeros = Tensor::zeros_like(&logits);
    let relu_logits = logits.clone().max_pair(zeros);
    let abs_logits = logits.clone().abs();
    let log1p_exp = (abs_logits.neg().exp() + 1.0).log();

    let bce = relu_logits - logits * targets.clone() + log1p_exp; // [B, L]

    // 2. Weight positive targets: sample_weight = 1 for NEG, pos_weight for POS
    let sample_weight = targets * (pos_weight - 1.0) + 1.0;
    let bce = bce * sample_weight;

    // 3. Zero out masked positions and average over known ones only
    let mask_float = mask.float(); // [B, L]
    let num_active = mask_float.clone().sum().clamp(1.0, f32::MAX);

    (bce * mask_float).sum() / num_active
}
