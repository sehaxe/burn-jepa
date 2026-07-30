//! # burn-jepa — Self-Supervised Losses for Burn
//!
//! | arXiv | Loss | What |
//! |-------|------|------|
//! | [2511.08544](https://arxiv.org/abs/2511.08544) | `lejepa_loss` | Isotropic Gaussian — ∥mean∥² + ∥cov-I∥²/D |
//! | [2304.07193](https://arxiv.org/abs/2304.07193) | `koleo_loss` | KoLeo uniformity — -mean(log(min_dist)) |
//! | [2104.14294](https://arxiv.org/abs/2104.14294) | `EmaTeacher` | Momentum EMA — student tracks teacher |
use burn::module::{Module, Param};
use burn::tensor::{backend::Backend, Tensor};

/// LeJEPA: isotropic Gaussian regularization (Balestriero & LeCun, 2025).
///
/// Constrains embeddings toward an isotropic Gaussian N(0, I).
/// Single trade-off hyperparameter: weight of this loss against MSE.
///
/// ```text
/// L = ∥mean(z)∥² + ∥cov(z) - I∥² / D
/// ```
pub fn lejepa_loss<B: Backend>(z: Tensor<B, 2>) -> Tensor<B, 1> {
    let [_n, d] = z.dims();
    let dev = z.device();
    let mean_loss = z.clone().mean_dim(0).powf_scalar(2.0).sum();
    let z_c = z.clone() - z.clone().mean_dim(0).unsqueeze();
    let cov = z_c
        .clone()
        .transpose()
        .matmul(z_c)
        .div_scalar((_n.max(2) - 1) as f32);
    let eye = Tensor::eye(d, &dev);
    let cov_loss = (cov - eye).powf_scalar(2.0).sum().div_scalar(d as f32);
    (mean_loss + cov_loss).unsqueeze()
}

/// KoLeo: uniformity regularizer (Caron et al., DINOv2, 2023).
///
/// Encourages embeddings to spread uniformly on the unit sphere.
/// Subsamples to 256 tokens for O(n²) efficiency.
///
/// ```text
/// L = -mean(log(min_pairwise_dist))
/// ```
pub fn koleo_loss<B: Backend>(z: Tensor<B, 2>) -> Tensor<B, 1> {
    let [n, d] = z.dims();
    let dev = z.device();
    let m = 256usize.min(n);
    let z_sub = if m < n {
        let norms = z.clone().powf_scalar(2.0).sum_dim(1);
        let (_vals, idx) = norms.topk_with_indices(m, 0);
        let idx_2d = idx.expand([m, d]);
        z.clone().gather(0, idx_2d)
    } else {
        z
    };
    let z_sq = z_sub.clone().powf_scalar(2.0).sum_dim(1).clamp_min(1e-12);
    let z_norm = z_sub / z_sq.sqrt();
    let dots = z_norm.clone().matmul(z_norm.transpose());
    let dists = dots.neg().mul_scalar(2.0).add_scalar(2.0) + Tensor::eye(m, &dev).mul_scalar(1e6);
    let nn_dists = dists.min_dim(1).clamp_min(1e-8);
    nn_dists.log().neg().mean().unsqueeze()
}

/// Momentum EMA teacher — student tracks teacher via exponential moving average.
///
/// From DINO (2104.14294): `teacher = momentum * teacher + (1-momentum) * student`.
/// Default momentum: 0.996.
#[derive(Module, Debug)]
pub struct EmaTeacher<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub update_count: Param<Tensor<B, 1>>,
    #[module(skip)]
    pub momentum: f32,
}

impl<B: Backend> EmaTeacher<B> {
    pub fn new(value: Tensor<B, 1>, momentum: f32, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(value),
            update_count: Param::from_tensor(Tensor::ones([1], device)),
            momentum,
        }
    }

    pub fn update(&mut self, student: Tensor<B, 1>) {
        let m = self.momentum;
        let new = self.weight.val().clone().mul_scalar(m) + student.mul_scalar(1.0 - m);
        self.weight = Param::from_tensor(new);
        self.update_count = Param::from_tensor(self.update_count.val().clone().add_scalar(1.0));
    }

    pub fn val(&self) -> Tensor<B, 1> {
        self.weight.val().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use burn_ndarray::{NdArray, NdArrayDevice};
    type B = NdArray;
    fn dev() -> NdArrayDevice {
        NdArrayDevice::default()
    }
    fn as_scalar(t: Tensor<B, 1>) -> f32 {
        f32::from_le_bytes(t.into_data().bytes[..4].try_into().unwrap())
    }

    #[test]
    fn lejepa_finite() {
        let z = Tensor::<B, 2>::random([64, 32], Distribution::Default, &dev());
        assert!(as_scalar(lejepa_loss(z)).is_finite());
        let z2 = Tensor::<B, 2>::ones([32, 16], &dev());
        assert!(as_scalar(lejepa_loss(z2)).is_finite());
    }
    #[test]
    fn koleo_finite() {
        let z = Tensor::<B, 2>::random([128, 64], Distribution::Default, &dev());
        assert!(as_scalar(koleo_loss(z)).is_finite());
    }
    #[test]
    fn koleo_subsample() {
        let z = Tensor::<B, 2>::random([512, 16], Distribution::Default, &dev());
        assert!(as_scalar(koleo_loss(z)).is_finite());
    }
    #[test]
    fn ema_update() {
        let init = Tensor::<B, 1>::ones([4], &dev());
        let mut teacher = EmaTeacher::new(init, 0.9, &dev());
        let s = vec![2.0f32, 2.0, 2.0, 2.0];
        let student = Tensor::<B, 1>::from_floats(s.as_slice(), &dev());
        teacher.update(student);
        let vals: Vec<f32> = teacher
            .val()
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(
            (vals[0] - 1.1).abs() < 0.01,
            "EMA: {:.1}*0.9 + 2.0*0.1 = 1.1",
            vals[0]
        );
    }
}
