# burn-jepa - Self-Supervised Losses for Burn

[![CI](https://github.com/sehaxe/burn-jepa/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-jepa/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-jepa)](https://crates.io/crates/burn-jepa)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.21-orange.svg)](https://burn.dev)

Self-supervised losses for JEPA (Joint Embedding Predictive Architecture) from Meta AI.
LeJEPA isotropic Gaussian regularization + KoLeo uniformity.

> Papers:
> [LeJEPA](https://arxiv.org/abs/2511.08544) (Balestriero & LeCun, 2025),
> [DINOv2](https://arxiv.org/abs/2304.07193) (Caron et al., 2023),
> [DINO](https://arxiv.org/abs/2104.14294) (Caron et al., 2021).

## Install

```bash
cargo add burn-jepa
```

## Quick start

```rust
use burn_jepa::{lejepa_loss, koleo_loss, EmaTeacher};

// Constrain embeddings toward isotropic Gaussian
let reg = lejepa_loss(embeddings);

// Encourage uniform spread on unit sphere
let uni = koleo_loss(embeddings);

// EMA teacher tracking
let mut teacher = EmaTeacher::new(student_weights, 0.996, &device);
teacher.update(latest_student);
```

## API

| Export | What |
|--------|------|
| `lejepa_loss(z)` | Isotropic Gaussian: mean + cov-I / D |
| `koleo_loss(z)` | KoLeo uniformity: -mean(log(min_dist)) |
| `EmaTeacher` | Momentum EMA: teacher = 0.996 * teacher + 0.004 * student |

## License

AGPL-3.0. See [LICENSE](LICENSE).
