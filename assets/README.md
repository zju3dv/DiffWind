# Released Gaussian assets

All PLY files used by the public examples are stored with repository-relative
paths. The `model.ply` files contain Gaussian Splatting attributes; `filling.ply`
files contain internal MPM particles and are not rendered.

`cameras.json` stores the eight input-independent camera transforms used by
scenes whose original views are not orbit cameras.

| Scene | Surface Gaussians | Filling particles | Public use |
| --- | ---: | ---: | --- |
| Ficus | 203,930 | — | multiview + forward |
| Sweater | 52,601 | 17,533 | multiview + forward |
| Potted Plant | 27,842 | 9,280 | multiview + forward |
| Pants | 19,004 | 6,334 | multiview + forward |
| Alocasia | 329,460 | 15,000 | multiview + forward |
| Hat | 297,402 | — | multiview + forward |
| Dress | 67,171 | 23,016 | multiview + forward |
| Flag | 356,179 | — | multiview + forward |
