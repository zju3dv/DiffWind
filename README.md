# [ICLR 2026] DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics


### [Project Page](https://zju3dv.github.io/DiffWind/) | [OpenReview](https://openreview.net/forum?id=vKVzihkbQo) | [Arxiv (Coming soon)]() | [Supplementary (Coming soon)]()

> [DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics](https://zju3dv.github.io/DiffWind/),  
> Yuanhang Lei*, Boming Zhao*, Zesong Yang*, Xingxuan Li, Tao Cheng, Haocheng Peng, Ru Zhang, Yang Yang, Siyuan Huang, Yujun Shen, Ruizhen Hu, Hujun Bao, Zhaopeng Cui†

![teaser](https://raw.githubusercontent.com/huahuo359/open_access_assets/main/DiffWind-ICLR2026/teaser.png)
Abstract: *Modeling wind-driven object dynamics from video observations is highly challenging due to the invisibility and spatio-temporal variability of wind, as well as the complex deformations of objects. We present DiffWind, a physics-informed differentiable framework that unifies wind-object interaction modeling, video-based reconstruction, and forward simulation. Specifically, we represent wind as a grid-based physical field and objects as particle systems derived from 3D Gaussian Splatting, with
their interaction modeled by the Material Point Method (MPM). To recover wind-driven object dynamics, we introduce a reconstruction framework that jointly optimizes the spatio-temporal wind force field and object motion through differentiable rendering and simulation. To ensure physical validity, we incorporate the Lattice Boltzmann Method (LBM) as a physics-informed constraint, enforcing compliance with fluid dynamics laws. Beyond reconstruction, our method naturally supports forward simulation under novel wind conditions and enable new applications such as wind retargeting. We further introduce WD-Objects, a dataset of synthetic and real-world wind-driven scenes. Extensive experiments demonstrate that our method significantly outperforms prior dynamic scene modeling approaches in both reconstruction accuracy and simulation fidelity, opening a new avenue for video-based wind–object interaction modeling.*

## Method Overview
![pipeline](https://raw.githubusercontent.com/huahuo359/open_access_assets/main/DiffWind-ICLR2026/pipeline.png)

## ToDos
🔥 Feel free to raise any requests~
- [x] Release project page.
- [ ] Release paper.
- [ ] Release simulation codes.
- [ ] Release training codes.

### Citation
If you find this work useful for your research, please use the following BibTeX entry.
```
@inproceedings{lei2026diffwind,
    title     = {DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics},
    author = {Lei, Yuanhang and Zhao, Boming and Yang, Zesong and Li, Xingxuan and Cheng, Tao and Peng, Haocheng and Zhang, Ru and Yang, Yang and Huang, Siyuan and Shen, Yujun and Hu, Ruizhen and Bao, Hujun and Cui, Zhaopeng},
    booktitle = {Proceedings of the Fourteenth International Conference on Learning Representations (ICLR)},
    year      = {2026},
    url       = {https://openreview.net/forum?id=vKVzihkbQo}
}
```
