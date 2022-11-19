# Dataset
Most of the dataset(3996 images) was parsed by me from Google search and cleaned: [link](https://www.kaggle.com/datasets/erikgevondyan/smiling-or-not).

But this seemed to me insufficient, so I found another dataset in addition to this(1203 images): [link](https://www.kaggle.com/datasets/chazzer/smiling-or-not-face-data).
<br />
<br />
# Model
[Resnet-50](https://arxiv.org/pdf/1512.03385.pdf) pre-trained on [ImageNet](https://arxiv.org/pdf/1409.0575.pdf).

|        | TRAIN |  VAL  | TEST  |
|-------:|------:| -----:|------:|
|ACCURACY| 83.76 | 85.83 | 87.81 |
|LOSS    | 0.365 | 0.330 | 0.333 |

[Model statement](https://www.kaggle.com/datasets/erikgevondyan/smiling-or-not-model).
<br />
<h1>Deployment</h1>
HTML + CSS + Flask + Docker + VPS.

[Docker Image](https://hub.docker.com/repository/docker/gevondyanerik/smiling_or_not).
