# Custom CycleGAN

**Custom CycleGAN** is a pure and concise TensorFlow 2 implementation of the *Cycle-Consistent Adversarial Networks* paper : https://arxiv.org/pdf/1703.10593.pdf.

# Prerequisites

A ready to use Conda package manager. <br>
You may be interested by [Miniconda](<https://conda.io/en/latest/miniconda.html>) and the minimal installation it requires.

# Install

```bash
git clone git@github.com:neuronest/custom-cyclegan.git
cd custom-cyclegan
source install.sh
```

# Run TensorBoard server

```bash
tensorboard --port 6006 --logdir tensorboard
```

Note: TensorBoard internally restricts to **10** the number of distinct images displayed per tag. <br>
You can work around this limitation by adding the `--samples_per_plugin images=n` argument, with `n` a higher number, analogous to the maximum number of epochs. <br>
For example: 
```bash
tensorboard --port 6006 --logdir tensorboard --samples_per_plugin images=500
```

# Run a training experiment

```bash
python -m src.main
```
