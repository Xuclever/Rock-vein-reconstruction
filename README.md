# Rock vein-Reconstruction

### What is this repository for?

 本项目用于断裂岩脉重构，包括预处理（形态学、骨架化），岩脉重构。/This project is used for fracture and rock vein reconstruction, including pre-processing (morphology, skeleton), and rock vein reconstruction.

### How do I get set up?

Copy this folder to a win64 system and install Anaconda and Python 3 environment on the system.

### Documentation

#### Install

Install the libraries required for this project, which utilizes the Python environment and PyTorch.

```python
pip install -r requirements.txt
```

#### Usage

Before using this project, you need to open Anaconda and activate the environment.

```python
conda create -n rock-vein-reconstruction python=3.10
activate rock-vein-reconstruction
```

After activating the environment, you can run code on the command line.

```
python binarization.py
python crack_aco_reconstruction.py
```

#### Example

Input the imgs file from the example folder into the crack_aco_reconstruction.py code, and set the parameters as follows: "max_gap": 150, "min_gap": 10, "ant_count": 30, "max_iter": 50. Keep the remaining parameters unchanged.

```python
python crack_aco_reconstruction.py
```

