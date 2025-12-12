# Adaptive Token Selection for Scalable Point Cloud Transformers

Welcome to the AdaPT Repository. This repository contains a deep learning model for point cloud classification using transformers and adaptive token dropping.

```bash
@article{baiocchi2025adapt,
title = {Adaptive token selection for scalable point cloud transformers},
journal = {Neural Networks},
year = {2025},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2025.107477},
author = {Alessandro Baiocchi and Indro Spinelli and Alessandro Nicolosi and Simone Scardapane},
}
```

## Installation

To get started with this project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/ispamm/adaPT
   ```

2. Navigate to the project directory:

   ```bash
   cd adaPT
   ```

3. Create a Conda environment from the provided `Adapt_env.yaml` file:

   ```bash
   conda env create -f Adapt_env.yaml
   ```

4. Activate the Conda environment:

   ```bash
   conda activate Adapt_env
   ```

## Usage

Once you have set up the Conda environment, you can run the code using the following steps:

1. Ensure you have activated the Conda environment as mentioned in the Installation section.

2. Run the main script:

   ```bash
   python main.py
   ```

   Replace `main.py` with the name of the main script in your project.

## Hyperparameters

This project uses Hydra for hyperparameter management. Edit the `config.yaml` file to change the desired parameters.


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or create a pull request. For major changes, please open an issue first to discuss the proposed changes.
