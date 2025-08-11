# Protein A-like Peptide Generation Based on a Generalized Diffusion Model

This repository contains the code for the paper "Protein A-like Peptide Generation Based on a Generalized Diffusion Model". The core functionality is implemented in a Jupyter Notebook (`GDM.ipynb`).

## Requirements

Install the required dependencies to run the code:
- Python 3.8+
- PyTorch 1.10+
- NumPy
- pandas
- matplotlib
- Jupyter Notebook
- torchgeometry
- einops
- tqdm
- comet_ml (optional, for experiment tracking)

Install via pip:pip install torch numpy pandas matplotlib jupyter torchgeometry einops tqdm comet_ml

## Quick Start

The main entry point is `GDM.ipynb`. Follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ProteinGeneration-GeneralizedDiffusion
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open `GDM.ipynb` and run all cells sequentially. This will:
   - Handle data loading and preprocessing
   - Initialize the diffusion model
   - Support training (from scratch) or loading pre-trained checkpoints
   - Generate Protein A-like peptides
   - Include basic evaluation and visualization of results


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation

If you use this code in your research, please cite our paper

## Contact

For questions, please contact us or open an issue in the repository.
