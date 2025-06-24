# Fashion-MNIST Deep Learning Pipelines

This repository contains two modular deep learning pipelines built with **PyTorch**, demonstrating practical applications of unsupervised and supervised learning techniques on the **Fashion-MNIST** dataset.

- **Autoencoder Pipeline**: A denoising autoencoder designed for robust latent-space learning, enabling both image reconstruction and synthetic sample generation.
- **Classification Pipeline**: A regularized feedforward neural network optimized for high-accuracy image classification through techniques such as Batch Normalization and Dropout.

### Key Technologies & Practices

- **Framework**: PyTorch (including native modules such as `nn.Sequential`, `DataLoader`, and GPU support via `torch.device`)
- **Optimization**: Adam optimizer with StepLR scheduler for controlled learning rate decay
- **Regularization**: Gaussian noise injection, Dropout layers, and Batch Normalization for improved generalization
- **Data Handling**:
  - Custom `Dataset` subclass for reading raw `.gz` Fashion-MNIST files directly (bypassing torchvision APIs)
  - Manual normalization and reshaping to match network expectations
- **Visualization & Evaluation**:
  - Matplotlib-based reconstruction and generation plotting
  - Seaborn-driven confusion matrix visualization
  - Misclassification analysis for failure case diagnostics

These pipelines reflect real-world deep learning workflows in rapid prototyping and academic-industrial research environments. They are designed with clarity, reproducibility, and performance in mind.