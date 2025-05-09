# HealthFactFinder: Medical Misinformation Detection System

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9%2B-orange)](https://www.tensorflow.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/docs/transformers/index)

## Overview

HealthFactFinder is an advanced medical misinformation detection system that combines multiple AI models to verify medical claims with high accuracy and confidence. The system uses a confidence-weighted ensemble approach that integrates vector similarity search, CNN-BiLSTM deep learning models, and PubMedBERT for robust fact verification.

## Key Features

- **Confidence-Weighted Ensemble System** - Combines multiple models for superior verification accuracy
- **Vector Search Engine** - Efficiently retrieves relevant medical literature
- **CNN-BiLSTM Model** - Achieves 85% accuracy for medical claim classification
- **PubMedBERT Integration** - Leverages domain-specific language understanding
- **Probabilistic Confidence Scoring** - Adaptive thresholds improve verification precision from 78% to 92%
- **Explainable Results** - Provides reasoning and evidence for verification decisions

## Installation

```bash
# Clone repository
git clone https://github.com/33Surya66/HealthFactFinder.git
cd HealthFactFinder

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python download_models.py
```

## Usage

### Command Line Interface

```bash
# Verify a single medical claim
python verify_claim.py --claim "Regular consumption of vitamin C prevents the common cold"

# Batch verification from file
python verify_claims.py --input claims.txt --output results.json

# Run with custom confidence threshold
python verify_claim.py --claim "Drinking lemon water detoxifies the liver" --threshold 0.85
```

## System Architecture

```
              ┌─────────────────┐
              │   User Input    │
              │  Medical Claim  │
              └────────┬────────┘
                       │
                       ▼
┌───────────────────────────────────────────┐
│        Ensemble Verification System       │
├───────────────┬───────────────┬───────────┤
│ Vector Search │  CNN-BiLSTM   │ PubMedBERT│
│    Engine     │    Model      │   Model   │
└───────┬───────┴───────┬───────┴─────┬─────┘
        │               │             │
        ▼               ▼             ▼
┌───────────────────────────────────────────┐
│      Confidence Weighting Algorithm       │
└───────────────────────┬───────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────┐
│       Probabilistic Confidence Score      │
└───────────────────────┬───────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────┐
│              Final Verdict               │
│     (Supported/Unsupported/Misleading)    │
└───────────────────────────────────────────┘
```

## Model Details

### Vector Search Engine
- **Purpose**: Retrieves relevant medical literature based on semantic similarity
- **Implementation**: Uses FAISS or similar vector database
- **Corpus**: Indexed PubMed abstracts and medical guidelines

### CNN-BiLSTM Model
- **Architecture**: Convolutional layers for feature extraction followed by BiLSTM for sequence learning
- **Accuracy**: 85% on benchmark datasets
- **Features**: Captures both local patterns and long-range dependencies in text

### PubMedBERT
- **Base Model**: Domain-specific BERT variant trained on medical literature
- **Fine-tuning**: Adapted for fact verification tasks
- **Strength**: Strong understanding of medical terminology and concepts

## Performance Metrics

| Metric    | Score |
|-----------|-------|
| Precision | 92%   |
| Recall    | 87%   |
| F1 Score  | 89%   |
| Accuracy  | 85%   |

## Contributing

We welcome contributions to improve HealthFactFinder! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Dataset

The system is trained and evaluated on a combination of:
- MedFact dataset (custom annotated medical claims)
- FEVER dataset (adapted for medical domain)
- PubMed abstracts
- Medical guidelines from WHO, CDC, and NIH

## Future Work

- Integration with real-time news monitoring systems
- Mobile application for on-the-go fact checking
- API services for third-party integration
- Enhanced explanation generation
- Support for multiple languages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use HealthFactFinder in your research, please cite:

```
@software{ravi2025healthfactfinder,
  author = {Ravi Surya Chandra Prakash, Swayam Khatri, Renish Thesiya, Sushrut Nisal},
  title = {HealthFactFinder: Medical Misinformation Detection System},
  year = {2025},
  url = {https://github.com/33Surya66/HealthFactFinder}
}
```

## Contact

Surya Chandra Prakash Ravi - [surya03.ravi@gmail.com](mailto:surya03.ravi@gmail.com)
