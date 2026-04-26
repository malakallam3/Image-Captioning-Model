# Image Captioning Model

An end-to-end deep learning project for automatic image caption generation using the Flickr8k dataset. This project combines a CNN-based feature extractor (Xception) with an LSTM-based sequence model to generate descriptive captions for images.

## Features

- Image preprocessing and caption cleaning
- Vocabulary creation and tokenizer generation
- Feature extraction using Xception (Transfer Learning)
- Sequence generation for training
- Encoder-Decoder architecture for image captioning
- Model training and checkpoint saving
- Automatic handling of missing dependencies and dataset paths

## Model Architecture

The model follows an Encoder-Decoder framework:

### Encoder (CNN)
- Pretrained Xception model
- Removes classification head (`include_top=False`)
- Extracts 2048-dimensional image feature vectors

### Decoder (LSTM)
- Embedding layer for word representations
- LSTM for sequence learning
- Dense layers for next-word prediction
- Softmax output over the vocabulary

## Dataset

This project uses the **Flickr8k** dataset:

- 8,000 images
- 5 captions per image
- Training split defined by:
  - `Flickr_8k.trainImages.txt`

Expected directory structure:

```bash
project/
│
├── Flickr8k_text/
│   ├── Flickr8k.token.txt
│   └── Flickr_8k.trainImages.txt
│
├── Flickr8k_Dataset/        # or Flicker8k_Dataset/
│   ├── image1.jpg
│   └── ...
│
└── main.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/malakallam3/image-captioning-model.git
cd image-captioning-model
```

Install dependencies:

```bash
pip install tensorflow keras pillow numpy tqdm
```

## Running the Project

Run the training script:

```bash
python main.py
```

The script will:

1. Clean and process captions  
2. Build the vocabulary  
3. Extract image features  
4. Create training sequences  
5. Train the model  
6. Save checkpoints in:

```bash
models/
```

## Output Files

Generated during execution:

- `descriptions.txt` — cleaned captions  
- `features.p` — extracted image features  
- `tokenizer.p` — saved tokenizer  
- `models/model_*.h5` — trained model checkpoints

## Training Parameters

Current default settings:

- Epochs: 10  
- Steps per epoch: 5  
- Batch size: 32  

You can modify these in `main.py`.

## Technologies Used

- Python
- TensorFlow / Keras
- Xception
- LSTM
- NumPy
- Pillow


## License

This project is for educational and research purposes.
