
# Image_captioner

## Introduction
This project focuses on developing a hybrid image captioning model that integrates InceptionV3 and LSTM networks. The goal is to generate meaningful captions for images by extracting image features and utilizing sequential text data.


## Project Overview
The Image Captioner project consists of the following main components:

- Data Preparation: Preprocessing the dataset of images and their corresponding captions.
- Model Architecture: Designing and implementing the neural network architecture for the image captioning task.
- Training and Evaluation: Training the model on the prepared dataset and evaluating its performance.
- Inference: Generating captions for new images using the trained model.
## Data Preprocessing
To prepare the dataset for the image captioning model, several preprocessing steps were performed. The steps include extracting image features, loading captions, cleaning captions, and saving the processed data. Here’s a detailed breakdown of the preprocessing pipeline:

### Utility Functions:

- mytime: A utility function to get the current timestamp, used for logging.
- extract_features: Extracts features from images using either InceptionV3 or VGG16 models.
- load_captions: Loads captions from a text file and structures them into a dictionary.
- clean_captions: Cleans the captions by removing punctuation, converting to lowercase, and removing non-alphabetic tokens.
- save_captions: Saves the cleaned captions to a file for later use.
### Feature Extraction:

- Depending on the model type (InceptionV3 or VGG16), images are loaded and resized to the appropriate target size.
- Images are then converted to numpy arrays, preprocessed, and passed through the chosen CNN model to extract features.
- The extracted features are stored in a dictionary where the keys are image IDs and the values are the feature vectors.
### Caption Processing:

- Captions are loaded from a text file. Each line in the file is split into an image ID and a caption.
- Captions are then cleaned by removing punctuation, converting to lowercase, and filtering out non-alphabetic tokens.
- The cleaned captions are saved to a file for later use.
### Preprocessing Function:

- The preprocessData function orchestrates the entire preprocessing pipeline.
- It checks if the features and captions have already been processed and saved. If not, it performs the necessary steps to extract features and process captions.
- This ensures that the preprocessing step is efficient and only performed when necessary.
## Data Loading

The data loading phase involves reading image and caption data, filtering and preprocessing them for model training. This includes loading specific training and validation datasets, creating sequences, and preparing tokenizers. Below are the key functions and their roles in the data loading process:

### Utility Functions:

- load_set: Reads a file containing image identifiers and returns a set of these IDs.
- load_cleaned_captions: Loads and cleans captions, wrapping them with startseq and endseq tokens.
- load_image_features: Loads precomputed image features and filters them based on provided IDs.
- to_lines: Converts a dictionary of captions into a list of caption strings.
- create_tokenizer: Fits a tokenizer on the given captions, creating a mapping from words to unique integer values.
- calc_max_length: Calculates the length of the longest caption in the dataset.
### Sequence Creation:

- create_sequences: Generates sequences of image features, input sequences, and output words from captions.
- data_generator: A generator function to create batches of data for model training, ensuring efficient memory usage and scalability.
### Data Loading Functions:

- loadTrainData: Loads and preprocesses the training data, including image features and captions, and prepares the tokenizer.
- loadValData: Loads and preprocesses the validation data, ensuring the availability of necessary image features and captions.
## Model Architecture
The model architecture for generating image captions is divided into two main components: a Convolutional Neural Network (CNN) for image feature extraction and a Recurrent Neural Network (RNN) for caption generation.

### CNN Component
The CNN component is responsible for extracting features from images. Two popular pre-trained models are used for this purpose: InceptionV3 and VGG16. These models are pre-trained on the ImageNet dataset and are fine-tuned to extract a fixed-size feature vector for each image. InceptionV3 outputs a 2048-dimensional vector, while VGG16 outputs a 4096-dimensional vector.

### RNN Component
The RNN component uses Long Short-Term Memory (LSTM) layers to process the sequence of words in the captions. The architecture includes:

- Image Input Layer: Takes the feature vector output from the CNN.
- Caption Input Layer: Takes the sequence of words in the caption.
- Embedding Layer: Converts words to dense vectors of fixed size.
- LSTM Layer: Processes the embedded word vectors.
- Dense Layers: Combines the image and caption features and predicts the next word in the sequence.




## Caption Generation

The caption generation process starts with a seed word ("startseq") and iteratively predicts the next word in the sequence until the end of the caption ("endseq") is reached. Two methods are used for this process:

- Greedy Search: Selects the word with the highest probability at each step.
- Beam Search: Explores multiple possible sequences at each step to find the most likely sequence. Beam search with a specified beam width (e.g., 3) is used to improve the quality of generated captions.
## Evaluation
The model is evaluated using BLEU scores, which measure the similarity between the generated captions and the actual captions. BLEU scores range from 0 to 1, with 1 indicating a perfect match. BLEU-1 to BLEU-4 scores are calculated, considering different n-gram precisions.
## Workflow

- Feature Extraction: The CNN component extracts features from each image.
- Caption Encoding: Captions are encoded into sequences of integers using a tokenizer.
- Model Training: The model is trained to predict the next word in the sequence given the previous words and the image features.
- Caption Generation: The trained model generates captions for new images using greedy or beam search.
- Evaluation: The generated captions are evaluated using BLEU scores to assess their quality.
This architecture efficiently combines image and text processing to generate coherent and contextually relevant captions for images.
## References
- [Show and Tell: A Neural Image Caption Generator](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)
- [How to Develop a Deep Learning Photo Caption Generator from Scratch](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)
