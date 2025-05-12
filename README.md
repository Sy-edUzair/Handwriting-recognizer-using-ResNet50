# Project: Handwriting Writer Identification using Deep Learning

This project explores the use of deep learning models for writer identification based on handwritten text samples. It includes implementations of a Siamese Network and a ResNet50-based classifier to distinguish between different writers using the IAM Handwriting Database.



## Features

This project demonstrates writer identification using two different deep learning approaches applied to the IAM Handwriting Database.

**Common Features Across Both Notebooks:**

*   **Dataset:** Utilizes the IAM Handwriting Database, specifically focusing on a subset of images from the top writers.
*   **Data Preprocessing:** Includes steps for loading images, converting them to a suitable format (RGB), resizing, and normalizing pixel values.
*   **Writer Filtering:** The dataset is initially filtered to include samples from the top 50 writers, and then further refined to focus on the top 10 writers for model training and evaluation.
*   **Label Encoding:** Writer IDs are converted into numerical labels suitable for model training.
*   **Data Splitting:** The dataset is divided into training and testing sets to evaluate model performance.
*   **Deep Learning Framework:** Leverages TensorFlow and Keras for building, training, and evaluating the neural network models.
*   **Model Evaluation:** Performance is assessed using standard classification metrics on the test set.
*   **Visualization:** Includes code to display sample images from the dataset along with their corresponding writer labels.

**Siamese_Net.ipynb Specifics:**

*   **Model Architecture:** Implements a convolutional neural network (CNN) designed for writer identification. While named "Siamese_Net", the provided notebook trains this network as a direct classifier for writer identification among the top 10 writers. The architecture consists of several convolutional, batch normalization, max-pooling, and dense layers.
*   **Image Input Size:** Processes images resized to 128x128 pixels.
*   **Training:** The model is trained to classify handwriting samples into one of the top 10 writer categories.

**Resnet.ipynb Specifics:**

*   **Model Architecture:** Utilizes a pre-trained ResNet50 model as a feature extractor. The base ResNet50 model (with weights pre-trained on ImageNet) is followed by custom classification layers (GlobalAveragePooling2D, Dense layers) to adapt it for the writer identification task.
*   **Transfer Learning:** Employs transfer learning by leveraging the powerful feature extraction capabilities of ResNet50.
*   **Image Input Size:** Processes images resized to 224x224 pixels, as expected by the ResNet50 architecture, and uses its specific `preprocess_input` function.
*   **Training:** The model is fine-tuned or trained to classify handwriting samples from the top 10 writers.



## Dataset

This project utilizes the **IAM Handwriting Database**, a well-known resource for handwritten text recognition and writer identification research. The notebooks specifically use a subset of this database available via Kaggle: [IAM Handwriting Database - Top 50 Writers](https://www.kaggle.com/datasets/tejasreddy/iam-handwriting-top50).

**Key aspects of dataset usage in this project:**

*   **Source:** The data is downloaded using the `kagglehub` library from the path `tejasreddy/iam-handwriting-top50`.
*   **Content:** The dataset consists of images of handwritten English text.
*   **Structure:** The images are located in the `data_subset/data_subset` directory within the downloaded dataset. A file named `forms_for_parsing.txt` is used to map form IDs (derived from image filenames) to writer IDs.
*   **Filtering:**
    *   Initially, the project considers data from the top 50 writers available in this subset.
    *   For the model training and evaluation in both notebooks, the data is further filtered to include only samples from the **top 10 most frequent writers**. This is done to create a more focused classification task.
*   **Image Preprocessing:**
    *   Images are read using OpenCV (`cv2`).
    *   Converted from BGR to RGB color format.
    *   Resized to a target dimension (128x128 for `Siamese_Net.ipynb` and 224x224 for `Resnet.ipynb`).
    *   Pixel values are normalized (divided by 255.0 for `Siamese_Net.ipynb`). The `Resnet.ipynb` uses the specific `preprocess_input` function from `tensorflow.keras.applications.resnet50` which handles its own normalization and scaling suitable for the ResNet50 model.
*   **Labeling:** Writer IDs serve as the labels for the classification task. These are encoded into numerical format using `sklearn.preprocessing.LabelEncoder` and then converted to categorical format using `tensorflow.keras.utils.to_categorical`.

## Models

This project implements and evaluates two distinct deep learning models for the task of writer identification on the IAM Handwriting Database. Both models are trained to classify handwriting samples from the top 10 writers identified in the dataset.

### 1. ResNet50 based embeddings generator and binary classifier (from `Siamese_Net.ipynb`)

*   **Objective:** To classify a given handwriting sample to one of the 10 selected writers.
*   **Architecture:** ResNet50 with global pooling and dense layers to produce embeddings.
*   **Training:**
    *   Optimizer: Adam (`optimizer=\'adam\'`).
    *   Loss Function: Contrastive Loss (`loss=\'contrastive_loss()\'`).
    *   Metrics: Accuracy (`metrics=[\'accuracy\']`).
    *   Callbacks: `EarlyStopping` (to prevent overfitting by stopping training if validation loss doesn\'t improve) and `ReduceLROnPlateau` (to reduce learning rate if validation loss plateaus).
*   **Input Image Size:** 128x128 pixels (RGB).

### 2. ResNet50-based Classifier (from `Resnet.ipynb`)

This notebook utilizes a pre-trained ResNet50 model, a very deep convolutional network, leveraging transfer learning for the writer identification task.

*   **Objective:** To classify a given handwriting sample to one of the 10 selected writers using features extracted by ResNet50.
*   **Architecture:**
    *   **Base Model:** `ResNet50` pre-trained on the ImageNet dataset. The top (classification) layer of the ResNet50 model is excluded (`include_top=False`).
    *   **Input Layer:** Expects images of size (224, 224, 3), which is standard for ResNet50.
    *   **Custom Head:** The output of the ResNet50 base is then fed into custom layers for classification:
        *   `GlobalAveragePooling2D()`: To reduce the spatial dimensions of the feature maps.
        *   `Dense(1024, activation=\'relu\')`: A fully connected layer.
        *   `Dropout(0.5)`: For regularization.
        *   `Dense(num_classes, activation=\'softmax\')`: The output layer, where `num_classes` is 10.
    *   **Trainable Layers:** The notebook sets the layers of the base ResNet50 model to be non-trainable initially (`base_model.trainable = False`) to leverage the pre-trained weights, and only the custom head is trained. It\'s a common practice to later fine-tune some of the later layers of the base model, though the notebook snippet doesn\'t explicitly show this fine-tuning step being activated.
*   **Training:**
    *   Optimizer: Adam (`optimizer=\'adam\'`).
    *   Loss Function: Categorical Crossentropy (`loss=\'categorical_crossentropy\'`).
    *   Metrics: Accuracy (`metrics=[\'accuracy\']`).
    *   Callbacks: `EarlyStopping` and `ReduceLROnPlateau`.
*   **Input Image Size:** 224x224 pixels (RGB), preprocessed using `tensorflow.keras.applications.resnet50.preprocess_input`.

Both models aim to achieve the same goal of writer identification but employ different architectural strategies: one building a CNN from scratch and the other using a powerful pre-trained model via transfer learning.

## Requirements

This project relies on several Python libraries for data processing, model building, and visualization. The key dependencies are listed below. Both notebooks (`Siamese_Net.ipynb` and `Resnet.ipynb`) share the same set of core requirements.

*   **Python 3.x**
*   **TensorFlow** (version 2.18.0 as per notebook output, though a compatible version should work)
*   **NumPy** (version 2.0.2 as per notebook output)
*   **Pandas** (version 2.2.2 as per notebook output)
*   **OpenCV-Python (`opencv-python`)** (version 4.11.0.86 as per notebook output)
*   **Matplotlib** (version 3.10.0 as per notebook output)
*   **Scikit-learn (`scikit-learn`)** (version 1.6.1 as per notebook output)
*   **KaggleHub (`kagglehub`)** (version 0.3.12 as per notebook output, for downloading the dataset)
*   **Tqdm** (for progress bars)
*   **Collections** (Python built-in)
*   **OS** (Python built-in)
*   **Random** (Python built-in)

These libraries are typically installed using `pip`. The first cell in both notebooks includes the command `!pip install tensorflow numpy pandas opencv-python matplotlib scikit-learn kagglehub`, which attempts to install or verify these core packages.


## Installation

To set up the environment and run this project, follow these steps:

1.  **Clone the Repository (if applicable):**
    If this project is part of a Git repository, clone it to your local machine:
    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2.  **Set up a Python Environment:**
    It is highly recommended to use a virtual environment (e.g., `venv` or `conda`) to manage dependencies.

    *   Using `venv`:
        ```bash
        python3 -m venv venv
        source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
        ```
    *   Using `conda`:
        ```bash
        conda create -n writer_id_env python=3.9  # Or your preferred Python 3.x version
        conda activate writer_id_env
        ```

3.  **Install Dependencies:**
    The primary dependencies are listed in the `Requirements` section. You can install them using `pip`. The notebooks themselves contain a cell to install these libraries:
    ```python
    !pip install tensorflow numpy pandas opencv-python matplotlib scikit-learn kagglehub
    ```
    Run this command in your terminal (if not using a Jupyter notebook directly for installation) or ensure this cell is executed in the notebooks.

4.  **Download the Dataset:**
    The notebooks use `kagglehub` to download the IAM Handwriting Database subset. Ensure you have Kaggle API credentials set up if required by `kagglehub` for direct downloads, or manually download the dataset from [IAM Handwriting Database - Top 50 Writers](https://www.kaggle.com/datasets/tejasreddy/iam-handwriting-top50) and place it in the expected path (`/kaggle/input/iam-handwriting-top50` as referenced in the notebooks, or adjust the paths in the notebooks: `image_dir` and `txt_path`).

    The notebooks execute the following to download the data:
    ```python
    import kagglehub
    path = kagglehub.dataset_download("tejasreddy/iam-handwriting-top50")
    # The default download path might be ~/.cache/kagglehub/datasets/tejasreddy/iam-handwriting-top50
    # The notebooks then assume paths like 	'/kaggle/input/iam-handwriting-top50/...	'
    # You may need to adjust these paths or symlink the downloaded data if running locally outside Kaggle.
    ```
    If running locally and not on a platform like Kaggle or Colab where `/kaggle/input/` is standard, you will need to modify the `image_dir` and `txt_path` variables in the notebooks to point to the correct location of the downloaded dataset files.
    For example, if `kagglehub.dataset_download` downloads to `~/.cache/kagglehub/datasets/tejasreddy/iam-handwriting-top50`, then:
    `image_dir = 	'~/.cache/kagglehub/datasets/tejasreddy/iam-handwriting-top50/data_subset/data_subset	'`
    `txt_path = 	'~/.cache/kagglehub/datasets/tejasreddy/iam-handwriting-top50/forms_for_parsing.txt	'`
    (Remember to expand `~` to your actual home directory path if Python	's `os.path.join` doesn	't do it automatically in this context).

5.  **Jupyter Notebook Environment:**
    If you don	't have Jupyter Notebook or JupyterLab installed:
    ```bash
    pip install notebook jupyterlab
    ```
    Then, you can launch JupyterLab or Jupyter Notebook:
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
    Navigate to the directory containing the `.ipynb` files and open them.

## Usage

After completing the installation steps, you can run the project notebooks:

1.  **Launch Jupyter Environment:**
    Open your terminal, navigate to the project directory, activate your virtual environment, and start Jupyter Lab or Jupyter Notebook:
    ```bash
    # If you used venv
    source venv/bin/activate
    # If you used conda
    conda activate writer_id_env

    # Launch Jupyter
    jupyter lab
    # or
    jupyter notebook
    ```

2.  **Open the Notebooks:**
    In the Jupyter interface, navigate to and open either `Siamese_Net.ipynb` or `Resnet.ipynb`.

3.  **Run the Cells:**
    Execute the cells in the notebook sequentially. Key steps in each notebook include:
    *   **Dependency Installation:** The first cell installs necessary Python packages.
    *   **Import Libraries:** Imports all required modules.
    *   **Dataset Download:** Downloads the IAM Handwriting dataset using `kagglehub`. Ensure your environment can access the internet and, if necessary, that Kaggle API credentials are configured or the dataset is manually placed and paths are adjusted as noted in the Installation section.
    *   **Path Configuration:** Sets paths for the image directory and the forms parsing file. **Important:** If you are not running on a Kaggle environment, you will likely need to modify the `image_dir` and `txt_path` variables to point to the actual location of your dataset files after downloading them (see Installation section for details).
    *   **Image Preprocessing:** Defines functions to load, resize, and preprocess images.
    *   **Data Loading and Filtering:** Loads image data and labels, maps forms to writers, and filters the dataset to the top 10 writers.
    *   **Label Encoding:** Converts writer labels to a numerical format suitable for the model.
    *   **Data Splitting:** Splits the data into training and testing sets.
    *   **Model Definition:** Defines the neural network architecture (Custom CNN in `Siamese_Net.ipynb` or ResNet50-based in `Resnet.ipynb`).
    *   **Model Compilation:** Configures the model for training (optimizer, loss function, metrics).
    *   **Model Training:** Trains the model on the training data. This step can be computationally intensive and may take some time, especially if running on a CPU. Both notebooks are configured to run on a GPU if available (as indicated by Colab metadata).
    *   **Model Evaluation:** Evaluates the trained model on the test set and prints metrics like accuracy and loss.
    *   **Visualization:** Displays some sample images with their labels.

**Expected Output:**

*   The notebooks will output progress during data loading, model training (epochs, loss, accuracy), and final evaluation metrics.
*   Plots of sample images and potentially training history (if added) will be displayed.

**Note on Execution Time and Resources:**

*   Training deep learning models can be time-consuming, especially without a GPU. The notebooks were originally run in an environment with GPU acceleration.
*   Ensure you have sufficient disk space for the dataset and Python environment.

## Contributing

Contributions to this project are welcome. If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request if this project is hosted on a collaborative platform like GitHub.
For major changes, please open an issue first to discuss what you would like to change.

## License

This project is provided as is. Please refer to the license file in the repository if one is provided. If no license is specified, assume the code is for educational and demonstration purposes and standard copyright laws apply.


