# Translator English to Vietnamese with Attention

This project uses a Seq2Seq model with Luong Attention mechanism to translate sentences from English to Vietnamese.

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment and install the required packages:
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. Download the pre-trained models and place them in the `models/` directory.

## Usage

### Run the Web Application

1. Run the Flask application:
    ```sh
    python app.py
    ```

2. Open your browser and go to `http://127.0.0.1:5000`.

### Train the Model

1. Open and run the notebooks in the `training/` directory to train the model:
    - `training_seq2seq_and_attention.ipynb`
    - `translator_luong_attention.ipynb`
    - `translator_no_attention.ipynb`

## Code Structure

- `app.py`: Contains the Flask application code and model classes such as `LuongAttention`, `Encoder`, and `Decoder`.
- `dataset/`: Contains the training data files.
- `models/`: Contains the pre-trained models and tokenizer.
- `static/`: Contains CSS files.
- `templates/`: Contains HTML files.
- `training/`: Contains Jupyter notebooks for training the model.
