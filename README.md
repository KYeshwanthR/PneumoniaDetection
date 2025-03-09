# Pneumonia Detection

This project uses a pre-trained EfficientNet model to detect pneumonia from X-ray images. The model is deployed using Streamlit for a web-based interface.

## Requirements

Make sure you have the required packages installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the application, use the following command:

```bash
streamlit run main.py
```

This will start a local web server. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

1. Upload an X-ray image using the file uploader.
2. The application will analyze the image and display the prediction (Normal or Pneumonia).
3. A Grad-CAM visualization will also be displayed to show the regions of the image that contributed to the prediction.

## Running the Notebook in Google Colab

1.  Open the `ModelCode.ipynb` notebook in Google Colab.
2.  Change the runtime to GPU with T4. Go to `Runtime` -> `Change runtime type` -> `Hardware accelerator` and select `GPU`. Make sure that the GPU type is T4.

