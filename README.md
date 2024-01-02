# Morphological Feature Extraction using OpenCV

This repository contains an application built on Streamlit for phenotyping and extracting morphological features using OpenCV.

## Introduction
Phenotyping is a process of measuring and analyzing observable traits (phenotypes) of an organism. Morphological feature extraction plays a crucial role in phenotyping, as it helps in quantifying and characterizing various physical traits of an organism.

This application leverages OpenCV, a popular computer vision library, to extract morphological features from images. With the help of a user-friendly interface built on Streamlit, users can easily upload images and visualize the extracted features.

## Features
- Upload images for morphological feature extraction
- Display the uploaded images and extracted features
- Interactive visualization of morphological features
- Export the extracted features as a CSV file

## Installation
To run the application locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies using the following command:
   ```
   pip install -r requirements.txt
   ```
3. Run the application by executing the following command:
   ```
   streamlit run app.py
   ```
4. Open your web browser and navigate to `http://localhost:8501` to access the application.

## Usage
1. Upload an image by clicking on the "Upload Image" button.
2. Wait for the image to be processed.
3. Once the image is processed, the uploaded image along with the extracted morphological features will be displayed.
4. Use the interactive visualization to explore the extracted features.
5. Click on the "Export Features" button to export the extracted features as a CSV file.

## Contributing
Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
- [Streamlit](https://www.streamlit.io/)
- [OpenCV](https://opencv.org/)
