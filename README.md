# Potato Disease Classification

## Overview

This repository contains the code and resources for a potato disease classification project. The objective is to develop a machine learning model capable of accurately identifying diseases affecting potato crops. The model aims to provide a practical tool for farmers to quickly diagnose and manage potato diseases.

## Dataset

The dataset used for this project was sourced from Kaggle and includes images of both healthy and diseased potato plants. The dataset posed various challenges, such as diverse lighting conditions, different disease stages, and potential noise in the images.

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis was crucial to understanding the distribution of healthy and diseased samples. Visualizations provided insights into the diversity of the dataset, guiding subsequent preprocessing steps.

## Data Preprocessing

Rigorous data preprocessing techniques were implemented to address challenges related to image quality, resolution disparities, and class imbalances. These steps were vital for ensuring the model's robustness in real-world conditions.

## Modeling

The classification task was approached using a deep learning model with transfer learning, utilizing a pre-trained convolutional neural network (CNN). The model demonstrated promise during training, capturing intricate patterns associated with different diseases.

## Evaluation

The model achieved an impressive accuracy of 96% on the test dataset, highlighting its capability to accurately classify healthy and diseased potato plants. Evaluation metrics, including precision, recall, and F1 score, further validate the model's performance.

## Challenges

Throughout the project, challenges related to data quality and model interpretability were encountered. Fine-tuning the model to handle subtle variations in disease symptoms required careful consideration and parameter tuning.

## Deployment

The next steps involve deploying the model on Google Cloud to make it accessible for broader use. Additionally, an Android application is in development to integrate the model, allowing farmers to conveniently diagnose potato diseases using their mobile devices.

## Acknowledgments

Special thanks to the Kaggle community for providing the dataset and valuable insights. The collaborative nature of the platform significantly contributed to the success of this project.

## Getting Started

To explore the project, follow these steps:

1. Clone this repository: `git clone [https://github.com/your-username/potato-disease-classification](https://github.com/DreamIsMl/potato-leaf-classification).git`
2. Install dependencies: `pip install -r requirements.txt`
3. Explore the notebooks in the `notebooks/` directory for detailed insights into data analysis and model development.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For inquiries or collaborations, feel free to contact [Hakim] at [azizulhakim8291@gmail.com].
