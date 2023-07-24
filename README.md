# PES Edit Face AI

This project aims to provide a face-making tool for customizing player faces in PES (Pro Evolution Soccer) game based on an image, using deep learning.

The project utilizes deep learning techniques to extract facial features from images and create embeddings (representations) of player faces.

These embeddings are then used to train a custom deep learning model capable of predicting various facial attributes (e.g "SkinColour", "UpperEyelidType", "BottomEyelidType", "FacialHairType", etc.) from the extracted facial embeddings.

## How It Works

- **Face Encoding**: The FaceEncoder class handles the process of encoding faces from images. It utilizes a deep learning model (e.g., "Facenet") to extract facial embeddings from the images. The embeddings are then used for facial attribute prediction.

- **Custom Deep Learning Model**: The project includes a custom deep learning model designed to predict facial attributes from facial embeddings. The model uses the extracted embeddings as input and predicts various facial attributes like "SkinColour," "UpperEyelidType," "BottomEyelidType," and "FacialHairType" as output.

- **Training the Model**: The project provides functionality to train the custom deep learning model using prepared training data. The data consists of facial embeddings and corresponding output labels for the facial attributes. The model is trained with a custom loss function called custom_loss to optimize the predictions.

- **Face Customization in PES**: With the trained model and facial embeddings, players can use the tool to customize player faces within the "PES" soccer video game. Instead of manually adjusting sliders and values in the game's Edit Menu, the tool automatically generates face attributes based on the provided facial embeddings.

## Previews

![Preview1](/Images/preview%20(1).jpg)
![Preview2](/Images/preview%20(2).jpg)
![Preview3](/Images/preview%20(3).jpg)
![Preview4](/Images/preview%20(4).jpg)
