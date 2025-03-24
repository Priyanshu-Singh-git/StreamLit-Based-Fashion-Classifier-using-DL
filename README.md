# StreamLit Based Fashion Product Classifier Using DL
## Overview -
Fashion Product Classifier is a model which will Predict the Type of the product , Recommended Season to use it , for which gender it is preferrable , colour of the product.
This model is solely DL based rather than the Using Clustering. This Model is on sample 20k images from a larger dataset containing 44K( you can train it on more dataset).
## Dataset used - 
The link for the dataset used in this is : https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
## How to Run - 
- First clone the Repo 
- Install required modules 
- run cmd on same path
- Run command `streamlit run fashionapp.py`
- it will forward you to browser 
- Upload the image and enjoy the predictions
- **Important Note** - Make sure you run the command in command prompt on this project folder path and also make this folder's path python's current working directory. 
### How to train the Model for more epoch :
- The Code is given in ipynb just change the path variables according to where you want to save the weight
- **Note**- The Notebook Contains Detailed Explanation of every cell and Methods done
### Results and Metric : 
#### Last Epoch Metrics

| Metric       | Value  |
|-------------|--------|
| Loss        | 0.1463 |
| Color Acc   | 0.9817 |
| Type Acc    | 0.9939 |
| Season Acc  | 0.9870 |
| Gender Acc  | 0.9968 |

---

#### Avg Validation Metrics

| Metric       | Value  |
|-------------|--------|
| Loss        | 4.0140 |
| Color Acc   | 0.7380 |
| Type Acc    | 0.7925 |
| Season Acc  | 0.7328 |
| Gender Acc  | 0.8630 |

### Model Architecture
- Backbone was Resnet50 followed by 4 parallel dense layer for 4 outputs
- Thats why u see for accuracies above (for four outputs)

