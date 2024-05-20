from flask import Flask, request, render_template
import os
from PIL import Image
import numpy as np
import pandas as pd
from fastai.vision.all import *
from fastai.vision.all import *
# # import tensorflow as tf
# from predict import *
app = Flask(__name__)

# Load your pre-trained model
# model = tf.keras.models.load_model('your_model.h5')
image_folder_path = Path("C:\\Users\\yelar\\Desktop\\project_finalyear\\images")
csv_file_path = "C:\\Users\\yelar\\Desktop\\project_finalyear\\train.csv"

# # Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Print the column names to verify
# print(df.columns)
# Create DataBlock and DataLoaders
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=ColReader('image_id', pref=image_folder_path),
    get_y=ColReader('label'),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(192, method='squish'),
    batch_tfms=aug_transforms(size=128, min_scale=0.75)
)

dls = dblock.dataloaders(df)
arch = 'convnext_small.fb_in22k'
model = vision_learner(dls, arch, metrics=[error_rate,accuracy], n_out=10)

# # Load the saved parameters (state dictionary) into the model
model.load_state_dict(torch.load('C:\\Users\\yelar\\Desktop\\FULL_STACK\\flask\\model.pth'))
# from PIL import Image
# from fastai.vision.all import *

# Assuming you have already defined dls and learner as in your previous code
# Load the test image
app.config['UPLOAD_FOLDER'] = 'C:/Users/yelar/Desktop/FULL_STACK/flask/static'
@app.route('/')
def index():
    return render_template('index.html')
# app.config['UPLOAD_FOLDER'] = 'static'
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        # Ensure the upload folder exists
        upload_folder = Path(app.config['UPLOAD_FOLDER'])
        upload_folder.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded image
        image_path = upload_folder / 'test1.jpg'
        file.save(image_path)
        
        # Preprocess image
        print("hello ")
        
        # Load and process the image
        test_image = Image.open(image_path)
            # Here you can add code to preprocess or analyze the image
            # For example:
            # test_image = test_image.resize((128, 128))

            # Make predictions (assuming a function `make_predictions` exists)
            # result = make_predictions(test_image)
            # return jsonify(result)
        # print(test_image)
        # Make predictions
        class_names = ["bacterial_leaf_blight", "bacterial_leaf_streak", "bacterial_panicle_blight", 
                   "blast", "brown_spot","dead_heart", "downy_mildew", "hispa", "normal", 
                   "tungro"]
        test_image_path = "C:\\Users\\yelar\Desktop\\project_finalyear\\images\\100101.jpg"
        test_image = PILImage.create(test_image_path)
        preds, _ = model.get_preds(dl=dls.test_dl([test_image]))
        predicted_index = torch.argmax(preds).item()
        predicted_class_name = class_names[predicted_index]
        print("Predicted Class:", predicted_class_name)
        print("-------------------------------------------")
        predicts=[]
        print("Predicted Class:", "Probability:")
        for x,y in zip(class_names,preds[0]):
            y=str(y)
            y=y.replace('tensor',"")
            y=y.replace('(','')
            y=y.replace(')','')
            y= '{:.3f}'.format(float(y))
            print(x,"------------>",y)
            predicts.append(x)
            predicts.append(y)

        # print(preds)
        # Define class names
        # print(preds)
        return render_template('predict.html',predicts=predicts)
if __name__ == '__main__':
    app.run(debug=True)
