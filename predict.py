import os
import pandas as pd
from fastai.vision.all import *

def predict_with_model():
    image_folder_path = Path("C:\\Users\\yelar\\Desktop\\project_finalyear\\images")
    csv_file_path = "C:\\Users\\yelar\\Desktop\\project_finalyear\\train.csv"
    df = pd.read_csv(csv_file_path)

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

    # Load the model
    arch = 'convnext_small.fb_in22k'  # Updated model name
    model = vision_learner(dls, arch, metrics=[error_rate,accuracy], n_out=10)
    model_file_path = "C:\\Users\\yelar\\Desktop\\project_finalyear\\model.pth"
    model.load_state_dict(torch.load(model_file_path))

    # Perform prediction
    test_image_path = "C:\\Users\\yelar\\Desktop\\project_finalyear\\test5.jpeg"
    test_image = PILImage.create(test_image_path)
    # print(test_image)
    # Make predictions
    preds, _ = model.get_preds(dl=dls.test_dl([test_image]))

    class_names = ["bacterial_leaf_blight", "bacterial_leaf_streak", "bacterial_panicle_blight", 
                   "blast", "brown_spot","dead_heart", "downy_mildew", "hispa", "normal", 
                   "tungro"]
    predicted_index = torch.argmax(preds).item()
    predicted_class_name = class_names[predicted_index]
    print("Predicted Class:", predicted_class_name)
    print("-------------------------------------------")
    predicts={}
    print("Predicted Class:", "Probability:")
    for x,y in zip(class_names,preds[0]):
        y=str(y)
        y=y.replace('tensor',"")
        y=y.replace('(','')
        y=y.replace(')','')
        y= '{:.6f}'.format(float(y))
        print(x,"------------>",y)
        predicts[x]=y

if __name__ == '__main__':
    predict_with_model()
