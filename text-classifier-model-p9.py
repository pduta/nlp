import pandas as pd 
from simpletransformers.classification import ClassificationModel, 
ClassificationArgs 
# 1. Prepare your data 
train_data = [ 
["I love this movie!", 1], 
["That was a terrible experience.", 0], 
["Best day ever!", 1], 
    ["I hate waiting in line.", 0] 
] 
train_df = pd.DataFrame(train_data, columns=["text", "labels"]) 
 
# 2. Configure the model 
model_args = ClassificationArgs() 
model_args.num_train_epochs = 3 
model_args.overwrite_output_dir = True 
model_args.reprocess_input_data = True 
 
# 3. Initialize the Model (using DistilBERT for speed) 
# 'outputs/' is where the trained model will be saved 
model = ClassificationModel( 
    "distilbert",  
    "distilbert-base-uncased",  
    args=model_args, 
    use_cuda=False # Set to True if you have an NVIDIA GPU + CUDA installed 
) 
 
# 4. Train the model 
print("--- Training Started ---") 
model.train_model(train_df) 
 
# 5. Make a prediction 
predictions, raw_outputs = model.predict(["This is the best thing I've seen 
all year!"]) 
print(f"Prediction: {predictions[0]}") # Output will be 1
