from flask import Flask, render_template,request
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt


sys.path.append('../model')
sys.path.append('../data')

from inference import *

path = os.getcwd()
app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = os.path.join(path,'static')
prep_image = prepImage((512,512))
processed_file_location = os.path.join(app.config["IMAGE_UPLOADS"], 'processed_img.jpeg')
segmented_file_location = os.path.join(app.config["IMAGE_UPLOADS"], 'segmented_img.jpeg')
uploaded_file_location = os.path.join(app.config["IMAGE_UPLOADS"], 'uploaded_img.jpeg')
infer = inferModel()

@app.route("/")
def home():
  
  if os.path.exists(processed_file_location):
    os.remove(processed_file_location)
  return render_template("index.html")


@app.route("/", methods=['POST'])
def upload_data():
  

  if request.form.get('btn_identifier') == "segment-btn-id":
    
    if os.path.exists(processed_file_location):
      image = Image.open(processed_file_location).convert('RGB')
      segmented_image,class_list= infer.infer(image)
      plt.imsave(segmented_file_location, segmented_image)
      
      return render_template("index.html",uploaded_image='processed_img.jpeg',segmented_image='segmented_img.jpeg',class_list=class_list)

    else:  
      return render_template("index.html")

  elif request.form.get('btn_identifier')== "upload-btn-id":

    if os.path.exists(processed_file_location):
      os.remove(processed_file_location)

    image = request.files['upload-image-id']
    if image.filename == '':
      flash('No selected file')
      return redirect(request.url)

    image.save(os.path.join(app.config["IMAGE_UPLOADS"], 'uploaded_img.jpeg'))
    

    image = Image.open(uploaded_file_location).convert('RGB')
    image = prep_image.prep(image)
    image.save(processed_file_location)

    return render_template("index.html", uploaded_image='processed_img.jpeg')


if __name__ == "__main__":
  try:
    port = int(sys.argv[1])
  except Exception as e:
    port = 80
  app.run(host='0.0.0.0', port=port,debug=False)