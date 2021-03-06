import keras
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from flask import jsonify
from keras.models import load_model
import numpy as np
# import keras

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
def create_model():

        model = keras.applications.ResNet50(input_shape=(128,128, 3), classes=12, weights=None)
        return model

def load_trained_model(weights_path):
        model = create_model()
        model.load_weights(weights_path)
        return model

def prepare_image(image):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize((128,128))
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image)

			# classify the input image and then initialize the list
			# of predictions to return to the client
			model=load_trained_model('C:/Users/vedan/Desktop/unschool_minor/un.h5')
			preds = model.predict(image)
			preds=np.array(preds)
                        #print(preds)
	# 		results = imagenet_utils.decode_predictions(preds)
	# 		data["predictions"] = []
        #
	# 		# loop over the results and add them to the list of
	# 		# returned predictions
	# 		for (imagenetID, label, prob) in results[0]:
	# 			r = {"label": label, "probability": float(prob)}
	# 			data["predictions"].append(r)
    #
	# 		# indicate that the request was a success
	# 		data["success"] = True
    #
	# # return the data dictionary as a JSON response
	# return flask.jsonify(data)
	return jsonify({'prediction': str(np.argmax(preds))})
        

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	# model=load_trained_model('C:/Users/vedan/Desktop/unschool_minor/un.h5')
	app.run()
