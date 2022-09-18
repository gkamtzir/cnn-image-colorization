from training import train_model
from testing import test_model, render_layer_output

# Make sure you have already downloaded the data mentioned in the README file.
train_model("./data/train_color", None, 50, "cnn_new")

# Puts to the test the final model.
test_model("./models/cnn_6th_2000_full.pt", "cnn_testing", "./data/test_color", None)

# Renders the outputs of the layers.
render_layer_output("./models/cnn_6th_2000_full.pt", "", "./data/train_color", 16)