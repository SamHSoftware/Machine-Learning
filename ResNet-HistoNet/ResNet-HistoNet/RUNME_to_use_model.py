from code_module import use_resnet, select_folder, select_file, custom_accuracy

# Function to use the trained ResNet on a folder of images. 
# Function input arg 1: image_ext [list] --> Strings of the image extensions to be considered for processing. 
use_resnet(image_ext = ['.tif', '.png'])