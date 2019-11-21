import numpy as np
import os, argparse
import tensorflow as tf
import matplotlib as mlp
import matplotlib.pyplot as plt
from PIL import Image
from lapjv import lapjv
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from tensorflow.python.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Flatten, Dense
from keras.regularizers import l2


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', type=int, help="number of small images in a row/column in output image")
parser.add_argument('-d', '--dir', type=str, help="source directory for images")
parser.add_argument('-r', '--res', type=int, default=200, help="width/height of output square image")
parser.add_argument('-n', '--name', type=str, default='tsne_grid.jpg', help='name of output image file')
parser.add_argument('-p', '--path', type=str, default='./', help="destination directory for output image")
parser.add_argument('-x', '--per', type=int, default=50, help="tsne perplexity")
parser.add_argument('-i', '--iter', type=int, default=5000, help="number of iterations in tsne algorithm")
parser.add_argument('-w', '--weights', type=str, default=5000, help="path to a file containing vgg16 weights where last layer is embedding")

args = parser.parse_args()
out_res = args.res
out_name = args.name
out_dim = args.size
to_plot = np.square(out_dim)
perplexity = args.per
tsne_iter = args.iter

if out_dim == 1:
    raise ValueError("Output grid dimension 1x1 not supported.")

if os.path.exists(args.dir):
    in_dir = args.dir
else:
    raise argparse.ArgumentTypeError("'{}' not a valid directory.".format(in_dir))
    
if os.path.exists(args.weights):
    weights_path = args.weights
else:
    raise argparse.ArgumentTypeError("'{}' weight path does not exist.".format(in_dir))

if os.path.exists(args.path):
    out_dir = args.path
else:
    raise argparse.ArgumentTypeError("'{}' not a valid directory.".format(out_dir))

def build_model(weights_path):
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape = (200,200,3))

    x = Flatten()(vgg16.output)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    y = Dense(46, activation='softmax', name='img')(x)

    x_bbox = Flatten()(vgg16.output)
    x_bbox = Dense(4096, activation='relu')(x_bbox)
    x_bbox = Dense(4096, activation='relu')(x_bbox)
    bbox = Dense(4, kernel_initializer='normal', name='bbox')(x_bbox)

    final_model = Model(inputs=vgg16.input,
                    outputs=[y, bbox])

    
    final_model.load_weights(weights_path)
    
    embed_model = Sequential()
    for layer in final_model.layers[:-7]: # just exclude last layer from copying
        embed_model.add(layer)
    
    embed_model.add(final_model.layers[-6])
    embed_model.add(final_model.layers[-4])
    
    return embed_model

def load_img(in_dir):
    pred_img = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
    img_collection = []
    for idx, img in enumerate(pred_img):
        img = os.path.join(in_dir, img)
        img_collection.append(image.load_img(img, target_size=(out_res, out_res)))
    if (np.square(out_dim) > len(img_collection)):
        raise ValueError("Cannot fit {} images in {}x{} grid".format(len(img_collection), out_dim, out_dim))
    return img_collection

def get_activations(model, img_collection):
    activations = []
    for idx, img in enumerate(img_collection):
        if idx == to_plot:
            break;
        print("Processing image {}".format(idx+1))
        img = img.resize((200, 200), Image.ANTIALIAS)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        activations.append(np.squeeze(model.predict(x)))
    return activations

def generate_tsne(activations):
    tsne = TSNE(perplexity=perplexity, n_components=2, init='random', n_iter=tsne_iter)
    X_2d = tsne.fit_transform(np.array(activations)[0:to_plot,:])
    X_2d -= X_2d.min(axis=0)
    X_2d /= X_2d.max(axis=0)
    return X_2d

def save_tsne_grid(img_collection, X_2d, out_res, out_dim):
    grid = np.dstack(np.meshgrid(np.linspace(0, 1, out_dim), np.linspace(0, 1, out_dim))).reshape(-1, 2)
    cost_matrix = cdist(grid, X_2d, "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    row_asses, col_asses, _ = lapjv(cost_matrix)
    grid_jv = grid[col_asses]
    out = np.ones((out_dim*out_res, out_dim*out_res, 3))

    for pos, img in zip(grid_jv, img_collection[0:to_plot]):
        h_range = int(np.floor(pos[0]* (out_dim - 1) * out_res))
        w_range = int(np.floor(pos[1]* (out_dim - 1) * out_res))
        out[h_range:h_range + out_res, w_range:w_range + out_res]  = image.img_to_array(img)

    im = image.array_to_img(out)
    im.save(out_dir + out_name, quality=100)

def main():
    model = build_model(weights_path)
    img_collection = load_img(in_dir)
    activations = get_activations(model, img_collection)
    print("Generating 2D representation.")
    X_2d = generate_tsne(activations)
    print("Generating image grid.")
    save_tsne_grid(img_collection, X_2d, out_res, out_dim)

if __name__ == '__main__':
    main()