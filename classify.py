#!/usr/bin/env python3

# Command
# python3 classify.py --model-name test --captcha-dir ~/Downloads/validation_data/ --output ~/Downloads/stuff.txt --symbols symbols.txt


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy as np
import argparse
import tensorflow as tf

def decode(characters, y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    with tf.device('/gpu:0'):
        with open("output.txt", 'w') as output_file:
            json_file = open(args.model_name + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = tf.keras.models.model_from_json(loaded_model_json)
            model.load_weights(args.model_name + '.h5')
            model.compile(loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(1e-3, amsgrad=False),
                        metrics=['accuracy'])

            for x in os.listdir(args.captcha_dir):
                # load image and preprocess it
                image_path = os.path.join(args.captcha_dir, x)
                raw_data = cv2.imread(image_path)
                rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                image = np.array(rgb_data) / 255.0
                image = cv2.resize(image, (128, 64))
                image = np.expand_dims(image, axis=0)  # Shape becomes (1, 64, 128, 3)
                prediction = model.predict(image)
                output_file.write(x + ", " + decode(captcha_symbols, prediction) + "\n")

                print('Classified ' + x)


if __name__ == '__main__':
    main()
