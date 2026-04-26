import pickle
import numpy as np
from PIL import Image
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args["image"]

def extract_features(filename, feature_extractor):
    try:
        image = Image.open(filename)
    except (FileNotFoundError, OSError) as error:
        raise ValueError(
            "Couldn't open image. Make sure the image path and extension are correct."
        ) from error
    image = image.convert("RGB")
    image = image.resize((299, 299))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = feature_extractor.predict(image, verbose=0)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    try:
        from keras.preprocessing.sequence import pad_sequences
    except ImportError:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

    in_text = '<start>'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == '<end>':
            break
    return in_text.replace('<start>', '').replace('<end>', '').strip()

def define_model(vocab_size, max_length):
        try:
            from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
            from keras.models import Model
        except ImportError:
            from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
            from tensorflow.keras.models import Model

        #CNN model from 2048 nodes to 256 nodes
        inputs1 = Input(shape=(2048,), name = 'input_1')
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)

        #LSTM sequence model
        inputs2 = Input(shape=(max_length,), name = 'input_2')
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)

        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print(model.summary())
        return model


class KerasCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("keras."):
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                return super().find_class(f"tensorflow.{module}", name)
        return super().find_class(module, name)


def load_tokenizer(path):
    with open(path, 'rb') as tokenizer_file:
        try:
            return pickle.load(tokenizer_file)
        except ModuleNotFoundError:
            tokenizer_file.seek(0)
            return KerasCompatUnpickler(tokenizer_file).load()
    
max_length = 34
try:
    tokenizer = load_tokenizer('tokenizer.p')
except ModuleNotFoundError as error:
    raise SystemExit(
        "Unable to load tokenizer. Install keras or tensorflow in this environment."
    ) from error
vocab_size = len(tokenizer.word_index) + 1

try:
    model = define_model(vocab_size, max_length)    
    model.load_weights('models/model_9.h5')
    try:
        from keras.applications.xception import Xception
    except ImportError:
        from tensorflow.keras.applications.xception import Xception
    xception_model = Xception(include_top=False, pooling='avg')
except ImportError as error:
    raise SystemExit(
        "Unable to run caption model. Install keras or tensorflow in this environment."
    ) from error

photo = extract_features(img_path, xception_model)

description = generate_desc(model, tokenizer, photo, max_length)
print(description)
