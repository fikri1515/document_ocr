import tensorflow as tf
from tensorflow.keras.models import load_model
import re
import time
from tqdm import tqdm

class char_batch_predict:
    def __init__(self, model_path, labels, data, batch=32):

        print(f"[info] loading model from model_path: {model_path}")
        self._model_path = model_path
        self._model = load_model(model_path)
        self.batch_size = batch

        indexed_non_char_val, char_image_val_index, char_image_val = self.char_image(data)

        _batch_dataset = self._create_batch(char_image_val, self.batch_size)
        self._inference_result = self._char_list_inference(self._model, labels, _batch_dataset)

        final_result = list(zip(char_image_val_index, self._inference_result))
        final_result = indexed_non_char_val + final_result
        final_result.sort(key=lambda x: x[0])
        final_result = [x[1] for x in final_result]

        ascii_codes, text = self.convert_to_ascii_codes(final_result)

        self._ascii_codes = ascii_codes
        self._text_result = text

    def char_image(self, data):
        non_char_val = []
        char_val = []

        for index, val in enumerate(data):
            if val[0] == 'char':
                char_val.append([index, val[1]])
            else:
                non_char_val.append([index, val[1]])

        char_image = [char_list[1] for char_list in char_val]
        char_image_index = [char_list[0] for char_list in char_val]

        return non_char_val, char_image_index, char_image

    def _create_batch(self, char_image, batch_size):
        print(f"[info] creating batch dataset")

        batch_image = []

        for image in char_image:
            inputs = tf.cast(image, dtype=tf.float32) / 255.0

            inputs = tf.expand_dims(inputs, axis=0)
            batch_image.append(inputs)

        dataset = tf.data.Dataset.from_tensor_slices(batch_image)
        dataset = dataset.batch(batch_size)

        print('[info] batch dataset created')
        print(f"[info] len of batches is {len(dataset)}, with size per batch: {batch_size}")

        return dataset

    def _char_list_inference(self, model, labels, dataset):
        inference_result = []

        total_batches = len(dataset)

        start_time = time.time()
        for batch_num, batch in enumerate(tqdm(dataset, desc='processing batches', total=total_batches)):

            batch = tf.squeeze(batch, axis=1)

            # batch prediction
            predictions = model.predict(batch, verbose=0)
            predicted_classes = tf.argmax(predictions, axis=-1)

            for pred_class in predicted_classes.numpy():
                inference_result.append(labels[pred_class])

            print(f'Batch {batch_num + 1}/{total_batches} processed')

        result_time = time.time() - start_time
        print(f"[info] result time for batch prediction: {result_time:.2f} seconds")

        return inference_result
    
    def convert_to_ascii_codes(self, char_list):
        ascii_codes = []
        for item in char_list:
            if item == ' ':
                ascii_codes.append(ord(' '))
            elif item == '\n':
                ascii_codes.append(ord('\n'))
            else:
                match = re.match(r'char\d+_(.)', item)
                if match:
                    char = match.group(1)
                    ascii_codes.append(ord(char))

        text = ''.join(chr(code) for code in ascii_codes)
        return ascii_codes, text