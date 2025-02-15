import cv2 as cv
import numpy as np
import argparse
import json

from core.text_detection import DB_text_detection
from core.text_detection import EAST_text_detection
from core.segmentation import char_segmentation
from config.config_helpers import Config

from utils.image_utils.Image import showImage
from utils.image_utils.Image import Image_preprocessing
from utils.image_utils import warp_image_crop
from utils.image_utils.Image import image_padding
from core.segmentation import word_segmentation

def main():

    # argparsing for CLI command
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help='image path input')
    parser.add_argument('-c', '--config', help='show configuration', action='store_true')
    parser.add_argument('-w', '--write', help='saving file', action='store_true')
    parser.add_argument('-m', '--model', help='where models that want to be used', type=str, choices=['ResNet50', 'ResNet101'], default='ResNet50')
    parser.add_argument('-d', '--display', help='displaying results', action='store_true')

    args = parser.parse_args()

    # loading json config
    json_config = Config(f"config\\config.json")

    # image processing
    if args.image:
        from core.char import char_batch_predict

        image_path = args.image
        DB_model_path = f"models\\{json_config.get('image.DB_detection_model')}"
        EAST_model_path = f"models\\{json_config.get('image.EAST_detection_model')}"

        if args.model == 'ResNet50':
            char_recognition_model_path = f"models\\ResNet\\ResNet50\\{json_config.get('models.ResNet')[0]}"
        elif args.model == 'ResNet101':
            char_recognition_model_path = f"models\\ResNet\\ResNet101\\{json_config.get('models.ResNet')[1]}"
        else:
            print('[info] model has not been found')

        class_names = json_config.get('models.label')

        try:
            image = cv.imread(image_path)
            print('[info] image successfully loaded')
            print('[info] preprocessing image')
        except:
            print("[info] image could not be loaded")

        frame = Image_preprocessing(image, json_config.get('image.scaling_factor'))
        DB_frame = DB_text_detection(frame._thresh, DB_model_path)
        EAST_frame = EAST_text_detection(frame._thresh, EAST_model_path)

        output_size = json_config.get('image.output_image_size')
        
        boxes, confidences = warp_image_crop.sort_boxes(DB_frame._boxes, DB_frame._confidence)
        croppedLineImage = warp_image_crop.crop_image(frame._default_cvImage, boxes)

        # data loader function definition
        char_list = []
        for lines in croppedLineImage:
            wordImage = word_segmentation(lines)

            # error handling for empty char detection
            if wordImage._cropped_image:
                for index, word in enumerate(wordImage._cropped_image):
                    charImage = char_segmentation(word)

                    if charImage._mask:
                        for mask in charImage._mask:
                            mask = np.clip(mask, 0, 255).astype(np.uint8)
                            padded_image = image_padding(mask, image_size=(output_size, output_size), interpolation=0)

                            # charlist
                            char_list.append(['char', padded_image])

                    # charlist
                    if index == len(wordImage._cropped_image) - 1:
                        pass
                    else:
                        char_list.append(['space', ' '])

            char_list.append(['break', '\n'])

        char = char_batch_predict(char_recognition_model_path, class_names, char_list)

        if args.display:
            print('[info] displaying results\n')
            print(char._text_result)

        if args.write:

            with open(f'{image_path}_result.txt', 'w') as file:
                file.write(char._text_result)

            print(f'[info] OCR result was saved as {image_path}_result.txt')

    # showing configuration
    if args.config:
        print('[info] Show configuration')
        print(json.dumps(json_config._config, indent=4))

if __name__ == "__main__":
    main()