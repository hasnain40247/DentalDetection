from tkinter import *
from tkinter.filedialog import askopenfile
import os

from dep import PATH_TO_SAVED_MODEL, PATH_TO_LABELS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


class App(Tk):
    def __init__(self):
        self.window = Tk()
        self.window.title("Dental Record Detection")
        photo = PhotoImage(file="tooth.png")
        self.window.iconphoto(False, photo)
        self.window.config(width=500, height=300, bg="#71DFE7", padx=15, pady=15)
        self.label = Label(text="Dental Detection!", font=("Helvetica", 65, "bold"), fg="#FFE652")
        self.label.config(bg="#71DFE7")
        self.label.grid(column=0, row=0, pady=8)
        self.button = Button(activebackground="#FFE652", command=self.open_file, text="Upload An Image", padx=8,
                             font=("Helvetica", 22), fg="white", highlightthickness=0, bg="#009DAE").grid(column=0,
                                                                                                          row=1, pady=8)
        self.file_path = ""
        self.label2 = Label(text="", font=("Helvetica", 12, "bold"), fg="white")
        self.label2.config(bg="#71DFE7")
        self.label2.grid(column=0, row=3, pady=9)
        self.window.mainloop()

    def detectImage(self):
        im = Image.open(self.file_path)
        print(self.file_path)
        print(im)
        IMAGE_PATHS = self.file_path
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        MIN_CONF_THRESH = float(0.60)
        print('Loading model...', end='')
        start_time = time.time()
        detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                            use_display_name=True)

        def load_image_into_numpy_array(path):
            return np.array(Image.open(path))

        print('Running inference for {}... '.format(IMAGE_PATHS), end='')
        image = cv2.imread(IMAGE_PATHS)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        image_with_detections = image.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=0.5,
            agnostic_mode=False)

        # DISPLAYS OUTPUT IMAGE
        cv2.imshow("img", image_with_detections)

        cv2.waitKey(0)

        self.label2 = Label(text="Dental Record Saved!", font=("Helvetica",
                                                               10,
                                                               "bold"), fg="white")
        self.label2.config(bg="#71DFE7")
        self.label2.grid(column=0, row=5, pady=9)

    def open_file(self):
        self.file_path = askopenfile(mode='r', filetypes=[('Image Files', '*')]).name

        if self.file_path is not None:
            self.label2.config(text=f"{self.file_path.split('/')[-1]} uploaded")
            self.button2 = Button(activebackground="#FFE652", command=self.detectImage,
                                  text="Run Detection", padx=8, font=("Helvetica",
                                                                      22), fg="white", highlightthickness=0,
                                  bg="#009DAE").grid(column=0, row=4, pady=10)
        else:
            pass
