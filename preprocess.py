import imutils, dlib, cv2 
from imutils.face_utils import rect_to_bb, shape_to_np
import numpy as np
import time, pickle, json
from math import floor, sqrt, ceil
from tqdm import tqdm
from os.path import basename, exists, abspath, join, dirname
from os import system, listdir, makedirs
from time import time
from matplotlib import pyplot as plt
import copy
import argparse
import multiprocessing
from itertools import product
from functools import partial

'''
Note that the Dlib face extraction will run super slow on the process
and really needs the GPU processing power.
To check on how to work with it, take a peek at the following link:
https://stackoverflow.com/questions/51697468/how-to-check-if-dlib-is-using-gpu-or-not
'''

parser = argparse.ArgumentParser()
parser.add_argument('--srcDir', help='Source of video data')
parser.add_argument('--destDir', help='Path to video')
args = parser.parse_args()

# https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
PREDICTOR_PATH = abspath('./shape_predictor_68_face_landmarks.dat') 

# https://github.com/justadudewhohacks/face-recognition.js-models/blob/master/models/mmod_human_face_detector.dat
DETECTOR_PATH = abspath('./mmod_human_face_detector.dat')

# work out your paths here
BACKEND_PATH = abspath('../')
PREPROC_DIR = abspath('./preprocessed')

DEBUG_SHOW_FRAMES = False
DEBUG_SHOW_PATCHES = False

class Video:
    def __init__(self, video_path, frame_size=1000):
        self.video_path = video_path
        self.frame_size = frame_size
        self.reader = cv2.VideoCapture(video_path)
        self.length = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
        self.last_frame = self.next_frame(True)
    
    def next_frame(self, return_frame = False):
        success, frame = self.reader.read()

        if not success:
            return success

        h,w,_ = np.shape(frame)
        if( h>w):
            frame = imutils.resize(frame, height=self.frame_size)
        else:
            frame = imutils.resize(frame, width=self.frame_size)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if return_frame:
            return (frame, gray)
        else:
            self.last_frame = (frame, gray)
            return success

class Preprocessor:
    def __init__(self,
            face_dims=(256, 256),
            frame_size=960
        ):
        self.frame_size = frame_size
        self.face_dims = face_dims
        self.faces = []
        self.lp = []
        self.landmarks = []
        self.bboxes = []
        self.frames = []

    @staticmethod
    def main_dimension(shape, mult=3):
        left_eye = shape[36]  # left corner of left eye
        right_eye = shape[45]  # right corner of right eye

        return round(sqrt((left_eye[1]-right_eye[1])**2 + (left_eye[0]-right_eye[0])**2) * mult) 

    def ewma_vectorized(self, data, alpha, offset=None, dtype=None, order='C', out=None):
        """
        Calculates the exponential moving average over a vector.
        Will fail for large inputs.
        :param data: Input data
        :param alpha: scalar float in range (0,1)
            The alpha parameter for the moving average.
        :param offset: optional
            The offset for the moving average, scalar. Defaults to data[0].
        :param dtype: optional
            Data type used for calculations. Defaults to float64 unless
            data.dtype is float32, then it will use float32.
        :param order: {'C', 'F', 'A'}, optional
            Order to use when flattening the data. Defaults to 'C'.
        :param out: ndarray, or None, optional
            A location into which the result is stored. If provided, it must have
            the same shape as the input. If not provided or `None`,
            a freshly-allocated array is returned.
        """
        data = np.array(data, copy=False)
        
        if dtype is None:
            if data.dtype == np.float32:
                dtype = np.float32
            else:
                dtype = np.float64
        else:
            dtype = np.dtype(dtype)
    
        if data.ndim > 1:
            # flatten input
            data = data.reshape(-1, order)
    
        if out is None:
            out = np.empty_like(data, dtype=dtype)
        else:
            assert out.shape == data.shape
            assert out.dtype == dtype
    
        if data.size < 1:
            # empty input, return empty array
            return out
    
        if offset is None:
            offset = data[0]
    
        alpha = np.array(alpha, copy=False).astype(dtype, copy=False)
    
        # scaling_factors -> 0 as len(data) gets large
        # this leads to divide-by-zeros below
        scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                                   dtype=dtype)
        # create cumulative sum array
        np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                    dtype=dtype, out=out)
        np.cumsum(out, dtype=dtype, out=out)
    
        # cumsums / scaling
        out /= scaling_factors[-2::-1]
    
        if offset != 0:
            offset = np.array(offset, copy=False).astype(dtype, copy=False)
            # add offsets
            out += offset * scaling_factors[1:]
    
        return out

    def get_bb(self, rect, shape, mult):
        # convert dlibs rectangle to opencv style box
        # (x,y,w,h)
        (x,y,w,h) = rect_to_bb(rect)
        hds = self.main_dimension(shape, mult) / 2
        c = [x + w/2, y + h/2]

        # x1, y1, x2, y2
        return int(c[0] - hds), int(c[1] - hds), int(c[0] + hds), int(c[1] + hds)
    
        #draw polylines
    def drawPolyline(self, image, landmarks, start, end, isClosed=False):
            points = []
            for i in range(start, end+1):
                    point = [landmarks[i][0], landmarks[i][1]]
                    points.append(point)

            points = np.array(points, dtype=np.int32)
            cv2.polylines(image, [points], isClosed, (255, 255, 255), 2, 16)


    def drawPolylines(self, image, landmarks):
            self.drawPolyline(image, landmarks, 0, 16)           # Jaw line
            self.drawPolyline(image, landmarks, 17, 21)          # Left eyebrow
            self.drawPolyline(image, landmarks, 22, 26)          # Right eyebrow
            self.drawPolyline(image, landmarks, 27, 30)          # Nose bridge
            self.drawPolyline(image, landmarks, 30, 35, True)    # Lower nose
            self.drawPolyline(image, landmarks, 36, 41, True)    # Left eye
            self.drawPolyline(image, landmarks, 42, 47, True)    # Right Eye
            self.drawPolyline(image, landmarks, 48, 59, True)    # Outer lip
            self.drawPolyline(image, landmarks, 60, 67, True)    # Inner lip

    def add_frame(self, length):
        self.bboxes = np.apply_along_axis(self.ewma_vectorized, 0, self.bboxes, 0.2).astype(int)
        for i in range(len(self.frames)):
            #Crop face
            frame = self.frames[i]
            
            x1, y1, x2, y2 = self.bboxes[i]

            x1 = int(max(x1 - length/2, 0))
            y1 = int(max(y1 - length/2, 0))
            x2 = int(min(x2 + length/2, frame.shape[1]))
            y2 = int(min(y2 + length/2, frame.shape[0]))


            patch = cv2.resize(frame[y1:y2, x1:x2], self.face_dims)
            scale = np.asarray([(self.face_dims[0]/(x2-x1)), (self.face_dims[1]/(y2-y1))])

            self.faces.append(patch)
            self.landmarks[i] -= [x1, y1]

            x = self.landmarks[i] * scale
            self.landmarks[i] = x.astype(int)

            temp = np.zeros(patch.shape)
            self.drawPolylines(temp, self.landmarks[i])
            self.lp.append(temp)

            if DEBUG_SHOW_PATCHES:
                cv2.rectangle(temp, (x1,y1), (x2, y2), (0, 225, 0), 2)
                cv2.imshow("Extracted", temp)
                key = cv2.waitKey(1) & 0xFF
            
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    exit()       

    @staticmethod
    def output_video(data, output_path, fps=30):
        print("Saving to", output_path)

        if not exists(dirname(output_path)):
            makedirs(dirname(output_path))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, data[0].shape[:2])
 
        for i in data:
            out.write(i)
        out.release()


    @staticmethod
    def output_imgs(data, output_path):
        print("Saving to", output_path)

        if not exists(output_path):
            makedirs(output_path)

        for i, item in enumerate(tqdm(data, "saving images")):
            cv2.imwrite(join(output_path, str(i) + ".jpg"), item)

    @staticmethod
    def output_landmarks(data, output_file):
        if not exists(dirname(output_file)):
            makedirs(dirname(output_file))
        f = open(output_file, 'wb')
        print("Creating a pickle of size: {}".format(np.shape(data)))
        pickle.dump(data, f)
        f.close()

    def reset(self):
        self.faces = []
        self.landmarks = []

    def run(self,
            video_path,
            save=True,
            multiplier=1,
            destination_path=PREPROC_DIR
        ):
        self.reset()
        bestBbox = None
        success = True
        video_name = basename(video_path).split(".")[0]
        video = Video(video_path, frame_size=self.frame_size)

        if not exists(destination_path):
            makedirs(destination_path)

        detector = dlib.cnn_face_detection_model_v1(DETECTOR_PATH)
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        total_frames = (0, video.length-1)
        bestDist = 0
        bestHds = 0

        t = time()
        for tf in tqdm(range(*total_frames), "FaceDetection"):
            if not success:
                success = video.next_frame()
                continue
            img, gray = video.last_frame
            rects = detector(gray, 1)

            if len(rects) == 0:
                success = video.next_frame()
                continue

            rects = rects[0]

            # determine the facial landmarks for the region
            # convert the landmarks to (x,y)-coordinates to numpy array
            self.frames.append(copy.deepcopy(img))
            rect = rects.rect
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            self.landmarks.append(shape)

            center = shape[30]  #Nose tip as center
            
            # Max distance from the center
            for(x,y) in shape:
                dist = sqrt((x-center[0])**2 + (y-center[1])**2)
                if dist > bestDist:
                    bestDist = dist

            hds = self.main_dimension(shape, multiplier)/2

            x1 = center[0] - hds
            y1 = center[1] - hds
            x2 = center[0] + hds
            y2 = center[1] + hds
            
            bbox = (x1, y1, x2, y2)
            
            self.bboxes.append(bbox)
            success = video.next_frame()

        self.add_frame(bestDist)

        # if DEBUG_SHOW_FRAMES:
            # for img in self.faces:
                # cv2.rectangle(img, (x1,y1), (x2, y2), (0, 225, 0), 2)

        if save:
            self.output_imgs(self.faces, join(destination_path, 'Faces', video_name))
            self.output_landmarks(self.landmarks, join(destination_path, 'Landmarks', video_name + '_l'))
            self.output_imgs(self.lp, join(destination_path, 'Landmark_Mask', video_name+'_lm'))

        cv2.destroyAllWindows()
        return "Processed {} frames in {} seconds".format(tf, (time()-t))


if __name__ == '__main__':

    vidList = list(listdir(join(args.srcDir)))
    vidList = vidList[int(len(vidList)/2):]
    for vid in vidList:
        print(join(args.destDir, 'Faces', vid.split('.')[0]))
        if not exists(join(args.destDir, 'Faces', vid.split('.')[0])):
            Preprocessor().run(
                    join(args.srcDir, vid),
                    save=True,
                    multiplier=1,
                    destination_path=join(args.destDir)
                )