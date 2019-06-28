##简单的——人脸相似度对比
##训练训练
import os,dlib,glob,numpy
from skimage import io # 人脸关键点检测器
import cv2
import configparser

class FaceDrill:
    def __init__(self, config_name="FACE_CONFIG", filepath=r'base_config/face_config.ini'):
        targer = configparser.ConfigParser()
        targer.read(filepath)
        self.predictor_path = targer.get(config_name, 'Predictor_Path')
        self.face_rec_model_path = targer.get(config_name, 'Face_Rec_Model_Path')
        self.faces_folder_path = targer.get(config_name, 'Faces_Folder_Path')
        self.descriptors_path = targer.get(config_name, 'Descriptors_Path')
        self.Candidate_Path = targer.get(config_name, 'Candidate_Path')


        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(self.predictor_path)
        self.facerec = dlib.face_recognition_model_v1(self.face_rec_model_path)

    def dectface(self):
        candidate = []  # 存放训练集人物名字
        descriptors = []  # 存放训练集人物特征列表
        print(self.faces_folder_path)
        for f in glob.glob(os.path.join(self.faces_folder_path , "*.jpg")):
            print("正在处理: {}".format(f))
            img = io.imread(f)
            candidate.append(f.split('\\')[-1].split('.')[0])  # 人脸检测
            dets = self.detector(img, 1)
            for k, d in enumerate(dets):
                shape = self.sp(img, d)  # 提取特征
                face_descriptor = self.facerec.compute_face_descriptor(img, shape)
                v = numpy.array(face_descriptor)
                descriptors.append(v)

        numpy.save(self.descriptors_path, descriptors)  # 存放训练集人物特征列表
        numpy.save(self.Candidate_Path, candidate)  # 存放训练集人物名字

        print('识别训练完毕！')

if __name__ == '__main__':
    print('开始训练！')
    FaceDrill().dectface()
