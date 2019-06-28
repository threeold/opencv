##简单的——人脸相似度对比
##图片对比
import os,dlib,glob,numpy
from skimage import io # 人脸关键点检测器
import cv2
import configparser

class FaceComp:
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

    def dectface(self,img_path):

        descriptors = numpy.load(self.descriptors_path)
        candidate = numpy.load(self.Candidate_Path)
        try:
            ##    test_path=input('请输入要检测的图片的路径（记得加后缀哦）:')
            img = io.imread(img_path)
            dets = self.detector(img, 1)
        except:
            return '输入路径有误，请检查！'

        dist = []
        for k, d in enumerate(dets):
            shape = self.sp(img, d)
            # print(shape)
            face_descriptor = self.facerec.compute_face_descriptor(img, shape)
            d_test = numpy.array(face_descriptor)
            # print(d_test)
            for i in descriptors:  # 计算距离
                dist_ = numpy.linalg.norm(i - d_test)
                dist.append(dist_)  # 训练集人物和距离组成一个字典
        if dist:
            c_d = dict(zip(candidate, dist))
            cd_sorted = sorted(c_d.items(), key=lambda d: d[1])
            # print(cd_sorted)
            # print("识别到的人物最有可能是: ", cd_sorted[0][0])

            if cd_sorted[0][1] > 0.38:
                #cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 1, 8, 0)
                #cv2.putText(img, 'unkown rate {}'.format(cd_sorted[0][1]), (d.left(), d.top() - 10),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                print('the perion is unkown rate{}'.format(cd_sorted[0][1]))
                print('the perion like {}'.format(cd_sorted[0][0]))  ##比较像的哪个人
                #cv2.imshow("Image", img)
                #cv2.waitKey(0)
                return '比对失败'
            else:
                #cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 1, 8, 0)
                #cv2.putText(img, cd_sorted[0][0], (d.left(), d.top() - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                print('the perion is {}'.format(cd_sorted[0][0]))
                print('the perion is rate{}'.format(cd_sorted[0][1]))
                #cv2.imshow("Image", img)
                #cv2.waitKey(0)
                return '比对成功'
        else:
            return '定位不到脸部'

if __name__ == '__main__':
    print('开始比对！')
    while True:
        test_path = input('请输入要检测的图片的路径（记得加后缀哦）:')
        res=FaceComp().dectface(test_path)
        print(res)
        ##FaceComp().dectface(r'images\1234.jpg')
        ### images\1234.jpg
        ### images\4355.jpg

