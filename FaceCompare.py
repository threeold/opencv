##简单的——人脸相似度对比
import os,dlib,glob,numpy
from skimage import io # 人脸关键点检测器

predictor_path = "shape_predictor_68_face_landmarks.dat" # 人脸识别模型、提取特征值
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat" # 训练图像文件夹
faces_folder_path =r'images\face' # 加载模型
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)



candidate = [] # 存放训练集人物名字
descriptors = [] #存放训练集人物特征列表
for f in glob.glob(os.path.join(faces_folder_path,"*.jpg")):
    print("正在处理: {}".format(f))
    img = io.imread(f)
    candidate.append(f.split('\\')[-1].split('.')[0]) # 人脸检测
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = sp(img, d) # 提取特征
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        v = numpy.array(face_descriptor)
        descriptors.append(v)
print('识别训练完毕！')


try:
    ##    test_path=input('请输入要检测的图片的路径（记得加后缀哦）:')
    img = io.imread(r'images\1234.jpg')
    dets = detector(img, 1)
except:
    print('输入路径有误，请检查！')

dist = []
for k, d in enumerate(dets):
    print("aaaaaaaaaaa")
    shape = sp(img, d)
    #print(shape)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    d_test = numpy.array(face_descriptor)
    #print(d_test)
    for i in descriptors:  # 计算距离
        dist_ = numpy.linalg.norm(i - d_test)
        dist.append(dist_)  # 训练集人物和距离组成一个字典
c_d = dict(zip(candidate, dist))
cd_sorted = sorted(c_d.items(), key=lambda d: d[1])
print(cd_sorted)
print("识别到的人物最有可能是: ", cd_sorted[0][0])
