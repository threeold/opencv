# encoding: utf-8
# #老杨的猫,环境:PYCHARM，python3.6,opencv3
import cv2, os
import cv2.face as fc  # 此处有坑,找不到脸,这样引用程序可以运行，欢迎大牛指点，CV2和CV3的结构区别没有搞清楚，应该怎么样引用才是正确的
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # pip install pillow


# 由于cv2.putText（）不支持汉字，把图像里加入需要显示的文字，可以为汉字
def cvtopil(img, posion, txt):  # 图像数组，文字位置，文字内容
    pil_im = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.truetype("simhei.ttf", 50, encoding="utf-8")  # 第一个参数为字体文件路径，第二个为字体大小
    draw.text(posion, txt, (0, 0, 255), font=font)  # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体
    image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)  # 把图像数组由RGB转为CV2处理的BGR格式
    return image  # 返回处理后图像文件
    # 脸部图像采集模块。采集脸部图像，转为200*200的大小后，存到每个人对应的文件夹下


def dectface():
    # OPENCV3 自带的脸部检测XML文件
    # D:\Downloads...../haarcascade_frontalface_default.xml 文件所在路径为我的电脑里文件路径，检测脸部，可酌情修改
    face_cas = cv2.CascadeClassifier('D:\pythonweb\face_test/haarcascade_frontalface_default.xml')
    # 检测眼睛，这里用不到
    eye_cas = cv2.CascadeClassifier('D:\pythonweb\face_test/haarcascade_eye.xml')
    cap = cv2.VideoCapture(0)  # 打开默认摄像头
    count = 0  # 初始化计数器，用来生成文件名，如0.pgm,1.pgm,3.pgm.....


    #C:\Users\Administrator\.PyCharm2017.2\system\python_stubs\-1184660488\cv2\CascadeClassifier.py
    #连续读取摄像头图像，检测到脸部图像，把脸部图像以PGM的灰度格式格式保存在当前程序路径里的 jm1,jm2...文件夹下
    #采集要求：1光线适中，2脸部正直，3脸部图像大小以刚刚露出头发和下颌最佳，控制距离，太远采集图像精确度差，太近
    #没脸，4有效采集时间5秒左右就够用了，太多影响运行速度。

    while True:
        ret, frame = cap.read()  # 读取图像，RET为判断是否采集到图像，FRAME为采集到的一帧图像
        # 此处最好判断是否采集到图像  if ret：do else:err  偷懒了
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图像，用来判断脸部用
        # 找脸，参数:图像文件，压缩率（越小检测迭代次数越多，越慢，越详细，此处图像大，1.3-1.5均可），矩形个数最小值，flags不知道，最小检测窗口大小，最大检测窗口大小。返回一个脸部区域，左上角为（0，0）坐标系，起始x,y点，w宽，h高
        faces = face_cas.detectMultiScale(gray, 1.3, 5)
        # 用方框勾画出脸部位置

    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画矩形，位置，大小，颜色，通道
        # cv2.putText(img,'abc',(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2) 不能显示汉字
        frame = cvtopil(img, (x, y - 20), "老杨")  # 测试汉字用，可采集出图像是灰的，待解决
        f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))  # 从灰度图中，扣出脸部，设置固定大小像素
        # 保存到jm1里
        cv2.imwrite('F:\pytest\cvtest\detectface\jm1\ %s.pgm' % str(count), f)
        count += 1
        # 眼睛检测
        # eyes=eye_cas.detectMultiScale(f,1.03,5,0,(40,40))
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
        cv2.imshow("demo", frame)  # 显示图像
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # camera.release()
    # cv2.destroyAllWindows()


# 读取样本文件，并加载到一个列表里，返回值包括2部分，[文件列表，对应标签],标签用来对照姓名用。
def read_img(path, sz=None):

    pr_img = []  # 图像列表
    pr_flg = []  # 对应标签
    pr_count = 0  # 初始化检测到的人数
    path = r"D:\pythonweb\face_test\images"

    for dirname, dirnames, filenames in os.walk(path):  # 遍历当前程序目录我的图像文件保存在f:盘下
        #print(os.walk(path))
        #print(dirname,dirnames,filenames)
        for subdirname in dirnames:  # 遍历程序文件夹下的各个目录
            subject_path = os.path.join(dirname, subdirname)
            print(subject_path)
            for filename in os.listdir(subject_path):  # 遍历文件夹下文件
                print(filename)
                try:
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # 读取JM文件下PGM文件
                    print(filepath)
                    #print(im.shape)
                    if im.shape != (200, 200):  # 判断像素是否200
                        im = cv2.resize(im, (200, 200))
                        pr_img.append(np.asarray(im, dtype=np.uint8))  # 添加图像
                        pr_flg.append(pr_count)  # 添加标签
                except:
                    print("io error")
                pr_count += 1  # 另一个人的标签
    return [pr_img, pr_flg]


# 学习样本，比对样本和实例，根据对应算法返回标签和系数，依据标签对照姓名，系数值提现了准确度。
def face_rec():
    names = ['BAO BAO ', 'BA BA BA ', 'Bei Bei', 'MMMMMMM']  # 标签对应的名字0,baobao,1，bbb,2，beibei....
    [x, y] = read_img("D:")  # 调读取函数，返回图像、和标签列表
    y = np.asarray(y, dtype=np.int32)  # 转为NUMPY的ARRAY

    # CV自带的三种算法，现用LBPH算法。此处有坑，坑的我差点放弃，原来叫createLBPHFaceRecognizer，为什么我下载的这模样
    # model=fc.EigenFaceRecognizer_create()
    # model = fc.FisherFaceRecognizer_create()
    model = fc.LBPHFaceRecognizer_create()

    # 训练，此处应把训练结果保存，再用到时直接读取结果，效率更高，xml?json?pickle?
    model.train(np.asarray(x), np.asarray(y))

    # 下面读取摄像头图像，用矩形标识检测到脸部和训练后结果比对，打印出对应标签所对应名字
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('D:\pythonweb\face_test/haarcascade_frontalface_default.xml')
    while True:
        read, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            try:
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)

                params = model.predict(roi)  # predict（）函数做比对，返回一个元祖格式值 （标签，系数）。系数和算法有关，
                # 前2种算法值低于5000不可靠，LBPH低于50可靠，80-90不可靠，高于90纯蒙
                # 此处有文章可做，通过单位时间内检测到的系数平均值，可以得到更准确结果
                print(params)
                # 打印标签对应名字，如cvtopil的灰度问题解决，可cvtopil函数替换
                cv2.putText(img, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except:
                continue
        cv2.imshow("abc", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    '''
    1.先用dectface()采集样本，注释掉face_rec()，将采集样本保存到jm文件夹里，现在每次采集不同人样本，需要手动建立JM1，JM2.....以后应完善程序流程为，首先判断是否有此人样本，如没有自动建立文件夹并保存该人的样本。
    2.采集完成后，注释dectface()，用face_rec()通过比对得到是谁的结果。
    3.names=['BAO BAO ','BBBBBBB','Bei Bei','MMMMMMM'] 名字和对应标签应该存成文件，对应读取
    3.真心不会用类啊。。。咋用类实现整个流程呢？？
    4.从学PYTHON到现在也不过一个月时间，基础很不牢，cv2,CV3从引用到使用的区别搞不清楚呢，恳求大牛各种批评指导啊！！
    '''
    dectface()
    #face_rec()
