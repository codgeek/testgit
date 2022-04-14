# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python demo usage about MNN API """
from __future__ import print_function
import numpy as np
import MNN
import cv2
import PIL.Image

def inference(config):
    """ inference mobilenet_v1 using a specific picture """

    print("checking ", config['backend'], " inference:")
    interpreter = MNN.Interpreter("mobilenet_v1_open.mnn")

    session = interpreter.createSession(config)


    input_tensor = interpreter.getSessionInput(session)

    image = cv2.imread('panda.jpg')
    #cv2 read as bgr format
    image = image[..., ::-1]
    #change to rgb format
    image = cv2.resize(image, (224, 224))
    #resize to mobile_net tensor size
    image = image - (103.94, 116.78, 123.68)
    image = image * (0.017, 0.017, 0.017)
    #preprocess it
    image = image.transpose((2, 0, 1))
    #change numpy data type as np.float32 to match tensor's format
    image = image.astype(np.float32)


    tmp_input = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float,\
                    image, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    #constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    tmp_output = MNN.Tensor((1, 1001), MNN.Halide_Type_Float, np.ones([1, 1001]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
    output_tensor.copyToHostTensor(tmp_output)
    print("Embed MNN in DSW framework, expect mobilenet v1 classify type: 389, giant panda.\ncorresponding to index in: https://s3.amazonaws.com/onnx-model-zoo/synset.txt")
    probability = np.array(tmp_output.getData())
    print("In DSW, real classify: ", np.argmax(probability))

    topk_index = np.argsort(probability)
    print("In DSW, top 10 classify:", topk_index[-1:-10:-1])
    print("In DSW, top 10 predict data:", probability[topk_index[-1:-10:-1]])

if __name__ == "__main__":

    config = {}
    config['backend'] = "CPU"
    inference(config)

    config = {}
    config['backend'] = "CUDA"
    inference(config)
