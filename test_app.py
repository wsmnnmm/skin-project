import io
import base64
from gevent import monkey
monkey.patch_all()
import numpy as np
from gevent.pywsgi import WSGIServer
from resnet_attn import ResNet50_Self_Attn3, ResNet50_AgentAttn
from flask import Flask, jsonify, request
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser(description='PyTorch Sketch Me That Shoe Example')
parser.add_argument('--net', type=str, default='resnet50',
                    help='The model to be used (resnet50,efficientnet_b5,efficientnet_v2)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')  # 64
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--epoch_count', type=int, default=1,
                    help='the starting epoch count, we save the model by <epoch_count>,<save_latest_freq>+<epoch_count>...')
parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')

parser.add_argument('--weight_decay', type=float, default=0.0005, help='Adm weight decay')

parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--classes', type=int, default=419,
                    help='number of classes')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='NetModel', type=str,
                    help='name of experiment')
parser.add_argument('--normalize_feature', default=False, type=bool,
                    help='normalize_feature')
args = parser.parse_args()


app = Flask(__name__)

image_class_index ={
    "0":'Melanocytic nevi',
    "1": 'dermatofibroma',
    "2": 'Benign keratosis-like lesions ',
    "3": 'Basal cell carcinoma',
    "4": 'Actinic keratoses',
    "5": 'Vascular lesions',
    "6": 'Dermatofibroma'
}

model = ResNet50_Self_Attn3(pretrained=True, out_features=7)

if hasattr(model, 'fc'):
# 添加Dropout层，假设Dropout比例为0.5
    dropout_ratio = 0.5
# 访问并替换模型的全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
    nn.Dropout(p=dropout_ratio),
    nn.Linear(in_features=num_ftrs, out_features=7))
else:
    raise AttributeError("The model does not have a 'fc' attribute, check your model definition.")
# 因为我们只需要模型进行预测，所以将模型设置为评估模式
model.eval()
print(model)
# 加载最佳模型进行测试
checkpoint_path = 'model_best.pth.tar'.format(args.name)
checkpoint = torch.load(checkpoint_path)  # 加载最佳模型的检查点
model.load_state_dict(checkpoint['state_dict'])  # 将模型参数恢复到最佳状态
model.to('cuda')
print('已加载模型')

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整为模型输入所需的尺寸
    transforms.ToTensor(),  # 将PIL图像或numpy.ndarray转换为torch.Tensor，并归一化到[0., 1.]
    # 如果你的模型在训练时使用了归一化（比如减去均值，除以标准差），那么你也需要在这里做相同的操作
    # 例如，对于ImageNet的预训练模型，你可能需要添加以下步骤：
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_and_preprocess_image(img_path):
    # 加载图片
    # image = Image.open(img_path).convert('RGB')  # 确保图片是RGB格式
    # 应用预处理
    processed_image = preprocess(img_path)
    # 注意：这里返回的processed_image没有批次维度。
    # 在将其传递给模型之前，你需要这样做：
    processed_image1 = torch.unsqueeze(processed_image,0)
    return processed_image1


def get_prediction(img_path):
    tensor = load_and_preprocess_image(img_path)
    # 如果模型在GPU上，确保 tensor 也被转移到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)
    # 对图像进行预测
    with torch.no_grad():  # 在评估模式下关闭梯度计算
        output = model(tensor)  # 添加批次维度
    # 应用 softmax 获取概率分布
    probabilities = torch.softmax(output, dim=1)
    # 获取最高概率的索引
    _, predicted_idx = torch.max(probabilities, 1)
    # 转换为 Python 整数
    predicted_idx = predicted_idx.item()
    # 获取对应索引的类别名称
    predicted_class = image_class_index[str(predicted_idx)]  # 确保使用正确的键来访问字典

    # 获取最高概率的值
    highest_probability = probabilities[0][predicted_idx].item()

    return predicted_idx, predicted_class, highest_probability

'''
处理一个HTTP POST请求，该请求中包含了一个以Base64编码的图片。
函数主要目的:接收这张图片，对其进行解码，然后使用某种预测模型（这里假设是get_prediction函数）来预测图片中的对象，并返回预测结果。
'''
@app.route('/predict',methods=['POST'])
#def predict(): 定义了一个名为predict的函数，
# 它不接受任何参数（尽管在函数体内使用了request和jsonify等看似来自Flask框架的对象，这些对象通常通过Flask的路由装饰器传递给视图函数
def predict():
#检查当前HTTP请求的方法是否为POST。在Web开发中，POST请求通常用于提交数据，如表单数据或文件上传。
    if request.method =='POST':
    #从POST请求的表单数据中获取键为'picture'的值，这个值预期是一个Base64编码的图片字符串。
        img_base64=request.form.get('picture')
    #解码Base64图片：将Base64编码的字符串解码为原始的二进制数据。
        print('已处理图片')
        image = base64.b64decode(img_base64)
        #print(img_base64, image, 'image')
#将二进制数据转换为图片对象：先使用io.BytesIO将二进制数据包装成一个类文件对象，后使用Image.open函数打开这个类文件对象，将其转换为一个Image对象。
        image=Image.open(io.BytesIO(image))
    #进行预测：调用get_prediction函数，将图片对象作为参数传入，该函数返回三个值：class_id（类别ID）、class_name（类别名称）和prob（预测概率）。
        class_id,class_name,prob=get_prediction(image)
        #print(class_id,class_name,prob)
    #返回JSON格式的预测结果：使用jsonify函数（Flask框架提供的函数）将预测结果封装成一个JSON对象，并返回给客户端。
    # 这样，客户端就可以接收到一个易于解析和使用的预测结果了。
        return jsonify({'class_id':class_id,'class_name':class_name,'prob':prob})

#启动一个WSGI（Web Server Gateway Interface）服务器。
#WSGI是Python语言定义的Web服务器和Web应用程序或框架之间的一种简单而通用的接口。这允许Python的Web应用程序与各种Web服务器一起工作，而不需要修改应用程序代码。

if __name__=='__main__':
#创建了一个WSGIServer的实例。这个类（实际上它是一个特定框架或库中的类，比如werkzeug.serving.WSGIServer，是Flask开发服务器使用的一个类）用于启动一个WSGI服务器。
# '0.0.0.0' 是一个特殊的IP地址，表示服务器将监听所有可用的网络接口。5000 是服务器监听的端口号。
# app 是一个WSGI应用程序对象，它是符合WSGI标准的Python可调用对象（通常是一个函数或类实例），用于处理传入的HTTP请求。
    server=WSGIServer(('0.0.0.0',5000),app)
#调用serve_forever()方法会使服务器持续运行，接受并处理传入的HTTP请求，直到被明确关闭（例如，通过发送一个信号或按下Ctrl+C）。
    server.serve_forever()
