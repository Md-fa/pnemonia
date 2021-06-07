#Importing required packages
from preprocessing import full_prep
from config_submit import config as config_submit
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from layers import acc
from data_detector import DataBowl3Detector,collate
from data_classifier import DataBowl3Classifier
from utils import *
from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas
import matplotlib
# %matplotlib inline
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import sys
from preprocessing.step1 import *
from preprocessing.full_prep import lumTrans
from layers import nms,iou
from flask import Flask, render_template, request, send_file
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import os
import shutil
import logging
import base64
from PIL import Image
from base64 import encodebytes
import io
from pathlib import Path
from google.cloud import storage
from google.cloud.exceptions import NotFound
import json


sys.path.append('../training/')
sys.path.append('../')
sys.path.append('../preprocessing/')

#Logging all the steps
logging.basicConfig(level=logging.DEBUG)

logging.info('Program Execution Started')
datapath = os.path.join(str(os.getcwd()),'UPLOAD_FOLDER')
prep_result_path = config_submit['preprocess_result_path']
skip_prep = config_submit['skip_preprocessing']
skip_detect = config_submit['skip_detect']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


logging.info('Detection model loaded')
nodmodel = import_module(config_submit['detector_model'].split('.py')[0])
config1, nod_net, loss, get_pbb = nodmodel.get_model()
checkpoint = torch.load(config_submit['detector_param'])
nod_net.load_state_dict(checkpoint['state_dict'])

# torch.cuda.set_device(0)  # commented for CPU
# nod_net = nod_net.cuda()  # commented for CPU
# cudnn.benchmark = True    # commented for CPU
nod_net = DataParallel(nod_net)

logging.info('Casenet model loaded')
casemodel = import_module(config_submit['classifier_model'].split('.py')[0])
casenet = casemodel.CaseNet(topk=5)
config2 = casemodel.config
checkpoint = torch.load(config_submit['classifier_param'], encoding= 'unicode_escape')
casenet.load_state_dict(checkpoint['state_dict'])

# torch.cuda.set_device(0)  # commented for CPU
# casenet = casenet.cuda()  # commented for CPU
# cudnn.benchmark = True    # commented for CPU
casenet = DataParallel(casenet)
filename = config_submit['outputfile']


def test_casenet(model,testset):
    data_loader = DataLoader(
        testset,
        batch_size = 1,
        shuffle = False,
        num_workers = 1,
        # pin_memory=True) # commented for CPU
        pin_memory=False)

    #model = model.cuda()
    model.eval()
    predlist = []

    #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i,(x,coord) in enumerate(data_loader):
        #print(i,(x,coord))

        #commenting for cpu
        # coord = Variable(coord).cuda()
        # x = Variable(x).cuda()
        nodulePred,casePred,_ = model(x,coord)
        predlist.append(casePred.data.cpu().numpy())

    predlist = np.concatenate(predlist)
    return predlist

def preprocess_detect(datapath):
    if not skip_prep:
        logging.info("Preprocessing DICOMS")
        testsplit = full_prep(datapath,config_submit['preprocess_result_path'],
                            n_worker = config_submit['n_worker_preprocessing'],
                            use_existing=config_submit['use_exsiting_preprocessing'])
        logging.info("Done Preprocessing")
    else:
        testsplit = os.listdir(datapath)


    bbox_result_path = './bbox_result'
    if not os.path.exists(bbox_result_path):
        os.mkdir(bbox_result_path)
    #testsplit = [f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]

    if not skip_detect:
        margin = 32
        sidelen = 144
        config1['datadir'] = prep_result_path
        split_comber = SplitComb(sidelen,config1['max_stride'],config1['stride'],margin,pad_value= config1['pad_value'])

        dataset = DataBowl3Detector(testsplit,config1,phase='test',split_comber=split_comber)
        test_loader = DataLoader(dataset,batch_size = 1,
            shuffle = False,num_workers = 32,pin_memory=False,collate_fn =collate)

        # pbb,lbb = test_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=config_submit['n_gpu']) #commented for CPU
        pbb,lbb = test_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=1)


    config2['bboxpath'] = bbox_result_path
    config2['datadir'] = prep_result_path

    dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
    predlist = test_casenet(casenet,dataset).T
    df = pandas.DataFrame({'id':testsplit, 'cancer':predlist})
    df.to_csv(filename,index=False)
    return pbb
#---------------------------------------------------------------------------------------------
# downloading data from gcs bucket
def download(bucket_input,prefixs):
    storage_client = storage.Client.from_service_account_json('./terraform_key.json')
    bucket = storage_client.get_bucket(bucket_input)
    blobs = bucket.list_blobs(prefix=prefixs)  # Get list of files
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        file_split = blob.name.split("/")
        directory = "/".join(file_split[0:-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(blob.name) 


#----------------------------------------------------------------------------------------------   

# Initializing flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(str(os.getcwd()),'UPLOAD_FOLDER')
cors = CORS(app)

# Process images
@app.route("/process", methods=["POST"])
def processReq():
    global graph, sess
    try:
        try:
            os.remove('saved_figure.png')
            logging.info("Removed previous result image found")
        except:
            logging.info("No previous result image found")

        if os.path.exists(app.config['UPLOAD_FOLDER']):
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            logging.info("deleting the folder")

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.mkdir(app.config['UPLOAD_FOLDER'])
            sub_folder = os.path.join(app.config['UPLOAD_FOLDER'],'sample')
            os.mkdir(sub_folder)
            logging.info("creating the folder")

        files = request.files.getlist("file")
        for file in files:
            file.save(os.path.join(sub_folder, secure_filename(str(file.filename))))
        logging.info('file uploaded successfully')

    except:
        logging.error("Fatal error Uploading files and creating a folder")

    try:
        logging.info('Initializing detect')
        pbb = preprocess_detect(datapath)
        #Visualizing the results with plots and save it as image
        img = np.load('slice_image.npy')

        # pbb will be the bounding box of detected image
        pbb = pbb[pbb[:,0]>-1]

        pbb = nms(pbb,0.05)
        box = pbb[0].astype('int')[1:]

        ax = plt.subplot(1,1,1)
        plt.imshow(img[0,box[2]],'gray')
        plt.axis('off')
        rect = patches.Rectangle((box[2]-box[3],box[1]-box[3]),box[3]*2,box[3]*2,linewidth=2,edgecolor='red',facecolor='none')
        ax.add_patch(rect)
        plt.savefig('saved_figure.png')

    except:
        logging.error("Error Detecting")
    with open("saved_figure.png", "rb") as image_file:
        encoded_img = base64.b64encode(image_file.read())

    return encoded_img
#---------------------------------------------------------------------------------------------
#this function is used to get the input for download function to download the dataset from GCS bucket
@app.route("/download", methods=["POST"])
def input():
    global graph, sess
    try:
        try:
            os.remove('saved_figure.png')
            logging.info("Removed previous result image found")
        except:
            logging.info("No previous result image found")

        if os.path.exists(app.config['UPLOAD_FOLDER']):
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            logging.info("deleting the folder")

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.mkdir(app.config['UPLOAD_FOLDER'])
            sub_folder = os.path.join(app.config['UPLOAD_FOLDER'],'sample')
            os.mkdir(sub_folder)
            logging.info("creating the folder")

        bucket_input1 = json.load(request.files['datas'])
        #bucket_input =  file.read()
        #bucket_input1 = request.get_json('datas')

        #config_json = json.loads(data)
        bucket_input = bucket_input1['input_data_bucket']
        prefixs = bucket_input1['prefixs']
        download(bucket_input,prefixs)
        #moving the downloaded dataset to upload folder
        s1 = "./"+prefixs+"/"
        d1 = "./UPLOAD_FOLDER/sample/"
        file_names = os.listdir(s1)
        for file_name in file_names:
            shutil.move(os.path.join(s1, file_name), d1)
        file_path = s1
        os.rmdir(file_path)
 
    except:
        logging.error("Fatal error Uploading files and creating a folder")

    try:
        logging.info('Initializing detect')
        pbb = preprocess_detect(datapath)
        #Visualizing the results with plots and save it as image
        img = np.load('slice_image.npy')

        # pbb will be the bounding box of detected image
        pbb = pbb[pbb[:,0]>-1]

        pbb = nms(pbb,0.05)
        box = pbb[0].astype('int')[1:]

        ax = plt.subplot(1,1,1)
        plt.imshow(img[0,box[2]],'gray')
        plt.axis('off')
        rect = patches.Rectangle((box[2]-box[3],box[1]-box[3]),box[3]*2,box[3]*2,linewidth=2,edgecolor='red',facecolor='none')
        ax.add_patch(rect)
        plt.savefig('saved_figure.png')

    except:
        logging.error("Error Detecting")
    with open("saved_figure.png", "rb") as image_file:
        encoded_img = base64.b64encode(image_file.read())

    return encoded_img

#----------------------------------------------------------------------------------------------
@app.route("/test", methods=["GET"])
def testReq():
    logging.info('Testing function')
    return "Good to go!!!"

if __name__ == "__main__":
    # Create full paths
    app.run(debug=True, host='0.0.0.0', port=6005)
