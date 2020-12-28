from IPython.display import clear_output, Image, display
import PIL.Image as image
import io
import json
import torch
import numpy as np
from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
import utils
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
import wget
import pickle
import os
from flask import Flask, request, jsonify, render_template
class lxmert:
    def __init__(self,filename):
        self.filename = filename

    def prediction(self,x,filename):
        OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
        ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
        GQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
        VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
    
        
        # load object, attribute, and answer labels

        objids = utils.get_data(OBJ_URL)
        attrids = utils.get_data(ATTR_URL)
        gqa_answers = utils.get_data(GQA_URL)
        vqa_answers = utils.get_data(VQA_URL)

        # load models and model components
        frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")

        frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)

        image_preprocess = Preprocess(frcnn_cfg)

        lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        lxmert_gqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-gqa-uncased")
        lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")
        #print(self.filename)
        img_path =  'C:\\lxmert\\test\\'+self.filename
        
        frcnn_visualizer = SingleImageViz(img_path, id2obj=objids, id2attr=attrids)
        
        # run frcnn
        images, sizes, scales_yx = image_preprocess(img_path)
        output_dict = frcnn(
        images, 
        sizes, 
        scales_yx=scales_yx, 
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt"
        )
        # add boxes and labels to the image

        frcnn_visualizer.draw_boxes(
        output_dict.get("boxes"),
        output_dict.pop("obj_ids"),
        output_dict.pop("obj_probs"),
        output_dict.pop("attr_ids"),
        output_dict.pop("attr_probs"),
        )
        
        questions = [x]

        #Very important that the boxes are normalized
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")

        for question in questions:
            # run lxmert
            question = [question]

            inputs = lxmert_tokenizer(
            question,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
            )

            # run lxmert(s)
            output_gqa = lxmert_gqa(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=inputs.token_type_ids,
            return_dict=True,
            output_attentions=False,
            )
            output_vqa = lxmert_vqa(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=inputs.token_type_ids,
            return_dict=True,
            output_attentions=False,
            )
            # get prediction
            pred_vqa = output_vqa["question_answering_score"].argmax(-1)
            pred_gqa = output_gqa["question_answering_score"].argmax(-1)
        
            return "Question:"+ str(question)+"\nprediction from LXMERT GQA:"+ str(gqa_answers[pred_gqa])+"\nprediction from LXMERT VQA:"+ str(vqa_answers[pred_vqa])
        



