# trained on IP102 dataset
model_list = [
    {'model_name': 'resnet50',
     'checkpoint': "./output/train/resnet50/resnet50.pth.tar",
     'input_size': 224},
    {'model_name': 'vit_small_patch16_224',
     'checkpoint': "./output/train/vit_s16/vit_s16.pth.tar",
     'input_size': 224},
    {'model_name': 'volo_d1',
     'checkpoint': "./output/train/volo_d1/volo_d1.pth.tar",
     'input_size': 224},
    {'model_name': 'vip_s7',
     'checkpoint': "./output/train/vip_s7/vip_s7.pth.tar",
     'input_size': 224},
]