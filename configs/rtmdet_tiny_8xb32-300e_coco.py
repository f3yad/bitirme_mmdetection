# data_root = 'data/coco/'
data_root = 'datasets/spinexr/' # dataset root

# train_ann_file = 'annotations/instances_train2017.json'
# train_ann_file = 'train.json'
train_ann_file = 'train_full_10.json'
train_ann_file = 'train_aug_sample.json'

# val_ann_file = 'annotations/instances_val2017.json'
# val_ann_file = 'test.json'
val_ann_file = 'val.json'

test_ann_file = 'test.json'

# train_data_prefix = 'train2017/'
# train_data_prefix = 'train_images_jpg/'
train_data_prefix = 'train_images_aug/'
train_data_prefix = 'train_images_sample/'

# val_data_prefix = 'val2017/'
# val_data_prefix = 'test_images_jpg/'
val_data_prefix = 'val_images/'

test_data_prefix = 'test_images/'

# train_batch_size_per_gpu = 32
train_batch_size_per_gpu = 10

# train_num_workers = 10
train_num_workers = 1

# max_epochs = 300
max_epochs = 100

# stage2_num_epochs = 20
stage2_num_epochs = 1

# base_lr = 0.004
base_lr = 0.0001

# num_classes = 80
num_classes = 7

val_eval_ann_file = data_root + val_ann_file
test_eval_ann_file = data_root + test_ann_file

## DELETE
# train_ann_file = val_ann_file
# train_data_prefix = val_data_prefix

metainfo = {
    'classes': ("Osteophytes", "Spondylolysthesis", "Disc space narrowing", "Other lesions", "Surgical implant", "Foraminal stenosis", "Vertebral collapse",),
    'palette': [
        (220, 20, 60),
    ]
}

### BITIRME

auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'

custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),

    dict(
        switch_epoch=280,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    640,
                    640,
                ),
                type='RandomResize'),
            dict(crop_size=(
                640,
                640,
            ), type='RandomCrop'),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(type='PackDetInputs'),
        ],
        type='PipelineSwitchHook'),
]

dataset_type = 'CocoDataset'
default_hooks = dict(
    # checkpoint=dict(interval=10, max_keep_ckpts=3, type='CheckpointHook'),
    checkpoint=dict(interval=5, max_keep_ckpts=3, type='CheckpointHook', save_best="auto"), ### BITIRME
    # logger=dict(interval=50, type='LoggerHook'),
    logger=dict(interval=5, type='LoggerHook'),   ### BITIRME
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
interval = 10

# load_from = None
load_from = './checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'  ### BITIRME

log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)

model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.167,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        type='CSPNeXt',
        widen_factor=0.375),
    bbox_head=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        anchor_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        exp_on_reg=False,
        feat_channels=96,
        in_channels=96,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True,
            # class_weight=[0.36245528149124456, 0.9108114502010883, 0.8852609795355254, 0.6685188400764022, 0.8113804004214964, 0.7841140529531568, 1.0],
        ),
        norm_cfg=dict(type='SyncBN'),
        num_classes=num_classes,
        pred_kernel_size=1,
        share_conv=True,
        stacked_convs=2,
        type='RTMDetSepBNHead',
        with_objectness=False),
    data_preprocessor=dict(
        batch_augments=None,
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            57.375,
            57.12,
            58.395,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        expand_ratio=0.5,
        in_channels=[
            96,
            192,
            384,
        ],
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=1,
        out_channels=96,
        type='CSPNeXtPAFPN'),
    test_cfg=dict(
        max_per_img=300,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.65, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='RTMDet')

# optimizer
optim_wrapper = dict(
    # _delete_ = True,   ### BITIRME
    # optimizer=dict(lr=0.004, type='AdamW', weight_decay=0.05),
    optimizer=dict(lr=base_lr, type='AdamW', weight_decay=0.05),  ### BITIRME
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')

# learning rate
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        # end=1000,
        end=10,   ### BITIRME
        start_factor=1e-05,
        type='LinearLR'),
    dict(
        # use cosine lr from 10 to 20 epoch
      
        # T_max=150, # Must match the difference between begin and end.
        T_max=max_epochs - max_epochs // 2, ### BITIRME
        # begin=150,
        begin=max_epochs // 2,  ### BITIRME
        by_epoch=True,
        convert_to_iter_based=True,
        # end=300,
        end=max_epochs,  ### BITIRME
        # eta_min=0.0002,
        eta_min=base_lr * 0.05,   ### BITIRME
        type='CosineAnnealingLR'),
]

resume = False

test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        640,
        640,
    ), type='Pad'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]

train_cfg = dict(
    dynamic_intervals=[
        (
            280,
            1,
        ),
    ],
    # max_epochs=300,
    max_epochs=max_epochs,   ### BITIRME
    type='EpochBasedTrainLoop',
    # val_interval=10
    val_interval=1   #### BITIRME
)


train_dataloader = dict(
    batch_sampler=None,
    batch_size=train_batch_size_per_gpu,
    dataset=dict(
        metainfo=metainfo,  ### BITIRME
        ann_file=train_ann_file,
        backend_args=None,
        data_prefix=dict(img=train_data_prefix),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                max_cached_images=20,
                pad_val=114.0,
                random_pop=False,
                type='CachedMosaic'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1280,
                    1280,
                ),
                type='RandomResize'),
            dict(crop_size=(
                640,
                640,
            ), type='RandomCrop'),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                max_cached_images=10,
                pad_val=(
                    114,
                    114,
                    114,
                ),
                prob=0.5,
                random_pop=False,
                ratio_range=(
                    1.0,
                    1.0,
                ),
                type='CachedMixUp'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=train_num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

# train_pipeline = [
#     dict(backend_args=None, type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         img_scale=(
#             640,
#             640,
#         ),
#         max_cached_images=20,
#         pad_val=114.0,
#         random_pop=False,
#         type='CachedMosaic'),
#     dict(
#         keep_ratio=True,
#         ratio_range=(
#             0.5,
#             2.0,
#         ),
#         scale=(
#             1280,
#             1280,
#         ),
#         type='RandomResize'),
#     dict(crop_size=(
#         640,
#         640,
#     ), type='RandomCrop'),
#     dict(type='YOLOXHSVRandomAug'),
#     dict(prob=0.5, type='RandomFlip'),
#     dict(pad_val=dict(img=(
#         114,
#         114,
#         114,
#     )), size=(
#         640,
#         640,
#     ), type='Pad'),
#     dict(
#         img_scale=(
#             640,
#             640,
#         ),
#         max_cached_images=10,
#         pad_val=(
#             114,
#             114,
#             114,
#         ),
#         prob=0.5,
#         random_pop=False,
#         ratio_range=(
#             1.0,
#             1.0,
#         ),
#         type='CachedMixUp'),
#     dict(type='PackDetInputs'),
# ]

train_pipeline = [
    # dict(type='Mosaic', img_scale=(1333, 800), max_ratio=4 / 3, prob=0.5),  # Mosaic augmentation
    # dict(type='RandomFlip', flip_ratio=0.5),  # Random flip
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),  # Resize image
    # dict(type='RandomCrop', crop_size=(512, 512), prob=0.5),  # Random crop to improve class distribution
    # dict(type='PhotometricDistortion', prob=0.5),  # Color jitter for more diversity in minor class appearances
    # dict(type='RandomBrightness', prob=0.3),  # Random brightness adjustment
    # dict(type='RandomContrast', prob=0.3),  # Random contrast adjustment
    # dict(type='RandomSaturation', prob=0.3),  # Random saturation adjustment
    # dict(type='CutOut', n_holes=8, max_h_size=32, max_w_size=32, prob=0.5),  # Cutout augmentation
    # dict(type='RandomAffine', scaling_ratio_range=(0.8, 1.2), prob=0.4),  # Random affine transformation
    # dict(type='RandomHSV', h_shift=0.1, s_shift=0.1, v_shift=0.1, prob=0.3),  # Random HSV shifts
    # dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),  # Normalize
    # dict(type='Pad', size_divisor=32),  # Pad images to ensure divisible by 32
    # dict(type='DefaultFormatBundle'),  # Bundle data in standard format
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),  # Collect the data needed for the model
]

train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            640,
            640,
        ),
        type='RandomResize'),
    dict(crop_size=(
        640,
        640,
    ), type='RandomCrop'),
    dict(type='YOLOXHSVRandomAug'),
    dict(prob=0.5, type='RandomFlip'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        640,
        640,
    ), type='Pad'),
    dict(type='PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.6, type='nms')),
    type='DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale=(
                    640,
                    640,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    320,
                    320,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    960,
                    960,
                ), type='Resize'),
            ],
            [
                dict(prob=1.0, type='RandomFlip'),
                dict(prob=0.0, type='RandomFlip'),
            ],
            [
                dict(
                    pad_val=dict(img=(
                        114,
                        114,
                        114,
                    )),
                    size=(
                        960,
                        960,
                    ),
                    type='Pad'),
            ],
            [
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                    ),
                    type='PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')

val_dataloader = dict(
    batch_size=5,
    dataset=dict(
        metainfo=metainfo,   ### BITIRME
        ann_file=val_ann_file,
        backend_args=None,
        data_prefix=dict(img=val_data_prefix),
        data_root=data_root,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

val_evaluator = dict(
    ann_file=val_eval_ann_file,
    backend_args=None,
    format_only=False,
    collect_device = "gpu",
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    # type='CocoMetric',
    type='YOLOStylePR',    ### BITIRME
    # iou_threshold = 0.5,  ### BITIRME
    # iou_thrs = 0.5,  ### BITIRME
    classwise = True,   ### BITIRME
)


test_cfg = dict(type='TestLoop')

test_dataloader = dict(
    batch_size=5,
    dataset=dict(
        metainfo=metainfo,   ### BITIRME
        ann_file=test_ann_file,
        backend_args=None,
        data_prefix=dict(img=test_data_prefix),
        data_root=data_root,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

test_evaluator = dict(
    ann_file=test_eval_ann_file,
    backend_args=None,
    format_only=False,
    collect_device = "gpu",
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    # type='CocoMetric',
    type='YOLOStylePR', ### BITIRME
    classwise = True,   ### BITIRME
    )


# test_dataloader = val_dataloader   ### BITIRME
# test_evaluator = val_evaluator     ### BITIRME

vis_backends = [
    dict(type='LocalVisBackend'),
]


# visualizer = dict(
#     name='visualizer',
#     type='DetLocalVisualizer',
#     vis_backends=[
#         dict(type='LocalVisBackend'),
#     ])
### BITIRME
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')]
    )

