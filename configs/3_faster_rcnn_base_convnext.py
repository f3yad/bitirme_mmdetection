######################
## WORKING ConvNext ##
######################

# imports
custom_imports = dict(
        imports = [
            'mmpretrain.models',
        ],
        allow_failed_imports=False
    )

###############################################################
### model settings
###############################################################
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmpretrain.ConvNeXt',    ### bitirme
        arch="tiny",
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        gap_before_final_norm=False,    ### bitirme
        # norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=None,    ### bitirme
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],    ### bitirme
        # in_channels=[256, 512, 1024, 2048], ### ResNet base
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0, class_weight=2.0),  ## add class_weight
        loss_bbox=dict(type='L1Loss', loss_weight=5.0)), ## 1
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=7,    ### bitirme
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0, class_weight=2.0),  ## add class_weight
        loss_bbox=dict(type='L1Loss', loss_weight=5.0)), ## 1
            #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            # loss_bbox=dict(type='L1Loss', loss_weight=1.0))
        ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001, # 0.05  conf  ### bitirme
            nms=dict(type='nms', iou_threshold=0.7), # 0.5  iou  ### bitirme
            max_per_img=300 # 100  max-dets    ### bitirme
        )
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    )
)





###############################################################
### dataset settings
###############################################################
dataset_type = 'CocoDataset'
backend_args = None

data_root = 'datasets/spinexr/'
metainfo = dict(classes=(
    'Osteophytes',           # id 0
    'Spondylolysthesis',     # id 1
    'Disc space narrowing',  # id 2
    'Other lesions',         # id 3
    'Surgical implant',      # id 4
    'Foraminal stenosis',    # id 5
    'Vertebral collapse'     # id 6
))

train_anns = "train_fixed.json"
val_anns = "val_fixed.json"
test_anns = "test_fixed.json"

train_dir = "train_images/"
val_dir = "val_images/"
test_dir = "test_images/"

### delete
# train_anns = val_anns
# train_dir = val_dir


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,    ### bitirme
    num_workers=1,    ### bitirme
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        metainfo=metainfo,
        type=dataset_type,
        data_root=data_root,
        ann_file=train_anns,    ### bitirme
        data_prefix=dict(img=train_dir),    ### bitirme
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args
    )

)
val_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo=metainfo,
        type=dataset_type,
        data_root=data_root,
        ann_file=val_anns,    ### bitirme
        data_prefix=dict(img=val_dir),    ### bitirme
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

# test_dataloader = dict(
#     batch_size=4,
#     num_workers=1,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=test_anns,    ### bitirme
#         data_prefix=dict(img=test_dir),    ### bitirme
#         test_mode=True,
#         pipeline=test_pipeline
#     )
# )

val_evaluator = dict(
    type='CocoYoloMetric',
    ann_file=data_root + val_anns,
    metric='bbox',
    classwise=True,
    collect_device="gpu",
    format_only=False,
    backend_args=backend_args,
    ### bitirme
    # iou_thrs=[0.7],
    # proposal_nums=[300]
)

# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     classwise=True,
#     collect_device="gpu",
#     ann_file=data_root + test_anns,    ### bitirme
#     outfile_prefix='./work_dirs/spinexr/test',    ### bitirme
#     # iou_thrs=[0.7],               # 👈 IoU threshold
# )

test_evaluator = val_evaluator
test_dataloader = val_dataloader




###############################################################
### default runtime
###############################################################
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),    ### bitirme
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
    ### bitirme 
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',  # AP @[IoU=0.50:0.95] metriği
        patience=20,              # 20 epoch patience
        min_delta=0.001,           # Minimum değişim eşiği
        rule='greater'           # Metrik büyük olması için 'greater'
    )
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False






###############################################################
### training schedule for 1x
###############################################################
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)    ### bitirme
