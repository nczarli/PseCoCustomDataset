_base_ = "base.py"

num_classes = 3
dataset_type = 'CocoDataset'
classes = ('pitted', 'not_pitted', 'try_again')
model = dict(
    neck=dict(
        num_outs=6,
        add_extra_convs='on_input'
    ),
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4],
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=num_classes),
    ),
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                add_gt_as_proposals=False
            ),
        ),
    ),
)

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

strong_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
            type="RandResize",
            img_scale=[(1333, 400), (1333, 1200)],    
            multiscale_mode="range",
            keep_ratio=True),
            dict(
                type="ShuffledSequential",
                transforms=[
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type=k)
                            for k in [
                                "Identity",
                                "AutoContrast",
                                "RandEqualize",
                                "RandSolarize",
                                "RandColor",
                                "RandContrast",
                                "RandBrightness",
                                "RandSharpness",
                                "RandPosterize",
                            ]
                        ],
                    ),
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type="RandTranslate", x=(-0.1, 0.1)),
                            dict(type="RandTranslate", y=(-0.1, 0.1)),
                            dict(type="RandRotate", angle=(-30, 30)),
                            [
                                dict(type="RandShear", x=(-30, 30)),
                                dict(type="RandShear", y=(-30, 30)),
                            ],
                        ],
                    ),
                ],
            ),
            dict(
                type="RandErase",
                n_iterations=(1, 5),
                size=[0, 0.2],
                squared=True,
            ),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_student"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
            "flip",
            "flip_direction"
        ),
    ),
]
weak_pipeline = [
    dict(type="Sequential",
        transforms=[
        dict(type="RandFlip", flip_ratio=0.5),
        dict(
            type="RandResize",
            img_scale=[(1333, 400), (1333, 1200)],    
            multiscale_mode="range",
            keep_ratio=True,
        )],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_teacher"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
            "flip",
            "flip_direction"
        ),
    ),
]

unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    dict(type="PseudoSamples", with_bbox=True),
    dict(
        type="MultiBranch", unsup_student=strong_pipeline, unsup_teacher=weak_pipeline
    ),
]


data = dict(
    samples_per_gpu=5,
    workers_per_gpu=2,
    train=dict(
        sup=dict(
            type = dataset_type,
            classes=classes,
            ann_file="labels_generated/train/annotations/instances_default.json",
            img_prefix="labels_generated/train/images/",

        ),
        unsup=dict(
            type = dataset_type,
            classes=classes,
            ann_file="labels_generated/unlabelled/annotations/instances_default.json",
            img_prefix="labels_generated/unlabelled/images/",
            pipeline=unsup_pipeline,
        ),
    ),
    val=dict(
      type = dataset_type,
      classes=classes,
      ann_file="labels_generated/test/annotations/instances_default.json",
      img_prefix="labels_generated/test/images/",
    ),
    test=dict(
      type = dataset_type,
      classes=classes,
      ann_file="labels_generated/test/annotations/instances_default.json",
      img_prefix="labels_generated/test/images/",
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

semi_wrapper = dict(
    type="PseCo_FRCNN",
    model="${model}",
    train_cfg=dict(
        pseudo_label_initial_score_thr=0.3,
        rpn_pseudo_threshold=0.3,
        cls_pseudo_threshold=0.3,
        min_pseduo_box_size=0,
        unsup_weight=2.0,
        use_teacher_proposal=True,    
        use_MSL=True,
        # ------ PLA config ------- #
        PLA_iou_thres=0.4,
        PLA_candidate_topk=12,
    ),
    test_cfg=dict(
        inference_on="teacher"
        ),
)

fold = 1
percent = 1

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.999, warm_up=0),
    dict(type="GetCurrentIter")
]


find_unused_parameters=False 
backend="disk"

evaluation = dict(type="SubModulesDistEvalHook", interval=200, start=200)
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=400)
