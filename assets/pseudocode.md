```python
def eval_pointcloud(pred_cloud: PointCloud,
                    gt_reconstruction: Union[PointCloud, Mesh],
                    results: ResultsDict
                    ):

    if align_clouds and align_scale:
        rot, t, s = align(pred_cloud.pose, gt_cloud.pose)
        pred_cloud = s * rot * pred_cloud + t

    if align_clouds and not align_scale:
        rot, t = align(pred_cloud.pose, gt_cloud.pose)
        pred_cloud = rot * pred_cloud + t

    if crop_pred:
        pred_cloud = pred_cloud.crop(ref_cloud.bounding_box)
    if crop_ref:
        ref_cloud = ref_cloud.crop(pred_cloud.bounding_box)

    if register_clouds:
        if use_color_registration:
            icp_transform = colored_icp_register_both_clouds(pred_cloud, gt_cloud)
        else:
            icp_transform = icp_register_both_clouds(pred_cloud, gt_cloud)

        pred_cloud = pred_cloud.transform(icp_transform)

    acc = compute_accuracy(pred_cloud, gt_cloud)
    comp = compute_completeness(pred_cloud, gt_cloud)
    chamfer_dist = (acc + comp) / 2
    prec = compute_precision(pred_cloud, gt_cloud)
    recall = compute_recall(pred_cloud, gt_cloud)
    f_score = (2 * prec * recall) / (prec + recall)

    results.add(acc, comp, chamfer_dist, prec, recall, f_score)
```

```python
def eval_poses(pred_poses: List[Trajectory],
               ref_pose: Trajectory,
               results: ResultsDict
               ):
    # ORB-SLAM is not deterministic due to RANSAC, multithreading, and keyframe selection.
    # So we average over multiple runs of the same scene with the same parameters.
    for pred_pose in pred_poses:
        if align_scale:
            rot, t, s = align(pred_pose, ref_pose)
            pred_pose = s * rot * pred_pose + t
        else:
            rot, t = align(pred_pose, ref_pose)
            pred_pose = rot * pred_pose + t

        ate = compute_absolute_traj_error(pred_pose, ref_pose)
        rpe_rot, rpe_t = compute_relative_pose_error(pred_pose, ref_pose)

        results.add(ate, rpe_rot, rpe_t)

    results.average()
```


```python
def eval_depths(dataset: CustomDataset,
                cameras: List[int],
                sequences: List[int],
                depth_model: DepthModel,
                ):
    for cam_id in cameras:
        for seq_id in sequences:
            images, poses = dataset.get(cam_id, seq_id)
            images = get_static_images(poses) # KITTI360 has no static frames in poses

            for img_batch in batch(images):
                depth = depth_model.pred(img_batch)
                gt_depth = dataset.get_gt_depth(cam_id, seq_id, img_batch)
                mask = gt_depth > 0

                # Averages over valid pixels in current batch
                metrics = compute_metrics(depth, gt_depth, mask)
                results.add(seq_id, metrics)

            results.average_over_images()

        results.average_over_sequences()

    results.average_over_cameras()
```

```python
def eval_splats(...):
    # TODO think about this and figure it out, probably need to run multiple times as well
    pass
```




