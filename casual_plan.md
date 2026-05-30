训练构建
在Casual_CoAF下 新建一个训练目录，目录结构参考：
/project/llmsvgen/sunkai/robomaster_3d/CoAF/training
但只新建、保留下面功能需要的脚本
训练配置：
暂时无法在飞书文档外展示此内容
数据数量： 5000 条
跑 6w步 每 5000 存一步

训练的模型
为我构建以下训练脚本，主要目标是将相同的训练脚本，扩大到5000条数据，60k的规模上：
只训练Casual的模型，非因果的模型都不跑了；因为必定是要做成因果的
先训 4 个，按优先级排序：
1. 目标视频是：Depth + RGB：Depth的视频24帧，RGB视频25帧 / 分辨率 480 * 640
  1. 训练数据目录：/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset/composed/v4_depth_rgb
2. 目标视频是：Depth + RGB：Depth的视频24帧，RGB视频25帧
  1. 训练数据目录：/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset/composed/v4_depth_rgb
3. 目标视频时：Pose + RGB：Pose的视频24帧，RGB视频25帧
  1. 训练数据目录：/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset/composed/v1_pose_rgb
4. 目标视频是：Pose + Depth + RGB：Depth的视频24帧，RGB视频25帧
  1. 训练数据目录：/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset/composed/v5_pose_depth_rgb
5. 目标视频是：Flow+ RGB：Flow的视频24帧，RGB视频25帧
  1. 训练数据目录：/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset/composed/v2_flow_rgb

可以参考下面的训练脚本，进行构建：
/project/llmsvgen/sunkai/robomaster_3d/CoAF/training/cog_video_training/jobs/train/i2v/train_coaf_i2v_target_rgb_pose_cond_rgb_text_causal.sbatch
包括模型路径等

