#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

# render查找定义
# gaussian_renderer/__init__.py: 这是渲染器的接口文件，它会判断当前是采用 CUDA 实现还是 Python 实现。
# 它内部调用了 diff_gaussian_rasterization 库（这是编译好的 C++/CUDA 扩展）。
# diff_gaussian_rasterization 库调用 Rasterizer::forward(...)
#调用了forward.cu
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask


# 3DGS 的损失函数设计非常讲究，它并不是简单的单一指标，
# 而是由颜色保真度（L1 Loss）和结构相似度（SSIM）加权组合而成的。

# L1 Loss (Ll1)：
# 作用：计算渲染图和真实图每个像素的绝对误差。
# 本质：它保证了颜色数值上的精确匹配。但单纯使用 L1 往往会导致渲染出的图像“偏糊”
# （因为像素级平均化会丢失高频纹理细节）。

# SSIM Loss (1.0 - ssim_value)：
# 作用：SSIM (Structural Similarity) 衡量的是图像的结构、亮度和对比度。
# 本质：它不仅看像素值，还看局部邻域内的纹理特征。
# 这就是为什么 3DGS 渲染出的物体看起来很有质感、很锐利，因为它强制模型去学习纹理和结构。

# 用 L1 控制颜色，用 SSIM 控制纹理和清晰度。

        # Loss

        #从内存中取出相机对应的原始高清照片，并同步到 GPU。这是作为“参考系”存在的。
        gt_image = viewpoint_cam.original_image.cuda()
        #image 是 render(...) 的输出。它是模型根据当前这 59 个参数渲染出的结果。
        Ll1 = l1_loss(image, gt_image)
        
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
# 这是干嘛的？ 
# 有时候仅靠照片渲染无法完全确定物体的三维位置，导致“漂浮物”或深度错误。
# 做法：如果该相机有深度数据，它会额外引入一个深度方向的 L1 Loss，
# 强迫高斯球的深度分布符合相机捕捉到的几何信息。

        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
#1. 监控与数据记录
    # 将当前的损失函数值、PSNR 等指标通过 TensorBoard 记录下来，并定期将整个高斯球模型参数保存到硬盘。
    # 这是训练的“仪表盘”，确保你能看到训练是否收敛，以及防止训练中途崩溃导致进度丢失。
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
#统计梯度
    # 它收集 dL_dmean2D（二维位置梯度）。
    # 如果某个高斯球在图像上产生的误差极大，说明它在当前视角下“位置摆放不准确”，需要重点关注。
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

# 克隆 (Clone)：如果某个高斯球在某个区域有很大的误差梯度，系统会在原地再复制一个。
# 分裂 (Split)：如果某个高斯球太“大”（覆盖区域过广），
# 说明它没能表示细节，系统将其分裂成两个更小的高斯球，从而提高渲染分辨率。
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

#执行参数更新
    #optimizer.step()  Adam 优化器根据刚才 backward() 计算出来的梯度，对高斯球的 59 个属性进行实打实的修改。



            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)

#use_sparse_adam 是一种加速优化器，它只对“可见”的高斯球进行更新，从而进一步减少计算开销。
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)#如果不清空梯度，梯度会在每一轮迭代中不断累加，导致参数更新失控（数值爆炸）。
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


# 1. prepare_output_and_logger：建立实验记录环境
# 它的主要作用是“给实验安个家”。
# 路径管理：
#     它会自动检查 args.model_path。如果没提供，它会生成一个基于 uuid（唯一识别码）的文件夹。
#     这非常重要：因为研究人员通常会运行多次训练对比不同参数，这样可以防止不同实验的文件相互覆盖。
# 配置备份：
#     with open(..., "cfg_args")：它把这一轮训练所有的超参数（如 learning rate, SH degree 等）完整保存到了配置文件里。
#     意义：这保证了可复现性。即使几个月后，你也能看懂当时这个实验到底是用什么参数跑出来的。
# TensorBoard 初始化：
#     它启动了 SummaryWriter，这是为了后续你能用可视化界面查看 Loss 曲线。
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# 2. training_report：阶段性的质量评估
# 它的作用是“定期的期中考试”。训练过程中，你无法只靠看 Loss 降低来判断模型质量，必须“渲染出来看看”。

# 阶段性评估 (Testing iterations)：
#     它是通过 if iteration in testing_iterations: 触发的。它不会在每一步都运行（那样太慢了
#     而是在关键节点（如第 7,000 次、第 30,000 次）跑一次测试。

# 验证集测试 (Validation)：
#     它不仅跑测试集 (test)，还抽样跑一部分训练集 (train)。
#     renderFunc：这里它会重新调用渲染器，但这次不再进行 backward（梯度计算），而是纯粹的 Inference。

# 质量量化 (PSNR / L1)：
#     它对比渲染出的 image 和真实的 gt_image。
#     PSNR (峰值信噪比) 是衡量图像恢复质量的黄金标准。
# 可视化反馈：
#     tb_writer.add_images(...)：这是最直观的，它直接把渲染出的图片传给 
#     TensorBoard，你可以通过网页直接看到模型渲染的效果如何。
# 模型状态监控：
#     tb_writer.add_histogram("scene/opacity_histogram", ...)：它监控了所有高斯球的透明度分布。
#     如果这个分布坍缩了，说明模型可能过拟合或者训练出现了异常。


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
