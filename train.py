import os
import re
import math
import time
import logging
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (FontDiffuserModel,
                 ContentPerceptualLoss,
                 build_unet,
                 build_style_encoder,
                 build_content_encoder,
                 build_ddpm_scheduler,
                 build_scr)
from utils import (save_args_to_yaml,
                   x0_from_epsilon, 
                   reNormalize_img, 
                   normalize_mean_std)


logger = get_logger(__name__)


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def find_latest_checkpoint(output_dir):
    """
    Find latest checkpoint folder under output_dir.
    Accept patterns like: global_step_5000, global-step-5000, global_step-5000, etc.
    Returns (path, step) or (None, 0) if none found.
    """
    if not os.path.isdir(output_dir):
        return None, 0
    candidates = []
    for entry in os.listdir(output_dir):
        full = os.path.join(output_dir, entry)
        if not os.path.isdir(full):
            continue
        m = re.search(r'global[-_]?step[-_]?(\d+)', entry, flags=re.IGNORECASE)
        if m:
            try:
                step = int(m.group(1))
                candidates.append((step, full))
            except Exception:
                continue
    if not candidates:
        return None, 0
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1], candidates[-1][0]


def load_model_weights_only(unet, style_encoder, content_encoder, ckpt_dir, map_location=torch.device("cpu")):
    """
    Load old-style separate .pth files (unet.pth, style_encoder.pth, content_encoder.pth).
    Non-fatal on missing files.
    """
    loaded_any = False
    try:
        unet_path = os.path.join(ckpt_dir, "unet.pth")
        if os.path.exists(unet_path):
            unet.load_state_dict(torch.load(unet_path, map_location=map_location))
            loaded_any = True
    except Exception as e:
        print(f"⚠️ Failed to load {unet_path}: {e}")

    try:
        style_path = os.path.join(ckpt_dir, "style_encoder.pth")
        if os.path.exists(style_path):
            style_encoder.load_state_dict(torch.load(style_path, map_location=map_location))
            loaded_any = True
    except Exception as e:
        print(f"⚠️ Failed to load {style_path}: {e}")

    try:
        content_path = os.path.join(ckpt_dir, "content_encoder.pth")
        if os.path.exists(content_path):
            content_encoder.load_state_dict(torch.load(content_path, map_location=map_location))
            loaded_any = True
    except Exception as e:
        print(f"⚠️ Failed to load {content_path}: {e}")

    return loaded_any


def load_composite_checkpoint_if_exists(model, optimizer, lr_scheduler, ckpt_dir, accelerator):
    """
    Try to load checkpoint.pth in ckpt_dir. If present, restore model + optimizer + scheduler + global_step.
    Returns restored_step (int) or 0.
    """
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
    if not os.path.exists(ckpt_path):
        return 0
    try:
        # Map to CPU first (accelerator will move to device), or map to accelerator.device
        map_loc = torch.device("cpu")
        # If GPU available on this machine, map to that to avoid extra transfers
        if torch.cuda.is_available():
            map_loc = accelerator.device
        checkpoint = torch.load(ckpt_path, map_location=map_loc)

        # Load model parts (they are raw state dicts)
        if "unet" in checkpoint:
            model.unet.load_state_dict(checkpoint["unet"])
        if "style_encoder" in checkpoint:
            model.style_encoder.load_state_dict(checkpoint["style_encoder"])
        if "content_encoder" in checkpoint:
            model.content_encoder.load_state_dict(checkpoint["content_encoder"])

        # optimizer & scheduler will be loaded after accelerator.prepare()
        restored_optimizer_state = checkpoint.get("optimizer", None)
        restored_scheduler_state = checkpoint.get("lr_scheduler", None)
        restored_step = checkpoint.get("global_step", 0)

        # Return a dict containing loaded state dicts and step for later restoration
        return {"step": restored_step, "optimizer": restored_optimizer_state, "lr_scheduler": restored_scheduler_state}
    except Exception as e:
        print(f"⚠️ Failed to load composite checkpoint {ckpt_path}: {e}")
        return 0


def save_composite_checkpoint(model, optimizer, lr_scheduler, global_step, save_dir):
    """
    Save the composite checkpoint dict (model tensors + optimizer + lr_scheduler + global_step)
    in save_dir/checkpoint.pth. Also keep separate model .pth files for backward compatibility.
    """
    os.makedirs(save_dir, exist_ok=True)
    # Save separate model weights (backward compatible)
    torch.save(model.unet.state_dict(), os.path.join(save_dir, "unet.pth"))
    torch.save(model.style_encoder.state_dict(), os.path.join(save_dir, "style_encoder.pth"))
    torch.save(model.content_encoder.state_dict(), os.path.join(save_dir, "content_encoder.pth"))
    # Save composite checkpoint with optimizer and scheduler
    composite = {
        "unet": model.unet.state_dict(),
        "style_encoder": model.style_encoder.state_dict(),
        "content_encoder": model.content_encoder.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "global_step": global_step
    }
    torch.save(composite, os.path.join(save_dir, "checkpoint.pth"))


def main():

    args = get_args()

    logging_dir = f"{args.output_dir}/{args.logging_dir}"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=f"{args.output_dir}/fontdiffuser_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # Set training seed
    if args.seed is not None:
        set_seed(args.seed)

    # -------------------------
    # Build model & noise scheduler (models constructed but not yet moved to device)
    # -------------------------
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)
    # If args.phase_2 and no resume checkpoint, previous code loads phase 1 ckpt below.

    # -------------------------
    # Check for existing checkpoint folders (latest)
    # -------------------------
    resume_ckpt_dir, resume_step = find_latest_checkpoint(args.output_dir)
    composite_restore_info = None
    resumed = False

    if resume_ckpt_dir:
        print(f"🔍 Found existing checkpoint directory: {resume_ckpt_dir} (detected step {resume_step})")
        # Try composite checkpoint load (model parts + optimizer/scheduler + step)
        composite_restore_info = load_composite_checkpoint_if_exists(
            model=type("tmp", (), {"unet": unet, "style_encoder": style_encoder, "content_encoder": content_encoder}),
            optimizer=None,
            lr_scheduler=None,
            ckpt_dir=resume_ckpt_dir,
            accelerator=accelerator
        )
        if composite_restore_info and isinstance(composite_restore_info, dict) and composite_restore_info.get("step", 0) > 0:
            # We loaded model state dicts into the temp model above; now copy into actual models
            # (in practice we loaded directly into the unet/style/content above)
            resumed = True
            print(f"🔁 Composite checkpoint detected; model weights loaded (will restore optimizer/scheduler after accelerator.prepare()).")
        else:
            # Try old-style model-weight-only load as fallback
            loaded_any = load_model_weights_only(unet, style_encoder, content_encoder, resume_ckpt_dir, map_location=torch.device("cpu"))
            if loaded_any:
                resumed = True
                print(f"🔁 Legacy model weights found and loaded from {resume_ckpt_dir} (optimizer/scheduler not available).")
            else:
                print(f"ℹ️ Found checkpoint dir but no usable checkpoint files were loaded. Will try phase_1_ckpt (if phase 2) or start from scratch.")

    else:
        print("ℹ️ No previous checkpoints found in output_dir.")

    # If not resuming from composite and this is phase_2, load phase_1 weights as original behavior
    if not resumed and args.phase_2:
        try:
            unet.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/unet.pth", map_location=torch.device("cpu")))
            style_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/style_encoder.pth", map_location=torch.device("cpu")))
            content_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/content_encoder.pth", map_location=torch.device("cpu")))
            print(f"Loaded Phase 1 checkpoint from {args.phase_1_ckpt_dir}")
            logging.info(f"Loaded Phase 1 checkpoint from {args.phase_1_ckpt_dir}")
        except Exception as e:
            print(f"⚠️ Failed to load Phase 1 ckpt: {e}")
            logging.warning(f"Failed to load Phase 1 ckpt: {e}")

    # Build model wrapper
    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder)

    # Build content perceptual Loss and move to device (avoid CPU-GPU transfers)
    perceptual_loss = ContentPerceptualLoss()
    try:
        perceptual_loss = perceptual_loss.to(accelerator.device)
        perceptual_loss.eval()
        for p in perceptual_loss.parameters():
            p.requires_grad = False
        print("✅ Perceptual loss moved to device and frozen.")
    except Exception as e:
        print(f"⚠️ Perceptual loss device move/freeze failed: {e}")

    # Load SCR module for supervision (Phase 2)
    if args.phase_2:
        scr = build_scr(args=args)
        try:
            scr.load_state_dict(torch.load(args.scr_ckpt_path, map_location=torch.device("cpu")))
            scr.requires_grad_(False)
        except Exception as e:
            print(f"⚠️ Failed to load SCR checkpoint {args.scr_ckpt_path}: {e}")
            logging.warning(f"Failed to load SCR checkpoint {args.scr_ckpt_path}: {e}")

    # -------------------------
    # Dataset & Dataloader
    # -------------------------
    content_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    style_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    target_transforms = transforms.Compose(
        [transforms.Resize((args.resolution, args.resolution), 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    train_font_dataset = FontDataset(
        args=args,
        phase='train', 
        transforms=[
            content_transforms, 
            style_transforms, 
            target_transforms],
        scr=args.phase_2)
    num_workers = min(8, (os.cpu_count() or 2) // 2)  # e.g., 4 if system has 8 cores
    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=CollateFN(),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    # Build optimizer and learning rate
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,)

    # Accelerate preparation (model/optimizer/dataloader/scheduler)
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)
    ## move scr module to the target devices
    if args.phase_2:
        scr = scr.to(accelerator.device)

    # If composite restore info exists (from earlier) - apply optimizer & scheduler states now
    restored_step = 0
    if isinstance(composite_restore_info, dict):
        # composite_restore_info was returned in earlier call if composite checkpoint present
        restored_step = int(composite_restore_info.get("step", 0))
        restored_opt_state = composite_restore_info.get("optimizer", None)
        restored_sched_state = composite_restore_info.get("lr_scheduler", None)
        if restored_opt_state is not None:
            try:
                optimizer.load_state_dict(restored_opt_state)
                print("✅ Optimizer state restored from composite checkpoint.")
            except Exception as e:
                print(f"⚠️ Failed to restore optimizer state: {e}")
        if restored_sched_state is not None and lr_scheduler is not None:
            try:
                lr_scheduler.load_state_dict(restored_sched_state)
                print("✅ LR scheduler state restored from composite checkpoint.")
            except Exception as e:
                print(f"⚠️ Failed to restore lr_scheduler state: {e}")
    else:
        # If not composite and we did find legacy saved model files, resume_step = resume_step (from find_latest_checkpoint)
        if resume_ckpt_dir and resume_step > 0:
            restored_step = resume_step
            # advance lr_scheduler by restored_step as a best-effort (since we don't have exact scheduler state)
            if lr_scheduler is not None and restored_step > 0:
                try:
                    print(f"⏩ Advancing LR scheduler by {restored_step} steps (best-effort) to align with resumed global_step...")
                    for _ in range(restored_step):
                        lr_scheduler.step()
                except Exception as e:
                    print(f"⚠️ Could not advance lr_scheduler: {e}")

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(args=args, output_file=f"{args.output_dir}/{args.experience_name}_config.yaml")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Convert to the training epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # initialize global_step based on restored_step
    global_step = int(restored_step) if restored_step else 0
    if global_step > 0:
        progress_bar.update(global_step)
        print(f"🔁 Resuming training from global_step = {global_step}")

    # Training loop
    for epoch in range(num_train_epochs):
        train_loss = 0.0
        for step, samples in enumerate(train_dataloader):
            model.train()
            content_images = samples["content_image"]
            style_images = samples["style_image"]
            target_images = samples["target_image"]
            nonorm_target_images = samples["nonorm_target_image"]
            
            with accelerator.accumulate(model):
                # Sample noise that we'll add to the samples
                noise = torch.randn_like(target_images)
                bsz = target_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=target_images.device)
                timesteps = timesteps.long()

                # Add noise to the target_images according to the noise magnitude at each timestep
                noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

                # Classifier-free training strategy (vectorized, device-aware)
                context_mask = torch.bernoulli(torch.full((bsz,), args.drop_prob, device=target_images.device)).bool()
                if context_mask.any():
                    content_images[context_mask] = 1.0
                    style_images[context_mask] = 1.0


                # Predict the noise residual and compute loss
                noise_pred, offset_out_sum = model(
                    x_t=noisy_target_images, 
                    timesteps=timesteps, 
                    style_images=style_images,
                    content_images=content_images,
                    content_encoder_downsample_size=args.content_encoder_downsample_size)
                diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                offset_loss = offset_out_sum / 2
                
                # output processing for content perceptual loss
                pred_original_sample_norm = x0_from_epsilon(
                    scheduler=noise_scheduler,
                    noise_pred=noise_pred,
                    x_t=noisy_target_images,
                    timesteps=timesteps)
                pred_original_sample = reNormalize_img(pred_original_sample_norm)
                norm_pred_ori = normalize_mean_std(pred_original_sample)
                norm_target_ori = normalize_mean_std(nonorm_target_images)
                percep_loss = perceptual_loss.calculate_loss(
                    generated_images=norm_pred_ori,
                    target_images=norm_target_ori,
                    device=target_images.device)
                
                loss = diff_loss + \
                        args.perceptual_coefficient * percep_loss + \
                            args.offset_coefficient * offset_loss
                
                if args.phase_2:
                    neg_images = samples["neg_images"]
                    # sc loss
                    sample_style_embeddings, pos_style_embeddings, neg_style_embeddings = scr(
                        pred_original_sample_norm, 
                        target_images, 
                        neg_images, 
                        nce_layers=args.nce_layers)
                    sc_loss = scr.calculate_nce_loss(
                        sample_s=sample_style_embeddings,
                        pos_s=pos_style_embeddings,
                        neg_s=neg_style_embeddings)
                    loss += args.sc_coefficient * sc_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.ckpt_interval == 0:
                        save_dir = f"{args.output_dir}/global_step_{global_step}"
                        os.makedirs(save_dir, exist_ok=True)
                        # Save model weights + composite checkpoint
                        try:
                            save_composite_checkpoint(model, optimizer, lr_scheduler, global_step, save_dir)
                            logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Saved composite checkpoint on global step {global_step}")
                            print("✅ Saved composite checkpoint on global step {}".format(global_step))
                        except Exception as e:
                            print(f"⚠️ Failed saving composite checkpoint: {e}")
                            # fallback: save separate model weights only
                            try:
                                torch.save(model.unet.state_dict(), f"{save_dir}/unet.pth")
                                torch.save(model.style_encoder.state_dict(), f"{save_dir}/style_encoder.pth")
                                torch.save(model.content_encoder.state_dict(), f"{save_dir}/content_encoder.pth")
                                torch.save(model, f"{save_dir}/total_model.pth")
                                logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Save the legacy checkpoint on global step {global_step}")
                                print("Saved legacy checkpoint on global step {}".format(global_step))
                            except Exception as e2:
                                print(f"‼️ Failed to save any checkpoint: {e2}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if global_step % args.log_interval == 0:
                logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Global Step {global_step} => train_loss = {loss}")
            progress_bar.set_postfix(**logs)
            
            # Quit
            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()

if __name__ == "__main__":
    main()