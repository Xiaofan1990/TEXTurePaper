from pathlib import Path
from typing import Any, Dict, Union, List

import cv2
import einops
import imageio
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from matplotlib import cm
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from src import utils
from src.configs.train_config import TrainConfig
from src.models.textured_mesh import TexturedMeshModel
from src.stable_diffusion_depth import StableDiffusion
from src.training.views_dataset import ViewsDataset, MultiviewDataset
from src.utils import make_path, tensor2numpy


class TEXTure:

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        
        # otherwise some texture pixels will be hidden when from bad angle
        # assert(texture_resolution <= train_grid_size * (bad_normal * 0.9) and bilinear) or 
        # or assert(texture_resolution <= train_grid_size * 2 * (bad_normal * 0.9) and nerest)

        self.paint_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        utils.seed_everything(self.cfg.optim.seed)

        # Make view_dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        self.init_logger()
        utils.log_mem_stat()

        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        self.view_dirs = ['front', 'left', 'back', 'right', 'overhead', 'bottom']
        self.mesh_model = self.init_mesh_model()
        utils.log_mem_stat()
        self.diffusion = self.init_diffusion()
        utils.log_mem_stat()
        self.text_z, self.text_string = self.calc_text_embeddings()
        self.dataloaders = self.init_dataloaders()
        utils.log_mem_stat()
        self.back_im = torch.Tensor(np.array(Image.open(self.cfg.guide.background_img).convert('RGB'))).to(
            self.device).permute(2, 0,
                                 1) / 255.0

        self.log_image_cnt = 0


        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

        utils.log_mem_stat()

    def init_mesh_model(self) -> nn.Module:
        cache_path = Path('cache') / Path(self.cfg.guide.shape_path).stem
        cache_path.mkdir(parents=True, exist_ok=True)
        model = TexturedMeshModel(self.cfg.guide, device=self.device,
                                  render_grid_size=self.cfg.render.train_grid_size,
                                  cache_path=cache_path,
                                  texture_resolution=self.cfg.guide.texture_resolution,
                                  augmentations=False)

        model = model.to(self.device)
        logger.info(
            f'Loaded Mesh, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    def init_diffusion(self) -> Any:
        diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
                                          concept_name=self.cfg.guide.concept_name,
                                          concept_path=self.cfg.guide.concept_path,
                                          latent_mode=False,
                                          min_timestep=self.cfg.optim.min_timestep,
                                          max_timestep=self.cfg.optim.max_timestep,
                                          no_noise=self.cfg.optim.no_noise,
                                          use_inpaint=True, trainer = self)

        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        negative_prompt = self.cfg.guide.negative_prompt
        if not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text], [negative_prompt])
            text_string = ref_text
        else:
            text_z = []
            text_string = []
            for d in self.view_dirs:
                text = ref_text.format(d)
                text_string.append(text)
                logger.info(text)
                logger.info(negative_prompt)
                text_z.append(self.diffusion.get_text_embeds([text], [negative_prompt]))
        return text_z, text_string

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        init_train_dataloader = MultiviewDataset(self.cfg.render, device=self.device).dataloader()

        val_loader = ViewsDataset(self.cfg.render, device=self.device,
                                  size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = ViewsDataset(self.cfg.render, device=self.device,
                                        size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'train': init_train_dataloader, 'val': val_loader,
                       'val_large': val_large_loader}
        return dataloaders

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def paint(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.mesh_model.train()

        pbar = tqdm(total=len(self.dataloaders['train']), initial=self.paint_step,
                    bar_format='{desc}: {percentage:3.0f}% painting step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for data in self.dataloaders['train']:
            self.paint_step += 1
            pbar.update(1)
            self.paint_viewpoint(data)
            self.evaluate(self.dataloaders['val'], self.eval_renders_path)
            self.mesh_model.train()

            # TODO: Xiaofan: remove this.
            #if self.paint_step > 4:
            #    break;

        self.mesh_model.change_default_to_median()
        logger.info('Finished Painting ^_^')
        logger.info('Saving the last result...')
        self.full_eval()
        logger.info('\tDone!')

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):
        logger.info(f'Evaluating and saving model, painting iteration #{self.paint_step}...')
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video:
            all_preds = []
        for i, data in enumerate(dataloader):
            preds, textures, depths, normals = self.eval_render(data)

            # logger.info('Xiaofan before!')
            # logger.info(preds[0])
            pred = tensor2numpy(preds[0])
            # logger.info('Xiaofan after!')
            # logger.info(pred)


            if save_as_video:
                all_preds.append(pred)
            else:
                Image.fromarray(pred).save(save_path / f"step_{self.paint_step:05d}_{i:04d}_rgb.jpg")
                Image.fromarray((cm.seismic(normals[0, 0].cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(
                    save_path / f'{self.paint_step:04d}_{i:04d}_normals_cache.jpg')
                if self.paint_step == 0:
                    # Also save depths for debugging
                    torch.save(depths[0], save_path / f"{i:04d}_depth.pt")

        # Texture map is the same, so just take the last result
        # logger.info('Xiaofan texture!')
        # logger.info(textures[0])
        texture = tensor2numpy(textures[0])
        logger.info('Saving last result in ' + save_path.__str__())
        Image.fromarray(texture).save(save_path / f"step_{self.paint_step:05d}_texture.png")

        if save_as_video:
            logger.info('Saving As Video! ' + save_path.__str__())
            all_preds = np.stack(all_preds, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"step_{self.paint_step:05d}_{name}.mp4", video,
                                                           fps=25,
                                                           quality=8, macro_block_size=1)

            dump_vid(all_preds, 'rgb')
        logger.info('Done!')

    def full_eval(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = self.final_renders_path
        self.evaluate(self.dataloaders['val_large'], output_dir, save_as_video=True)
        # except:
        #     logger.error('failed to save result video')

        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / 'mesh')
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path)

            logger.info(f"\tDone!")

    def paint_viewpoint(self, data: Dict[str, Any]):
        logger.info(f'--- Painting step #{self.paint_step} ---')
        utils.log_mem_stat();
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        # If offset of phi was set from code
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}')

        # Set background image
        if self.cfg.guide.use_background_color:
            background = torch.Tensor([0, 0.8, 0]).to(self.device)
        else:
            background = F.interpolate(self.back_im.unsqueeze(0),
                                       (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size),
                                       mode='bilinear', align_corners=False)

        # Render from viewpoint
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background)
        render_cache = outputs['render_cache']
        rgb_render_raw = outputs['image']  # Render where missing values have special color
        depth_render = outputs['depth']
        # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        outputs = self.mesh_model.render(background=background,
                                         render_cache=render_cache, use_median=self.paint_step > 1)
        rgb_render = outputs['image']
        # Render meta texture map
        meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=render_cache)

                                     

                                                
        x_normals = outputs['normals'][:, -3:, :, :].clamp(0, 1)
        y_normals = outputs['normals'][:, -2:, :, :].clamp(0, 1)
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        normals = outputs['normals'].clamp(0, 1)
        z_normals_cache = meta_output['image'].clamp(0, 1)
        edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]

        logger.info("xiaofan z_normals shape:" + str(z_normals.shape))
        logger.info("xiaofan normal shape:" + str(outputs['normals'].shape))

        self.log_train_image(outputs['normals'], "normal_no_clamp")
        self.log_train_image(normals, "normal_clamp")
        self.log_train_image(rgb_render, 'rendered_input')
        self.log_train_image(depth_render[0, 0], 'depth', colormap=True)
        self.log_train_image(x_normals[0, 0], 'x_normals', colormap=True)
        self.log_train_image(y_normals[0, 0], 'y_normals', colormap=True)
        self.log_train_image(z_normals[0, 0], 'z_normals', colormap=True)
        self.log_train_image(z_normals_cache[0, 0], 'z_normals_cache', colormap=True)


        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs]
            text_string = self.text_string[dirs]
        else:
            text_z = self.text_z
            text_string = self.text_string
        logger.info(f'text: {text_string}')

        update_mask, generate_mask, refine_mask = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                                                                        depth_render=depth_render,
                                                                        z_normals=z_normals,
                                                                        z_normals_cache=z_normals_cache,
                                                                        edited_mask=edited_mask,
                                                                        mask=outputs['mask'])

        update_ratio = float(update_mask.sum() / (update_mask.shape[2] * update_mask.shape[3]))
        if self.cfg.guide.reference_texture is not None and update_ratio < 0.01:
            logger.info(f'Update ratio {update_ratio:.5f} is small for an editing step, skipping')
            return
        utils.log_mem_stat('before log traing image');

        self.log_train_image(rgb_render * (1 - update_mask), name='masked_input')
        self.log_train_image(rgb_render * refine_mask, name='refine_regions')
        utils.log_mem_stat('after log traing image');
        # Crop to inner region based on object mask
        min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs['mask'][0, 0])
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        cropped_rgb_render = crop(rgb_render)
        cropped_depth_render = crop(depth_render)
        cropped_update_mask = crop(update_mask)
        self.log_train_image(cropped_rgb_render, name='cropped_input')
        checker_mask = None

        #inpaint
        if self.paint_step > 1:
            checker_mask = self.generate_checkerboard(crop(update_mask), crop(refine_mask),
                                                      crop(generate_mask))
            self.log_train_image(F.interpolate(cropped_rgb_render, (512, 512)) * (1 - checker_mask),
                                 'checkerboard_input')
        self.diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1
        inputs = cropped_rgb_render.detach()
        if self.paint_step == 1:
            inputs = None
        impaint_img, steps_vis = self.diffusion.img2img_step_with_controlnet(text_z, inputs,
                                                                    cropped_depth_render.detach(),
                                                                    guidance_scale=self.cfg.guide.guidance_scale,
                                                                    strength=1.0, update_mask=cropped_update_mask,
                                                                    refine_mask = refine_mask,
                                                                    fixed_seed=self.cfg.optim.seed,
                                                                    check_mask=checker_mask,
                                                                    num_inference_steps = 20,
                                                                    random_init_latent = True,
                                                                    intermediate_vis=self.cfg.log.vis_diffusion_steps)
        self.log_train_image(impaint_img, name='inpaint_out')
        self.log_diffusion_steps(steps_vis, "inpaint")
        logger.info("impaint.shape\n"+str( impaint_img.shape )+str(impaint_img.dtype))
        #img2img
        generator = torch.manual_seed(self.cfg.optim.seed)
        with torch.autocast('cuda'):
            cropped_rgb_output = self.diffusion.img2img_pipe(
                prompt = text_string,
                negative_prompt = self.cfg.guide.negative_prompt,
                num_inference_steps=20, 
                generator=generator, 
                guidance_scale=self.cfg.guide.guidance_scale,
                image=impaint_img, 
                strength=0.3,
            ).images[0]

            self.log_train_image(cropped_rgb_output, name='img2img_out')

        # mine img2img Xiaofan: if not using impaint, pleasde remove update mask as well.
        self.diffusion.use_inpaint = True
        img_latents, steps_vis = self.diffusion.img2img_step_with_controlnet(text_z, impaint_img,
                                                                cropped_depth_render.detach(),
                                                                guidance_scale=self.cfg.guide.guidance_scale,
                                                                strength=0.3, 
                                                                fixed_seed=self.cfg.optim.seed,
                                                                num_inference_steps = 20,
                                                                intermediate_vis=self.cfg.log.vis_diffusion_steps,
                                                                return_latent = True,
                                                                use_control_net = True,
                                                                random_init_latent = False,
                                                                update_mask = cropped_update_mask,
                                                                )
        cropped_rgb_output = self.diffusion.decode_latents(img_latents)
        self.log_train_image(cropped_rgb_output, name='refine_output')
        logger.info("cropped_rgb_output.shape "+str(cropped_rgb_output.shape))
        
        self.log_diffusion_steps(steps_vis, "refine")

        

        #cropped_rgb_output = self.diffusion.upscaler(
        #     prompt = text_string,
        #     negative_prompt = self.cfg.guide.negative_prompt,
        #     image=img_latents,
        #     num_inference_steps=20,
        #     guidance_scale=7.5,
        #     generator = generator,
        #     output_type = "not pil :d"
        #).images[0]
        #cropped_rgb_output = torch.from_numpy(cropped_rgb_output)
        #cropped_rgb_output = torch.reshape(cropped_rgb_output, (1, cropped_rgb_output.shape[0], cropped_rgb_output.shape[1], -1))
        #cropped_rgb_output = cropped_rgb_output.permute((0, 3, 1, 2))
        #cropped_rgb_output = cropped_rgb_output.to(device=self.device)


        logger.info("cropped_rgb_output.shape after upscale"+str(cropped_rgb_output.shape))
        self.log_train_image(cropped_rgb_output, name='scaled_output')

        cropped_rgb_output = F.interpolate(cropped_rgb_output,
                                           (cropped_rgb_render.shape[2], cropped_rgb_render.shape[3]),
                                           mode='bilinear', align_corners=False)

        # Extend rgb_output to full image size
        rgb_output = rgb_render.clone()
        rgb_output[:, :, min_h:max_h, min_w:max_w] = cropped_rgb_output
        self.log_train_image(rgb_output, name='full_output')

        # Project back
        object_mask = outputs['mask']
        fitted_pred_rgb, current_z_normal_fitted = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                               object_mask=object_mask, update_mask=update_mask, z_normals=z_normals,
                                               z_normals_cache=z_normals_cache)
        self.log_train_image(fitted_pred_rgb, name='fitted')
        self.log_train_image(current_z_normal_fitted[0, 0], 'current_z_normal_fitted', colormap=True)

        return

    def eval_render(self, data):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        dim = self.cfg.render.eval_grid_size
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                         dims=(dim, dim), background='white')
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        rgb_render = outputs['image']  # .permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        diff = (rgb_render.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        uncolored_mask = (diff < 0.1).float().unsqueeze(0)
        rgb_render = rgb_render * (1 - uncolored_mask) + utils.color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals,
                                                                                light_coef=0.3) * uncolored_mask

        outputs_with_median = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                     dims=(dim, dim), use_median=True,
                                                     render_cache=outputs['render_cache'])

        meta_output = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                             background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=outputs['render_cache'])
        pred_z_normals = meta_output['image'][:, :1].detach()
        rgb_render = rgb_render.permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        texture_rgb = outputs_with_median['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        depth_render = outputs['depth'].permute(0, 2, 3, 1).contiguous().detach()

        return rgb_render, texture_rgb, depth_render, pred_z_normals

    # remove small pointed gaps from area. Which will help avoid grey patterns painted by inpaint.        
    def fill_small_gap(self, mask, kernel):
        mask = torch.from_numpy(
            cv2.dilate(mask[0, 0].detach().cpu().numpy(), kernel)).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        mask = torch.from_numpy(
            cv2.erode(mask[0, 0].detach().cpu().numpy(), kernel)).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        return mask

    def remove_small_area(self, mask, kernel):
        mask = torch.from_numpy(
            cv2.erode(mask[0, 0].detach().cpu().numpy(), kernel)).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        mask = torch.from_numpy(
            cv2.dilate(mask[0, 0].detach().cpu().numpy(), kernel)).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        return mask

    def calculate_trimap(self, rgb_render_raw: torch.Tensor,
                         depth_render: torch.Tensor,
                         z_normals: torch.Tensor, z_normals_cache: torch.Tensor, edited_mask: torch.Tensor,
                         mask: torch.Tensor):
        diff = (rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        # xiaofan why to float()?
        exact_generate_mask = (diff < 0.1).float().unsqueeze(0)


        ## only expand when smaller than kernal to prevent over expanding generate mask
        # generate expand kernel size should be render_resolution/unet_latent.shape[-1]
        # small area kernel size shold at least be render_resolution/vae_input_size, which is 512 ??
        small_are_kernel_size = math.ceil(self.cfg.render.train_grid_size/512)
        small_area_kernel = np.ones((small_are_kernel_size, small_are_kernel_size), np.uint8)
        generate_expand_kernel = np.ones((19, 19), np.uint8)
        generate_mask = self.remove_small_area(exact_generate_mask, generate_expand_kernel)
        generate_mask = exact_generate_mask - generate_mask
        # Extpand generate mask
        generate_mask = torch.from_numpy(
            cv2.dilate(generate_mask[0, 0].detach().cpu().numpy(), generate_expand_kernel)).to(
            generate_mask.device).unsqueeze(0).unsqueeze(0)
        generate_mask[exact_generate_mask==1]=1

        generate_mask = self.fill_small_gap(generate_mask, small_area_kernel)
        
        object_mask = torch.ones_like(depth_render)
        object_mask[depth_render == 0] = 0
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((7, 7), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)

        # Generate the refine mask based on the z normals, and the edited mask

        refine_mask = torch.zeros_like(depth_render)
        refine_mask[z_normals > z_normals_cache[:, :1, :, :] + self.cfg.guide.z_update_thr] = 1
        self.log_train_image(rgb_render_raw * refine_mask, name='refine_mask_step_0')
        refine_mask[generate_mask==1] = 0;


        # Xiaofan: TODO, clean up these logics
        if self.cfg.guide.initial_texture is None:
            logger.info("refine_mask[z_normals_cache[:, :1, :, :] == 0] = 0")
            # should be exact_generate_mask ?? they should be 1:1 mapping. Unless one of them is inaccurate
            refine_mask[z_normals_cache[:, :1, :, :] == 0] = 0
        elif self.cfg.guide.reference_texture is not None:
            refine_mask[edited_mask == 0] = 0
            refine_mask = torch.from_numpy(
                cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((31, 31), np.uint8))).to(
                mask.device).unsqueeze(0).unsqueeze(0)
            refine_mask[mask == 0] = 0
            # Don't use bad angles here
            refine_mask[z_normals < 0.4] = 0
            logger.info("Don't use bad angles here!!!!!!")
        else:
            # Update all regions inside the object
            refine_mask[mask == 0] = 0
        self.log_train_image(rgb_render_raw * refine_mask, name='refine_mask_step_1')
        
        refine_n_generate_mask = refine_mask.clone()
        refine_n_generate_mask[generate_mask==1] = 1
        refine_n_generate_mask = self.remove_small_area(refine_n_generate_mask, small_area_kernel)
        refine_n_generate_mask = self.fill_small_gap(refine_n_generate_mask, small_area_kernel)
        
        refine_mask = refine_n_generate_mask - generate_mask
        refine_mask[object_mask==0] = 0

        self.log_train_image(rgb_render_raw * refine_mask, name='refine_mask_step_2')

        update_mask = generate_mask.clone()
        update_mask[refine_mask == 1] = 1
        
        # Visualize trimap
        if self.cfg.log.log_images:
            trimap_vis = utils.color_with_shade(color=[112 / 255.0, 173 / 255.0, 71 / 255.0], z_normals=z_normals)
            trimap_vis[mask.repeat(1, 3, 1, 1) == 0] = 1
            trimap_vis = trimap_vis * (1 - generate_mask) + utils.color_with_shade(
                [255 / 255.0, 22 / 255.0, 67 / 255.0],
                z_normals=z_normals,
                light_coef=0.7) * generate_mask

            shaded_rgb_vis = rgb_render_raw.detach()
            shaded_rgb_vis = shaded_rgb_vis * (1 - generate_mask) + utils.color_with_shade([0.85, 0.85, 0.85],
                                                                                                 z_normals=z_normals,
                                                                                                 light_coef=0.7) * generate_mask

            if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
                refinement_color_shaded = utils.color_with_shade(color=[91 / 255.0, 155 / 255.0, 213 / 255.0],
                                                                 z_normals=z_normals)
                only_old_mask_for_vis = torch.bitwise_and(update_mask == 1, generate_mask == 0).float().detach()
                trimap_vis = trimap_vis * 0 + 1.0 * (trimap_vis * (
                        1 - only_old_mask_for_vis) + refinement_color_shaded * only_old_mask_for_vis)
            self.log_train_image(shaded_rgb_vis, 'shaded_input')
            self.log_train_image(trimap_vis, 'trimap')

        return update_mask, generate_mask, refine_mask

    def generate_checkerboard(self, update_mask_inner, improve_z_mask_inner, update_mask_base_inner):
        checkerboard = torch.ones((1, 1, 64 // 2, 64 // 2)).to(self.device)
        # Create a checkerboard grid
        checkerboard[:, :, ::2, ::2] = 0
        checkerboard[:, :, 1::2, 1::2] = 0
        checkerboard = F.interpolate(checkerboard,
                                     (512, 512))
        checker_mask = F.interpolate(update_mask_inner, (512, 512))
        only_old_mask = F.interpolate(torch.bitwise_and(improve_z_mask_inner == 1,
                                                        update_mask_base_inner == 0).float(), (512, 512))
        checker_mask[only_old_mask == 1] = checkerboard[only_old_mask == 1]
        return checker_mask

    def project_back(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                     z_normals_cache: torch.Tensor):
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)
        render_update_mask = object_mask.clone()

        render_update_mask[update_mask == 0] = 0

        blurred_render_update_mask = render_update_mask
        #blurred_render_update_mask = torch.from_numpy(
        #    cv2.dilate(render_update_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
        #    render_update_mask.device).unsqueeze(0).unsqueeze(0)
        #blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 21, 16)

        # Do not get out of the object
        blurred_render_update_mask[object_mask == 0] = 0

        #if self.cfg.guide.strict_projection:
        #    blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
        #    #Xiaofan: why we need this? didn't we already consdierred this in update mask?
        #    # Do not use bad normals
        #    z_was_better = z_normals + self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :]
        #    blurred_render_update_mask[z_was_better] = 0
        
        transition_mask = 1- blurred_render_update_mask

        # TODO ideally we should not only consider z_normal, but also distance? But if we consider distance, some point will never be painted. As it may be far from one angle but not visible from another angle even if close. 
        z_is_too_bad = z_normals<0.51 # should from config
        blurred_render_update_mask[z_is_too_bad] = 0
        transition_mask[z_is_too_bad] = 0


        temp_outputs = self.mesh_model.render(background=background,
                                             render_cache=render_cache)
        temp_rgb_render = temp_outputs['image']
        self.log_train_image(rgb_output * blurred_render_update_mask, 'project_update')
        self.log_train_image(rgb_output * transition_mask, 'project_transition')
        self.log_train_image(temp_rgb_render * (1-transition_mask-blurred_render_update_mask), 'project_keep')

        # Update the max normals
        # z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :])
        z_normals_cache[:, 0, :, :] = z_normals_cache[:, 0, :, :] * (1-update_mask)+ update_mask * z_normals[:, 0, :, :]


        optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99), eps=1e-15)

        nearest_loss_raito = 0.1
        
        project_update_mask = blurred_render_update_mask.flatten()
        project_transition_mask = transition_mask.flatten()
        old_texture_img = self.mesh_model.texture_img.detach().clone()

        for i in tqdm(range(200), desc='fitting mesh colors'):
            optimizer.zero_grad()
            def color_loss(mode, project_update_mask, project_transition_mask):
                outputs = self.mesh_model.render(background=background, mode=mode, render_cache=render_cache)
                rgb_render = outputs['image']
                if i % 50 == 0:
                    self.log_part_image(rgb_render, "part_rgb_render")
                    
                

                unmasked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)
                unmasked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)
                
                update_loss = ((unmasked_pred - unmasked_target.detach()).pow(2) * project_update_mask).mean()
                tranistion_loss = ((unmasked_pred - unmasked_target.detach()).pow(2) * project_transition_mask).mean()
                keep_loss = (self.mesh_model.texture_img -old_texture_img).pow(2).mean()

                return update_loss + 1e-4 * (keep_loss + 0.4 * tranistion_loss)

            loss = color_loss("bilinear", project_update_mask, project_transition_mask) + nearest_loss_raito*color_loss("nearest", project_update_mask, project_transition_mask)
            
            def z_normals_loss(mode, project_update_mask):
                meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                      use_meta_texture=True, render_cache=render_cache, mode=mode)
                current_z_normals = meta_outputs['image']
                current_z_mask = meta_outputs['mask'].flatten()
                # combined_mask = torch.bitwise_and(current_z_mask, update_mask)
                masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :1]
                masked_best_z_normal_map = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :1]

                return ((masked_current_z_normals - masked_best_z_normal_map.detach()).pow(2) * current_z_mask * project_update_mask).mean()

            loss+= z_normals_loss("bilinear", project_update_mask) + nearest_loss_raito*z_normals_loss("nearest", project_update_mask)

            loss.backward()
            optimizer.step()


        fitted_rgb = self.mesh_model.render(background=background, render_cache=render_cache)['image']
        self.log_part_loss(fitted_rgb, temp_rgb_render, "part loss")
        self.log_part_image(fitted_rgb, "part_fitted_rgb")
        
        fitted_z_normals = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                      use_meta_texture=True, render_cache=render_cache)['image']

        return fitted_rgb, fitted_z_normals
    
    def log_part_image(self, tensor: torch.Tensor,  name: str, colormap=False):
        # part_tensor = tensor[:, :, 375:443, 534:600]
        # scaled_part = torch.nn.functional.interpolate(input = part_tensor, scale_factor = (16, 16))
        # self.log_train_image(scaled_part, name, colormap)

        self.log_train_image(tensor, name, colormap)

    def log_part_loss(self, a: torch.Tensor , b: torch.Tensor, name):
        x = a[:, :, 440:443, 534:537]
        y = b[:, :, 440:443, 534:537]
        loss = (x-y).pow(2).mean()
        logger.info(name +": "+str(loss))


    def log_train_image(self, tensor, name: str, colormap=False):
        self.log_image_cnt += 1
        image = None
        if self.cfg.log.log_images:
            if isinstance(tensor, torch.Tensor):
                if colormap:
                    tensor = cm.seismic(tensor.detach().cpu().numpy())[:, :, :3]
                else:
                    tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
                image = Image.fromarray((tensor * 255).astype(np.uint8))
            else:
                image = tensor
            if image.mode == 'RGBA':
                image = image.convert('RGB')    
            image.save(
                self.train_renders_path / f'{self.paint_step:04d}_{self.log_image_cnt:04d}_{name}.jpg')

    def log_diffusion_steps(self, intermediate_vis: List[Image.Image], stage):
        if len(intermediate_vis) > 0:
            step_folder = self.train_renders_path / f'{self.paint_step:04d}_{stage}_diffusion_steps'
            step_folder.mkdir(exist_ok=True)
            for k, intermedia_res in enumerate(intermediate_vis):
                intermedia_res.save(
                    step_folder / f'{k:02d}_diffusion_step.jpg')

    def save_image(self, tensor: torch.Tensor, path: Path):
        if self.cfg.log.log_images:
            Image.fromarray(
                (einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy() * 255).astype(np.uint8)).save(
                path)
