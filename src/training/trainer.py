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
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        self.view_dirs = ['front', 'left', 'back', 'right', 'overhead', 'bottom']
        self.mesh_model = self.init_mesh_model()
        utils.log_mem_stat("init_mesh_model")
        self.diffusion = self.init_diffusion()
        utils.log_mem_stat("init_diffusion")
        self.text_z, self.text_string = self.calc_text_embeddings()
        self.dataloaders = self.init_dataloaders()
        utils.log_mem_stat("init_dataloaders")
        self.back_im = torch.Tensor(np.array(Image.open(self.cfg.guide.background_img).convert('RGB'))).to(
            self.device).permute(2, 0,
                                 1) / 255.0

        self.log_image_cnt = 0

        self.init_bad_size_threshold()

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    def init_bad_size_threshold(self):
        # otherwise some texture pixels will be hidden when from bad angle
        pixel_ratio = 2
        if (self.cfg.guide.texture_interpolation_mode == 'nereast'):
            pixel_ratio = 1
        # self.cfg.render.train_grid_size / 2 as image plane coordinates range is (-1, 1). Texture coordinates range is (0, 1)
        print("self.mesh_model.texture2mesh_ratio\n"+str(self.mesh_model.texture2mesh_ratio))
        self.bad_size_threshold = 1.4 * self.cfg.guide.texture_resolution / (self.mesh_model.texture2mesh_ratio *
            self.cfg.render.train_grid_size/2 * pixel_ratio)

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




    def adjust_camera(self, data: Dict[str, Any]):
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        fovyangle = np.pi/3
        # If offset of phi was set from code
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)

        # Render from viewpoint
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, fovyangle = fovyangle)
        min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs['mask'][0, 0])

        # TODO this should be same as 1/2 camera angle
        a = fovyangle / 2
        size = self.cfg.render.train_grid_size

        min_h_a  = utils.y2angle(min_h, a, size)
        max_h_a  = utils.y2angle(max_h, a, size)
        min_w_a  = utils.y2angle(min_w, a, size)
        max_w_a  = utils.y2angle(max_w, a, size)

        h_center = (min_h_a + max_h_a) / 2
        v_center = (min_w_a + max_w_a) / 2
        data['theta_adjustment'] = h_center
        data['phi_adjustment'] = -v_center
        data['fovyangle'] = max(math.fabs(max_h_a - min_h_a), math.fabs(max_w_a - min_w_a))

    def paint_viewpoint(self, data: Dict[str, Any]):
        logger.info(f'--- Painting step #{self.paint_step} ------------------------------------')
        self.adjust_camera(data)

        theta, phi, radius, theta_adjustment, phi_adjustment, fovyangle \
            = data['theta'], data['phi'], data['radius'], data['theta_adjustment'], data['phi_adjustment'], data['fovyangle']
        # If offset of phi was set from code
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}, fovyangle: {fovyangle}')

        # Set background image
        if self.cfg.guide.use_background_color:
            background = torch.Tensor([0, 0.8, 0]).to(self.device)
        else:
            background = F.interpolate(self.back_im.unsqueeze(0),
                                       (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size),
                                       mode='bilinear', align_corners=False)

        # Render from viewpoint
        outputs = self.mesh_model.render(theta=theta, phi=phi, elev_adjustment=theta_adjustment, azim_adjustment=phi_adjustment, fovyangle = fovyangle, radius=radius, background=background)
        render_cache = outputs['render_cache']
        rgb_render_raw = outputs['image']  # Render where missing values have special color
        depth_render = outputs['depth']
        # TODO this is still not accurate. 1) what we care about is actually minial width ratio in certain direction on x,y plane. Not area size, which is height * width
        # e.g for z_normal, it may reduce width by half and area size will only decrease by half, which is as bad as depth increased by 2, which decreases both height and width by half and decrease area size by 4.
        # in another word, this is propertional to depth^(-2) * tan(fovyangle/2)^(-2) * (z_normal - tan(angle_from_camera) * something :D). But we need depth * tan(fovyangle/2) * (z_normal - tan(angle_from_camera) * something :D)
        # 2) this is using average ratio of a face, this doesn't work if that face is very large. 
        # 3) And Because of 2), we can't fully fix 1) by just multiplying size_map with depth. As it'll make pixel with higher depth has larger size ratio comparing to other pxiel on the same face. Which is totally wrong as we want the opposite. 
        size_map = outputs['size_map']
        size_map = size_map * math.tan(fovyangle/2) * (-outputs["unnormalized_depth"])
        # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        outputs = self.mesh_model.render(background=background,
                                         render_cache=render_cache, use_median=self.paint_step > 1)
        rgb_render = outputs['image']
        # Render meta texture map
        meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=render_cache)
        
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        z_normals_none_clamp = outputs['normals'][:, -1:, :, :]
        logger.info("z_normals_none_clamp:" +str(z_normals_none_clamp.min())+" "+str(z_normals_none_clamp.max()))
        normals = outputs['normals'].clamp(0, 1)
        size_map_cache = meta_output['image'].clamp(0, 1)
        edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]

        logger.info("xiaofan z_normals shape:" + str(z_normals.shape))
        logger.info("xiaofan normal shape:" + str(outputs['normals'].shape))

        self.log_train_image(normals, "normal_clamp")
        self.log_train_image(rgb_render, 'rendered_input')

        temp = outputs["unnormalized_depth"][outputs["unnormalized_depth"]<0]
        logger.info("depth range "+str(temp.max())+" "+str(temp.min()))

        self.log_train_image(depth_render[0, 0], 'depth', colormap=True)
        self.log_train_image(z_normals[0, 0], 'z_normals', colormap=True)
        self.log_train_image(size_map_cache[0, 0], 'size_map_cache', colormap=True)


        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs]
            text_string = self.text_string[dirs]
        else:
            text_z = self.text_z
            text_string = self.text_string
        logger.info(f'text: {text_string}')

        update_mask, generate_mask, refine_mask, object_mask = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                                                                        depth_render=depth_render,
                                                                        z_normals=z_normals,
                                                                        size_map = size_map,
                                                                        size_map_cache=size_map_cache,
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

        #inpaint
        self.diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1
        inputs = rgb_render.detach()
        if self.paint_step == 1:
            inputs = None
        impaint_img, steps_vis = self.diffusion.img2img_step_with_controlnet(text_z, inputs,
                                                                    depth_render.detach(),
                                                                    guidance_scale=self.cfg.guide.guidance_scale,
                                                                    strength=1.0, update_mask=update_mask,
                                                                    refine_mask = refine_mask,
                                                                    fixed_seed=self.cfg.optim.seed,
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

        self.diffusion.use_inpaint = False
        img_latents, steps_vis = self.diffusion.img2img_step_with_controlnet(text_z, impaint_img,
                                                                depth_render.detach(),
                                                                guidance_scale=self.cfg.guide.guidance_scale,
                                                                strength=0.3, 
                                                                fixed_seed=self.cfg.optim.seed,
                                                                num_inference_steps = 20,
                                                                intermediate_vis=self.cfg.log.vis_diffusion_steps,
                                                                return_latent = True,
                                                                use_control_net = True,
                                                                random_init_latent = False,
                                                                update_mask = None,
                                                                )
        cropped_rgb_output = self.diffusion.decode_latents(img_latents)
        self.log_train_image(cropped_rgb_output, name='refine_output_without_inpaint')
        self.log_diffusion_steps(steps_vis, "refine_without_inpaint")

        cropped_rgb_output = self.diffusion.upscaler.do_upscale(utils.tensor2img_affecting_input(cropped_rgb_output))
        cropped_rgb_output = utils.image2tensor_affecting_input(cropped_rgb_output).to(self.device)


        logger.info("cropped_rgb_output.shape after upscale "+str(cropped_rgb_output.shape))
        self.log_train_image(cropped_rgb_output, name='scaled_output')

        # Extend rgb_output to full image size
        cropped_rgb_output = F.interpolate(cropped_rgb_output,
                                           (rgb_render.shape[2], rgb_render.shape[3]),
                                           mode='bilinear', align_corners=False)
        rgb_output = cropped_rgb_output
        self.log_train_image(rgb_output, name='full_output')

        # Project back
        fitted_pred_rgb, current_size_map_fitted = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                               object_mask=object_mask, update_mask=update_mask, z_normals=z_normals,
                                               size_map = size_map, size_map_cache=size_map_cache)
        self.log_train_image(fitted_pred_rgb, name='fitted')
        self.log_train_image(current_size_map_fitted[0, 0], 'current_z_normal_fitted', colormap=True)

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
                         depth_render: torch.Tensor, z_normals: torch.Tensor,
                         size_map: torch.Tensor, size_map_cache: torch.Tensor, edited_mask: torch.Tensor,
                         mask: torch.Tensor):
        object_mask = torch.ones_like(depth_render)
        object_mask[depth_render == 0] = 0
        # erode object mask to avoid background color leakaging into our texture. This mostly only happen when init latent is not randomized.
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((7, 7), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)
        #TODO this is not robust at all
        diff = (rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        exact_generate_mask = (diff < 0.1).float().unsqueeze(0)



        
        # small area kernel size shold at least be render_resolution/vae_input_size, which is 512 ??
        small_are_kernel_size = math.ceil(self.cfg.render.train_grid_size/512)
        small_area_kernel = np.ones((small_are_kernel_size, small_are_kernel_size), np.uint8)
        ## only expand when smaller than kernal to prevent over expanding generate mask
        # generate expand kernel size should be render_resolution/unet_latent.shape[-1]
        generate_expand_kernel = np.ones((19, 19), np.uint8)
        generate_mask = self.remove_small_area(exact_generate_mask, generate_expand_kernel)
        generate_mask = exact_generate_mask - generate_mask
        # Extpand generate mask
        generate_mask = torch.from_numpy(
            cv2.dilate(generate_mask[0, 0].detach().cpu().numpy(), generate_expand_kernel)).to(
            generate_mask.device).unsqueeze(0).unsqueeze(0)
        generate_mask[exact_generate_mask==1]=1

        generate_mask = self.fill_small_gap(generate_mask, small_area_kernel)
        generate_mask[object_mask ==0] = 0


        # Generate the refine mask based on the z normals, and the edited mask
        refine_mask = torch.zeros_like(depth_render)
        # TODO make this a config
        refine_mask[size_map > size_map_cache[:, :1, :, :] * 1.3] = 1
        # refine_mask[z_normals > z_normals_cache[:, :1, :, :] + self.cfg.guide.z_update_thr] = 1
        self.log_train_image(rgb_render_raw * refine_mask, name='refine_mask_step_0')
        refine_mask[generate_mask==1] = 0;


        # Xiaofan: TODO, clean up these logics
        if self.cfg.guide.initial_texture is None:
            logger.info("refine_mask[z_normals_cache[:, :1, :, :] == 0] = 0")
            # should be exact_generate_mask ?? they should be 1:1 mapping. Unless one of them is inaccurate
            refine_mask[size_map_cache[:, :1, :, :] == 0] = 0
        elif self.cfg.guide.reference_texture is not None:
            refine_mask[edited_mask == 0] = 0
            refine_mask = torch.from_numpy(
                cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((31, 31), np.uint8))).to(
                mask.device).unsqueeze(0).unsqueeze(0)
            refine_mask[mask == 0] = 0
            # Don't use bad angles here
            refine_mask[size_map < 0.4] = 0
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
        update_mask[(1-object_mask)==1] = 1
        
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
            self.log_train_image(trimap_vis, 'trimap')

        return update_mask, generate_mask, refine_mask, object_mask

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

    def calculate_transition_paddings(self, area):
        kernel_size = 21 # must be odd value
        dilated_area = torch.from_numpy(
            cv2.dilate(area[0, 0].detach().cpu().numpy(), np.ones((kernel_size, kernel_size), np.uint8))).to(
            area.device).unsqueeze(0).unsqueeze(0)
        blurred_area = utils.linear_blur(area, kernel_size//2)

        return blurred_area - area, dilated_area - blurred_area         

    def project_back(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                     size_map: torch.Tensor,
                     size_map_cache: torch.Tensor):
        render_update_mask = object_mask.clone()
        render_update_mask[update_mask == 0] = 0
        size_too_bad = size_map < self.bad_size_threshold
        render_update_mask[size_too_bad] = 0

        transition_update_mask, transition_keep_mask  = self.calculate_transition_paddings(render_update_mask)
        # Do not get out of the object. the object used here must be consitent, or even smaller, with object mask used in calculating trimap, as update mask also includes background.
        transition_update_mask[object_mask == 0] = 0
        transition_keep_mask[object_mask == 0] = 0

        temp_outputs = self.mesh_model.render(background=background,
                                             render_cache=render_cache)
        temp_rgb_render = temp_outputs['image']
        
        logger.info("z_normals range "+str(z_normals.min())+" "+ str(z_normals.max()))
        
        transition_update_mask[size_too_bad] = 0
        transition_keep_mask[size_too_bad] = 0

        logger.info("self.bad_size_threshold "+str(self.bad_size_threshold))

        # TODO size_map_cache shape currently is (1, 3, h, w). But actually we only need (1, 1, h, w)
        size_map_cache[:, 0, :, :] = size_map_cache[:, 0, :, :] * (1-render_update_mask)+ render_update_mask * size_map[:, 0, :, :]

        self.log_train_image(rgb_output * render_update_mask, 'project_update')
        self.log_train_image(rgb_output * transition_update_mask, 'project_transition')

        for i in range(30, 100, 5):
            self.log_train_image((size_map * (size_map < i/100.0))[0,0], 'size_map_'+str(i), colormap=True)

        self.log_train_image(temp_rgb_render * size_too_bad, 'size_too_bad')
        self.log_train_image(temp_rgb_render * transition_keep_mask, 'project_transition_keep')
        self.log_train_image(temp_rgb_render * (1-transition_update_mask-transition_keep_mask-render_update_mask), 'project_keep')
       
        optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99), eps=1e-15)

        nearest_loss_raito = 0.5
        
        render_update_mask = render_update_mask.flatten()
        transition_update_mask = transition_update_mask.flatten()
        transition_keep_mask = transition_keep_mask.flatten()

        old_texture_img = self.mesh_model.texture_img.detach().clone()
        unmasked_target = rgb_output.reshape(1, rgb_output.shape[1], -1).detach() * (render_update_mask+transition_update_mask)\
            + temp_rgb_render.reshape(1, temp_rgb_render.shape[1], -1).detach() * transition_keep_mask
            
        size_map_target = size_map_cache.reshape(1, size_map_cache.shape[1], -1)[:, :1]

        utils.log_mem_stat("before project back")
        for i in tqdm(range(200), desc='fitting mesh colors'):
            optimizer.zero_grad()
            def color_loss(mode, log=False):
                outputs = self.mesh_model.render(background=background, mode=mode, render_cache=render_cache)
                rgb_render = outputs['image']
                if i % 50 == 0 and log:
                    self.log_part_image(rgb_render, "part_rgb_render")
                    utils.log_mem_stat("fitting mesh colors")
                
                unmasked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)
                
                update_loss = ((unmasked_pred - unmasked_target).pow(2) * (transition_update_mask+transition_keep_mask+render_update_mask)).mean()
                keep_loss = (self.mesh_model.texture_img -old_texture_img).pow(2).mean()

                # TODO keep_loss actually shouldn't be put together with the rest until it's transformed using texture2mesh ratio and mesh2image ratio.
                return update_loss + 1e-4 * keep_loss

            loss = color_loss("bilinear", True) + nearest_loss_raito*color_loss("nearest")
            
            def size_map_loss(mode):
                meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                      use_meta_texture=True, render_cache=render_cache, mode=mode)
                size_map_pred = meta_outputs['image']
                real_size_map_pred = size_map_pred.reshape(1, size_map_pred.shape[1], -1)[:, :1]
                
                return ((real_size_map_pred - size_map_target.detach()).pow(2) * render_update_mask).mean()

            loss+= size_map_loss("bilinear") + nearest_loss_raito*size_map_loss("nearest")

            loss.backward()
            optimizer.step()


        fitted_rgb = self.mesh_model.render(background=background, render_cache=render_cache)['image']
        self.log_part_loss(fitted_rgb, temp_rgb_render, "part diff old")
        self.log_part_loss(fitted_rgb, rgb_output, "part diff new")
        self.log_part_image(fitted_rgb, "part_fitted_rgb")
        
        fitted_size_map = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                      use_meta_texture=True, render_cache=render_cache)['image']

        return fitted_rgb, fitted_size_map
    
    def log_part_image(self, tensor: torch.Tensor,  name: str, colormap=False):
        part_tensor = tensor[:, :, 1400:1410, 1420:1430]
        scaled_part = torch.nn.functional.interpolate(input = part_tensor, scale_factor = (16, 16))
        self.log_train_image(scaled_part, name, colormap)

        #self.log_train_image(tensor, name, colormap)

    def log_part_loss(self, a: torch.Tensor , b: torch.Tensor, name):
        x = a[:, :, 1400:1410, 1420:1430]
        y = b[:, :, 1400:1410, 1420:1430]
        loss = (x-y).pow(2).mean()
        logger.info(name +": "+str(loss))
        # print(str(x))
        # print(str(y))


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
