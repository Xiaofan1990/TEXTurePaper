from pickle import NONE
import kaolin as kal
import torch
import numpy as np
from loguru import logger
class Renderer:
    # from https://github.com/threedle/text2mesh

    def __init__(self, device, dim=(224, 224), interpolation_mode='nearest'):
        assert interpolation_mode in ['nearest', 'bilinear', 'bicubic'], f'no interpolation mode {interpolation_mode}'

        self.device = device
        self.interpolation_mode = interpolation_mode
        self.dim = dim
        self.background = torch.ones(dim).to(device).float()

    @staticmethod
    def get_xyz(elev, azim, r):
        x = r * torch.sin(elev) * torch.sin(azim)
        y = r * torch.cos(elev)
        z = r * torch.sin(elev) * torch.cos(azim)
        return x, y , z

    def get_camera_from_view(self, elev, azim, elev_adjustment=0, azim_adjustment=0 , r=3.0, look_at_height=0.0, fovyangle = np.pi / 3):
        x, y, z = Renderer.get_xyz(elev, azim, r)

        pos = torch.tensor([x, y, z]).unsqueeze(0)
        
        look_at = torch.zeros_like(pos)

        new_r = torch.sqrt((look_at_height - y) * (look_at_height - y) + x*x + z*z)
        camera_elev = torch.arccos((look_at_height - y) / new_r)
        camera_elev+=elev_adjustment

        dx, dy, dz = Renderer.get_xyz(camera_elev, azim+np.pi+azim_adjustment, new_r)
        look_at[:, 0] = x + dx
        look_at[:, 1] = y + dy
        look_at[:, 2] = z + dz

        direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

        camera_trans = kal.render.camera.generate_transformation_matrix(pos, look_at, direction).to(self.device)
        camera_proj = kal.render.camera.generate_perspective_projection(fovyangle).to(self.device)
        return camera_trans, camera_proj


    def normalize_depth(self, depth_map):
        assert depth_map.max() <= 0.0, 'depth map should be negative'
        object_mask = depth_map != 0
        # depth_map[object_mask] = (depth_map[object_mask] - depth_map[object_mask].min()) / (
        #             depth_map[object_mask].max() - depth_map[object_mask].min())
        # depth_map = depth_map ** 4
        min_val = 0.5
        depth_map[object_mask] = ((1 - min_val) * (depth_map[object_mask] - depth_map[object_mask].min()) / (
                depth_map[object_mask].max() - depth_map[object_mask].min())) + min_val

        # depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        # depth_map[depth_map == 1] = 0 # Background gets largest value, set to 0

        return depth_map

    def render_single_view(self, mesh, face_attributes, elev=0, azim=0, elev_adjustment=0, azim_adjustment =0, fovyangle = np.pi / 3, radius=2, look_at_height=0.0, calc_depth=True,dims=None, background_type='none'):
        dims = self.dim if dims is None else dims

        camera_trans, camera_proj = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                look_at_height=look_at_height, elev_adjustment = elev_adjustment, azim_adjustment=azim_adjustment, fovyangle = fovyangle)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            mesh.vertices.to(self.device), mesh.faces.to(self.device), camera_proj, camera_transform=camera_trans)

        if calc_depth:
            depth_map, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_vertices_camera[:, :, :, -1:])
            depth_map = self.normalize_depth(depth_map)
        else:
            depth_map = torch.zeros(1,64,64,1)

        image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_attributes)

        mask = (face_idx > -1).float()[..., None]
        if background_type == 'white':
            image_features += 1 * (1 - mask)
        if background_type == 'random':
            image_features += torch.rand((1,1,1,3)).to(self.device) * (1 - mask)

        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2), depth_map.permute(0, 3, 1, 2)


    def render_single_view_texture(self, verts, faces, uv_face_attr, texture_map, elev=0, azim=0, elev_adjustment = 0, azim_adjustment=0, fovyangle = np.pi / 3, radius=2,
                                   look_at_height=0.0, dims=None, background_type='none', render_cache=None, mode = None):
        if mode is None:
            mode = self.interpolation_mode
        
        dims = self.dim if dims is None else dims

        if render_cache is None:

            camera_trans, camera_proj = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), elev_adjustment = elev_adjustment, azim_adjustment=azim_adjustment, r=radius,
                                                    look_at_height=look_at_height, fovyangle = fovyangle)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                verts.to(self.device), faces.to(self.device), camera_proj, camera_transform=camera_trans)

            unnormalized_depth_map, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_vertices_camera[:, :, :, -1:])
            depth_map = unnormalized_depth_map.clone()
            depth_map = self.normalize_depth(depth_map)

            uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, uv_face_attr)
            uv_features = uv_features.detach()

        else:
            # logger.info('Using render cache')
            face_normals, uv_features, face_idx, unnormalized_depth_map, depth_map = \
                render_cache['face_normals'], render_cache['uv_features'], render_cache['face_idx'], render_cache['unnormalized_depth_map'], render_cache['depth_map']
        mask = (face_idx > -1).float()[..., None]

        image_features = kal.render.mesh.texture_mapping(uv_features, texture_map, mode=mode)
        image_features = image_features * mask

        if background_type == 'white':
            image_features += 1 * (1 - mask)
        elif background_type == 'random':
            image_features += torch.rand((1,1,1,3)).to(self.device) * (1 - mask)

        normals_image = face_normals[0][face_idx, :]

        render_cache = {'uv_features':uv_features, 'face_normals':face_normals,'face_idx':face_idx\
            , 'unnormalized_depth_map':unnormalized_depth_map, 'depth_map':depth_map}

        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2),\
               unnormalized_depth_map.permute(0, 3, 1, 2), depth_map.permute(0, 3, 1, 2), normals_image.permute(0, 3, 1, 2), render_cache

    def project_uv_single_view(self, verts, faces, uv_face_attr, elev=0, azim=0, radius=2,
                               look_at_height=0.0, dims=None, background_type='none'):
        # project the vertices and interpolate the uv coordinates

        dims = self.dim if dims is None else dims

        camera_trans, camera_proj = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                     look_at_height=look_at_height)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), faces.to(self.device), camera_proj, camera_transform=camera_trans)

        uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                          face_vertices_image, uv_face_attr)
        return face_vertices_image, face_vertices_camera, uv_features, face_idx

    def project_single_view(self, verts, faces, elev=0, azim=0, radius=2,
                               look_at_height=0.0):
        # only project the vertices
        camera_trans, camera_proj = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                     look_at_height=look_at_height)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), faces.to(self.device), camera_proj, camera_transform=camera_trans)

        return face_vertices_image
