import torch
import torchvision

class VideoProcessor(torch.nn.Module):
    def __init__(self, device, video_chunk_size = 50):
        self.device = device
        self.video_chunk_size = video_chunk_size

    def process_video(self, video, target_video_length_in_frames):
        full_video = self._remove_empty_frames(video)
        selected_frames = self._select_frames(full_video, target_video_length_in_frames)
        
        return selected_frames
    
    def load_video_frames(self, folder_path):
        video, _, fps_data = torchvision.io.read_video(folder_path)
        
        print(video.shape, fps_data)

        return video, fps_data['video_fps']
    
    def _select_frames(self, video, target_video_length_in_frames):
        video = video.to(self.device)     
        filtered_video_frames = []     
        processed_frames_count = 0
            
        for i in range(target_video_length_in_frames):
            frames_left_to_fill = target_video_length_in_frames - i
            total_frames_left = video.shape[0] - processed_frames_count
            frames_chunk_size = round(total_frames_left / frames_left_to_fill)
            
            frames = video[processed_frames_count : processed_frames_count + frames_chunk_size]
            filtered_video_frames.append(frames[0])
            
            processed_frames_count += frames_chunk_size
            
        filtered_video_frames = torch.stack(filtered_video_frames)    

        return filtered_video_frames.to('cpu')

    def _remove_empty_frames(self, video):
        black_pixel_percentage_threshold = 0.35
        black_pixel_iluminosity_threshold = 1

        filtered_video_frames = []
        
        video = video.to(self.device)
        
        for i in range(0, video.size(0), self.video_chunk_size):
            chunk = video[i : i + self.video_chunk_size]
            
            video_tensor_float = chunk.float()
            
            grayscale_tensor = (
                0.21 * video_tensor_float[:, :, :, 0]
                + 0.72 * video_tensor_float[:, :, :, 1]
                + 0.07 * video_tensor_float[:, :, :, 2]
            )
            
            grayscale_tensor = grayscale_tensor.unsqueeze(dim=3)
            
            black_pixel_count = (
                grayscale_tensor > black_pixel_iluminosity_threshold
            ).sum(dim=(1, 2, 3)).float()
            
            black_pixel_percentage = black_pixel_count / (180 * 320)

            frames_to_keep = black_pixel_percentage > black_pixel_percentage_threshold

            filtered_chunk = chunk[frames_to_keep]
            filtered_video_frames.append(filtered_chunk)

        filtered_video_tensor = torch.cat(filtered_video_frames, dim=0)

        return filtered_video_tensor.to('cpu')
        

            