from __future__ import annotations

import torch


def get_rope_index(
    spatial_merge_size: int,
    input_ids: torch.LongTensor,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Build the Qwen3-VL position ids used by the original training job."""

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    position_ids = torch.ones(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    image_index = 0
    video_index = 0

    for batch_index, tokens_tensor in enumerate(input_ids):
        tokens = tokens_tensor.tolist()
        vision_starts = torch.argwhere(tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = tokens_tensor[vision_starts + 1]
        image_count = int((vision_tokens == image_token_id).sum())
        video_count = int((vision_tokens == video_token_id).sum())
        remaining_images = image_count
        remaining_videos = video_count
        segments = []
        start = 0

        for _ in range(image_count + video_count):
            image_end = tokens.index(image_token_id, start) if remaining_images else len(tokens) + 1
            video_end = tokens.index(video_token_id, start) if remaining_videos else len(tokens) + 1
            if image_end < video_end:
                time, height, width = image_grid_thw[image_index]
                image_index += 1
                remaining_images -= 1
                end = image_end
            else:
                time, height, width = video_grid_thw[video_index]
                video_index += 1
                remaining_videos -= 1
                end = video_end

            time = int(time)
            height = int(height) // spatial_merge_size
            width = int(width) // spatial_merge_size
            text_length = end - start
            offset = segments[-1].max() + 1 if segments else 0
            segments.append(torch.arange(text_length).view(1, -1).expand(3, -1) + offset)

            time_ids = torch.arange(time).view(-1, 1).expand(-1, height * width).flatten()
            height_ids = torch.arange(height).view(1, -1, 1).expand(time, -1, width).flatten()
            width_ids = torch.arange(width).view(1, 1, -1).expand(time, height, -1).flatten()
            segments.append(
                torch.stack([time_ids, height_ids, width_ids]) + text_length + offset
            )
            start = end + time * height * width

        if start < len(tokens):
            offset = segments[-1].max() + 1 if segments else 0
            text_length = len(tokens) - start
            segments.append(torch.arange(text_length).view(1, -1).expand(3, -1) + offset)

        position_ids[:, batch_index, :] = torch.cat(segments, dim=1).to(position_ids.device)

    return position_ids
