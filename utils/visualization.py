import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_bev_plot(
    gt_points: Optional[torch.Tensor] = None,
    pred_points: Optional[torch.Tensor] = None,
    worldgrid_size: Tuple[int, int] = (480, 1440),
    gt_color: str = 'green',
    pred_color: str = 'blue',
    gt_alpha: float = 0.7,
    pred_alpha: float = 0.7,
    gt_size: int = 50,
    pred_size: int = 50,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100,
    show_grid: bool = True,
    grid_alpha: float = 0.3,
    show_statistics: bool = True,
    title: Optional[str] = None,
    label_fontsize: int = 16,
    legend_fontsize: int = 14,
    title_fontsize: int = 18
) -> np.ndarray:
    """Create bird's eye view visualization with improved styling.
    
    Args:
        gt_points: Ground truth points tensor of shape [N, 2]
        pred_points: Predicted points tensor of shape [M, 2]
        worldgrid_size: Size of world grid (H, W)
        gt_color: Color for ground truth points
        pred_color: Color for predicted points
        gt_alpha: Alpha transparency for ground truth points
        pred_alpha: Alpha transparency for prediction points
        gt_size: Size of ground truth points
        pred_size: Size of prediction points
        figsize: Figure size (width, height)
        dpi: DPI for the figure
        show_grid: Whether to show grid
        grid_alpha: Grid transparency
        show_statistics: Whether to show statistics text
        title: Optional title for the plot
        
    Returns:
        Bird's eye view visualization as numpy array
    """
    try:
        # Create figure with white background
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='white')
        
        # Convert tensors to numpy arrays
        gt_points_np = None
        pred_points_np = None
        # import ipdb; ipdb.set_trace()
        
        if gt_points is not None and len(gt_points) > 0:
            gt_points_np = gt_points.detach().cpu().numpy()
            
        if pred_points is not None and len(pred_points) > 0:
            pred_points_np = pred_points.detach().cpu().numpy()
        
        # Plot ground truth points
        if gt_points_np is not None and len(gt_points_np) > 0:
            ax.scatter(
                gt_points_np[:, 1], gt_points_np[:, 0],  # Note: x,y order for BEV
                c=np.array([gt_color])/255., alpha=gt_alpha, s=gt_size,
                label=f'Ground Truth',
                edgecolors='darkgreen', linewidth=0.5
            )
        
        # Plot prediction points
        if pred_points_np is not None and len(pred_points_np) > 0:
            ax.scatter(
                pred_points_np[:, 1], pred_points_np[:, 0],  # Note: x,y order for BEV
                c=np.array([pred_color])/255., alpha=pred_alpha, s=pred_size,
                label=f'Predictions',
                edgecolors='darkblue', linewidth=0.5
            )
        
        # # Set labels
        # ax.set_xlabel('X Coordinate', fontsize=label_fontsize)
        # ax.set_ylabel('Y Coordinate', fontsize=label_fontsize)
        
        # Add legend
        # if (gt_points_np is not None and len(gt_points_np) > 0) or \
        #    (pred_points_np is not None and len(pred_points_np) > 0):
        #     ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        #     ax.legend(loc='upper right', fontsize=legend_fontsize, framealpha=0.9)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Add grid
        if show_grid:
            ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5)
            # Make grid more dense by setting minor grid
        # Set axis limits based on worldgrid size
        ax.set_xlim(0, worldgrid_size[1])  # Width
        ax.set_ylim(0, worldgrid_size[0])  # Height
        # Invert y-axis to match image coordinates (0 at top)
        ax.invert_yaxis()
        
        # Set background to white
        ax.set_facecolor('white')
        
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure to a temporary buffer and load as image
        import io
        from PIL import Image
        
        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        
        # Load with PIL and convert to numpy
        pil_img = Image.open(buf)
        img_array = np.array(pil_img)
        
        # Convert RGB to BGR for OpenCV compatibility
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Close the figure to free memory
        plt.close(fig)
        buf.close()
        
        return img_array
        
    except Exception as e:
        print(f"Error creating BEV plot: {str(e)}")
        # Return a simple fallback image
        fallback_img = np.full((worldgrid_size[0], worldgrid_size[1], 3), 255, dtype=np.uint8)
        cv2.putText(fallback_img, f"Error: {str(e)}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return fallback_img

import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional

def draw_boxes(
    image: np.ndarray,
    boxes: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    is_normalized: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes on an image.
    
    Args:
        image: numpy array of shape (H, W, C)
        boxes: tensor of shape (N, 4) in [x1, y1, x2, y2] format
        labels: optional tensor of shape (N,) containing label indices
        scores: optional tensor of shape (N,) containing confidence scores
        color: BGR color tuple for boxes
        thickness: line thickness
        is_normalized: whether box coordinates are normalized [0-1]
    
    Returns:
        Image with drawn boxes
    """
    if len(boxes) == 0:
        return image
        
    img_h, img_w = image.shape[:2]
    boxes_np = boxes.detach().cpu().numpy()
    
    if is_normalized:
        boxes_np[:, [0, 2]] *= img_w
        boxes_np[:, [1, 3]] *= img_h
    
    boxes_np = boxes_np.astype(np.int32)
    
    for i, box in enumerate(boxes_np):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        if labels is not None and scores is not None:
            label = labels[i].item()
            score = scores[i].item()
            text = f"Person {score:.2f}"
            cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, thickness)
            
    return image

def visualize_predictions(
    image: np.ndarray,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: Optional[torch.Tensor] = None,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Visualize predictions and ground truth boxes on an image.
    
    Args:
        image: numpy array of shape (H, W, C)
        pred_boxes: predicted boxes tensor (N, 4) in [x1,y1,x2,y2] format
        pred_scores: prediction confidence scores (N,)
        gt_boxes: optional ground truth boxes tensor (M, 4)
        threshold: confidence threshold for showing predictions
    
    Returns:
        Image with visualizations
    """
    # Make copy of image
    vis_image = image.copy()
    
    # Draw ground truth boxes in green
    if gt_boxes is not None:
        vis_image = draw_boxes(
            vis_image, 
            gt_boxes,
            color=(0, 255, 0),
            thickness=2
        )
    
    # Filter predictions by threshold
    mask = pred_scores > threshold
    pred_boxes = pred_boxes[mask]
    pred_scores = pred_scores[mask]
    # Draw prediction boxes in red
    vis_image = draw_boxes(
        vis_image,
        pred_boxes,
        scores=pred_scores,
        color=(0, 0, 255), 
        thickness=2
    )
    
    return vis_image

def create_combined_visualization(
    camera_images: List[np.ndarray],
    bev_img: np.ndarray,
    frame_id: int,
    camera_titles: Optional[List[str]] = None,
    bev_scale: float = 1.5  # Scale factor for BEV size
) -> np.ndarray:
    """Create a combined visualization with camera views and BEV.
    
    Args:
        camera_images: List of camera view images
        bev_img: Bird's eye view image
        frame_id: Current frame ID
        camera_titles: Optional list of titles for camera views
        bev_scale: Scale factor for BEV size (1.0 = original size)
        
    Returns:
        Combined visualization image
    """
    n_cameras = len(camera_images)
    if n_cameras == 0:
        return bev_img
        
    # Get camera image dimensions
    cam_h, cam_w = camera_images[0].shape[:2]
    bev_h, bev_w = bev_img.shape[:2]
    
    # Scale BEV dimensions
    bev_h_scaled = int(bev_h * bev_scale)
    bev_w_scaled = int(bev_w * bev_scale)
    bev_img_scaled = cv2.resize(bev_img, (bev_w_scaled, bev_h_scaled))
    
    # Calculate grid layout
    n_cols = min(3, n_cameras)  # Max 3 cameras per row
    n_rows = (n_cameras + n_cols - 1) // n_cols
    
    # Create canvas with space for BEV at the bottom
    canvas_w = max(n_cols * cam_w, bev_w_scaled)
    canvas_h = n_rows * cam_h + bev_h_scaled + 20  # Add padding between cameras and BEV
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    
    # Place camera views
    for i, img in enumerate(camera_images):
        row = i // n_cols
        col = i % n_cols
        y1 = row * cam_h
        x1 = col * cam_w
        
        # Resize if needed
        if img.shape[:2] != (cam_h, cam_w):
            img = cv2.resize(img, (cam_w, cam_h))
            
        canvas[y1:y1+cam_h, x1:x1+cam_w] = img
        
        # Add camera title
        if camera_titles and i < len(camera_titles):
            cv2.putText(canvas, camera_titles[i], (x1+10, y1+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Place BEV at the bottom with padding
    y_bev = n_rows * cam_h + 20  # Add padding
    x_bev = (canvas_w - bev_w_scaled) // 2  # Center BEV horizontally
    canvas[y_bev:y_bev+bev_h_scaled, x_bev:x_bev+bev_w_scaled] = bev_img_scaled
    
    # Add frame counter
    cv2.putText(canvas, f"Frame: {frame_id}", (10, canvas_h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return canvas

def extract_bev_data(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    worldgrid_size: Tuple[int, int],
    threshold: float = 0.5,
    max_objects: int = 60
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract BEV ground truth and prediction points from model outputs.
    
    Args:
        outputs: Model outputs dictionary
        targets: Ground truth targets dictionary
        worldgrid_size: Size of the world grid
        threshold: Confidence threshold for predictions
        max_objects: Maximum number of objects to consider
    
    Returns:
        Tuple of (gt_points, pred_points) in absolute coordinates
    """
    device = list(outputs.values())[0].device
    
    # Extract ground truth points
    gt_points = None
    if 'bev_pids' in targets and 'bev_pts' in targets:
        num_targets = len(torch.where(targets["bev_pids"][0] != 0)[0])
        if num_targets > 0:
            target_points_norm = targets['bev_pts'][0][0:num_targets]  # reverse x,y
            worldgrid_tensor = torch.tensor(worldgrid_size, dtype=torch.float32, device=device)
            gt_points = target_points_norm * worldgrid_tensor
    
    # Extract prediction points
    pred_points = None
    if 'pred_logits' in outputs and 'pred_bev' in outputs:
        output_logits = outputs['pred_logits']  # [1, N, 2]
        output_bevs = outputs['pred_bev']       # [1, N, 2]
        
        prob = output_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(output_logits.shape[0], -1), max_objects, dim=1)
        scores = topk_values
        topk_points = topk_indexes // output_logits.shape[2]
        
        bev_points_absolute = torch.gather(output_bevs, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
        
        # Apply threshold
        mask = scores >= threshold
        bev_points_absolute = bev_points_absolute[mask]
        
        if len(bev_points_absolute) > 0:
            pred_points = bev_points_absolute.squeeze(0)
    
    return gt_points, pred_points 