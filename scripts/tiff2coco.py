import numpy as np
import cv2
import json
from PIL import Image

# 类别标签
class_labels = {1: 'Grass', 2: 'Herb', 3: 'Litter', 4: 'Soil', 5: 'Stone', 6: 'Wood', 7: 'Woodchip'}

# 读取 TIFF 文件
tiff_path = '/Users/lu/Downloads/Labelled_image_7_classes_RGB_2022-10-20 gardens_munich essbare_Stadt_AB_YA_combined.tif'
tiff_image = Image.open(tiff_path)
tiff_np = np.array(tiff_image)

# 生成COCO格式的标注
coco_annotations = {
    "images": [],
    "annotations": [],
    "categories": []
}

# 添加类别信息到JSON
for class_id, class_name in class_labels.items():
    coco_annotations["categories"].append({
        "id": class_id,
        "name": class_name,
        "supercategory": "none"
    })

# 图像信息
image_id = 1
coco_annotations["images"].append({
    "id": image_id,
    "file_name": tiff_path.split('/')[-1],
    "height": tiff_np.shape[0],
    "width": tiff_np.shape[1]
})

annotation_id = 1
for class_id in class_labels.keys():
    # 创建掩码
    mask = (tiff_np == class_id).astype(np.uint8)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if len(contour) < 3:  # 跳过小轮廓
            continue
        
        segmentation = contour.flatten().tolist()
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        coco_annotations["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": class_id,
            "segmentation": [segmentation],
            "area": area,
            "bbox": [x, y, w, h],
            "iscrowd": 0
        })
        
        annotation_id += 1

# 保存为COCO格式的JSON文件
output_json_path = 'output_coco_annotations.json'
with open(output_json_path, 'w') as f:
    json.dump(coco_annotations, f, indent=4)

print(f"COCO annotations saved to {output_json_path}")