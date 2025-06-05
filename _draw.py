from PIL import Image, ImageDraw

# Load your image
image_path = "/home/feyad/Desktop/staj/aug_1_f32553be5bcc6700092eb471ae0f3fbc.png"
image = Image.open(image_path).convert("RGB")

# Define bounding boxes in [x, y, width, height] format
bboxes = [
    [
        364.52737289957565,
        1689.9043308081993,
        419.17530217523665,
        1779.0714708014582
    ],
]

# Create a drawing context
draw = ImageDraw.Draw(image)

# Draw each bounding box
for bbox in bboxes:
    xmin, ymin, xmax, ymax = bbox
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

# Save the image with bounding boxes
output_path = "./draw1.png"
image.save(output_path)
print(f"Saved image with bounding boxes to: {output_path}")
