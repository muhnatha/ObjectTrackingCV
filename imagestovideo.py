import cv2
import glob

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (240, 320))

img_paths = sorted(glob.glob('pandasDataset/images/*.jpg'))

#img_path = 'pandasDataset/images/img0002.jpg'
#img = cv2.imread(img_path)
#print(img.shape)

if not img_paths:
    print("No images found in the folder!")
else:
    print(f"Found {len(img_paths)} images.")

for img_path in img_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        continue
    img = cv2.resize(img, (240, 320))
    out.write(img)

out.release()
print("Video saved as output.mp4")
