from PIL import Image
import os
from pathlib import Path

dataset_dirs = 'traindata' # Nama folder dataset
maximum_size_img = 256 # pixel
quality = 95

source = os.path.join(os.getcwd(), dataset_dirs) 
output = os.path.join(os.getcwd(), dataset_dirs+"_scaled") 
# Path(output).mkdir(exist_ok=True, parents=True)

for root, dirs, files in os.walk(dataset_dirs):
    for filename in files:
        foo = Image.open(os.path.join(root, filename))
        # print(source,img)
        foo = foo.convert('RGB')
        x, y = foo.size
        #print(x,y)
        x2 = min(maximum_size_img, x)
        y2 = int(y/x*x2)
        #print(x2,y2)
        foo = foo.resize((x2,y2),Image.ANTIALIAS)

        old_path = Path(root)
        new_path = os.path.join(output,old_path.relative_to(*old_path.parts[:1]))
        Path(new_path).mkdir(exist_ok=True, parents=True)
        print(os.path.join(new_path,filename))
        foo.save(os.path.join(new_path,filename),quality=quality,optimize=True)

