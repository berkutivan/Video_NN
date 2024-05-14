import os
from PIL import Image
from colorama import init
from video_functions import *
import torch
from pathlib import Path
from VNN import UNet, Generator


video_path = 'videos/video_from_youtube.mp4'
new_video_path = 'polongation_videos/'
images_path = 'data'
comprise_img_path = 'C:/Users/berku/Desktop/semestr_4/VNN/comprise_data'
prolog_img_path = 'C:/Users/berku/Desktop/semestr_4/VNN/new_data'
state = "params_less_1.pt"
RESCALE_SIZE_X , RESCALE_SIZE_Y = 448, 256
size = (RESCALE_SIZE_X,RESCALE_SIZE_Y)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = UNet((RESCALE_SIZE_Y,RESCALE_SIZE_X))
checkpoint =  torch.load(state, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()
model.to(device)



if __name__ == '__main__':
    init()

    cut_clip(video_path, images_path, size)
    my_gen = Generator(model, device, size)
    path = Path(images_path)

    iter = 10000000
    dirlist = [x.name for x in path.iterdir() ]


    print(Fore.CYAN + '\n[+] Запуск генерации видео \n')
    shutil.copy("data/" + dirlist[0], "new_data/frame10000000.jpg")
    for i in range(len(dirlist) - 1):
        link_img1 = "data/" + dirlist[i]
        link_img2 = "new_data/frame" + str(iter+2*i+1) + ".jpg"
        link_img3 = "data/" + dirlist[i+1]
        destination_path1 = "new_data/frame" + str(iter+2*i+2) + ".jpg"
        shutil.copy(link_img1, destination_path1)
        my_gen.safe(link_img1, link_img3, link_img2)

        if i%100 ==0:
            print(100*i/(len(dirlist) - 1))
    print(Fore.YELLOW + f'[+] Генерация выполнена\n')

    name = "my_video.mp4"
    clip_from_image(prolog_img_path, video_path, name)



    #==========__Let's__Comprise!__================================================================================

    pictures = os.listdir(prolog_img_path)
    for i in range(len(pictures)):
        name = pictures[i]
        output_path = comprise_img_path + '/' + name
        print(i)
        if i%2 ==1:
            img1_path = 'new_data/' + name
            img2_path = 'new_data/' +pictures[i-1]
            image = merge_images_horizontally(img2_path,img1_path)
            im = Image.fromarray((image).astype(np.uint8))
            im.save(output_path)
        else:
            img1_path = 'new_data/' + name
            image = merge_images_horizontally(img1_path, img1_path)
            im = Image.fromarray((image).astype(np.uint8))
            im.save(output_path)

    name = 'comprision.mp4'
    clip_from_image(comprise_img_path, video_path, name)
