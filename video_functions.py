import os.path
import time
from numpy import arange
from datetime import timedelta
import cv2
from tqdm import tqdm
import numpy as np
import os , shutil

from colorama import Fore
from colorama import init
from moviepy.editor import *
import subprocess

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def clip_from_image(path_dir, name_clip, name):
    try:
        #dur = cv2.VideoCapture(name_clip).get(cv2.CAP_PROP_FRAME_COUNT)
        fps = 2 * cv2.VideoCapture(name_clip).get(cv2.CAP_PROP_FPS)
    except ValueError:
        print(Fore.RED + '[-] Неверное значение длительности кадра')
        return

    if os.path.exists(path_dir):
        print(Fore.CYAN + '[+] Создание видео из картинок')
        os.chdir(path_dir)
        images = sorted(filter(lambda img: img.endswith(".jpg"), os.listdir(path_dir)))
        height, width = cv2.imread(images[0]).shape[:2]
        out_path = os.path.join('C:/Users/berku/Desktop/semestr_4/VNN/polongation_videos', name) # Создаем абсолютный путь к файлу видео
        video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for im in tqdm(images, desc="Обработка изображений", unit="файл"):
            frame = cv2.imread(im)
            video_writer.write(frame)
        video_writer.release()
        print(Fore.GREEN + f'[+] Видео создано и сохранено в папку: {out_path}')
    else:
        print(Fore.RED + '[-] Указанной директории не существует')

def extract(file):
    mp3 = None
    video_to_extract = None
    if file.endswith(".mp4") or file.endswith(".avi"):
        mp3_suff = f'.{file.replace(".", "_").split("_")[-1]}'
        mp3 = f'{file.removesuffix(mp3_suff)}.mp3'
        print(mp3)
        video_to_extract = VideoFileClip(os.path.join(os.getcwd(), file))

    print(Fore.CYAN + '\n[+] Запуск извлечения аудио\n')
    video_to_extract.audio.write_audiofile(os.path.join(os.getcwd(), mp3))
    print(Fore.YELLOW + f'[+] Аудио из файла: "{mp3}" извлечено\n')

# не используется пока что
def extract_mp3(path_file):
    file_in_dir = []
    if os.path.isdir(path_file):
        print(Fore.CYAN + '[+] Сканирование директории')
        file_in_dir = os.listdir(path_file)
    elif os.path.isfile(path_file):
        file_in_dir = os.listdir(os.getcwd())
        print(file_in_dir)

    video_to_extract = []
    mp3_list = []

    for file in file_in_dir:
        print(Fore.CYAN + f'\r[+] Добавляю файлы для извлечения: "{file}"', end='')
        print(file.endswith(".mp4") == True)
        if file.endswith(".mp4") or file.endswith(".avi"):
            mp3_suff = f'.{file.replace(".", "_").split("_")[-1]}'
            mp3_list.append(f'{file.removesuffix(mp3_suff)}.mp3')
            print(mp3_list)
            video_to_extract.append(VideoFileClip(os.path.join(os.getcwd(), file)))

    if len(video_to_extract) > 0:
        print(Fore.CYAN + '\n[+] Запуск извлечения аудио\n')
        for num, video in enumerate(video_to_extract):
            if os.path.exists(os.path.join(os.getcwd(), mp3_list[num])):
                mp3_name = f'{mp3_list[num].removesuffix(".mp3")}_{num + 1}.mp3'
                video.audio.write_audiofile(os.path.join(os.getcwd(), mp3_name))
                print(Fore.YELLOW + f'[+] Аудио из файла: "{mp3_name}" извлечено\n')
            else:
                video.audio.write_audiofile(os.path.join(os.getcwd(), mp3_list[num]))
                print(Fore.YELLOW + f'[+] Аудио из файла: "{mp3_list[num]}" извлечено\n')
        print(Fore.GREEN + '[+] Все видео файлы в директории обработаны. Аудио извлечено')
    else:
        print(Fore.RED + '\n[-] Файлов в директории не обнаружено')
        return   #

def merge_video_audio(path_file_v, path_file_a):
    if not os.path.exists(path_file_v):
        print(Fore.RED + '[-] С видеофайлом непорядок')
        return

    if not os.path.exists(path_file_a):
        print(Fore.RED + '[-] С аудиофайлом непорядок')

        return

    suff = f'.{os.path.split(path_file_v)[-1].split(".")[-1]}'
    vid_name = f'{os.path.split(path_file_v)[-1].removesuffix(suff)}_aud{suff}'
    if path_file_v.endswith(".mp4") or path_file_v.endswith(".avi") and path_file_a.endswith(".mp3"):
        print(Fore.CYAN + '[+] Добавляю аудио к видео')
        videoclip = VideoFileClip(path_file_v)
        audioclip = AudioFileClip(path_file_a)

        videoclip.audio = audioclip
        videoclip.write_videofile(os.path.join(os.path.split(path_file_v)[0], vid_name))
        print(Fore.GREEN + f'[+] Аудио добавлено. Файл сохранен: "{os.path.join(os.path.split(path_file_v)[0], vid_name)}"')
        return
    else:
        print(Fore.RED + '[-] Неверный формат файлов')
        return

def delete(image_path):
    for filename in os.listdir(image_path):
        file_path = os.path.join(image_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def cut_clip(video_path, data_path, size):
    cam = cv2.VideoCapture(video_path)
    try:

        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

        # frame
    currentframe = 10000000

    while (True):

        # reading from frame

        ret, frame = cam.read()

        if ret:
            frame = cv2.resize(frame, size)
            # if video is still left continue creating images
            name = './data/frame' + str(currentframe) + '.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 2
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

def merge_images_horizontally(image1_path, image2_path):
    image1 = cv2.imread(image1_path).astype(np.uint8)
    image2 = cv2.imread(image2_path).astype(np.uint8)

    if image1 is None or image2 is None:
        raise ValueError("Не удалось загрузить одно или оба изображения")

    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    new_height = max(height1, height2)
    new_width = width1 + width2

    merged_image = 255*np.ones((new_height, new_width, 3), dtype=np.uint8)
    merged_image[:height1, :width1, :] = image1
    merged_image[:height2, width1:, :] = image2

    return merged_image