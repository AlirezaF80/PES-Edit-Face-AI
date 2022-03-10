import glob
from typing import List

import pytesseract
import unidecode
from PIL import Image
from tqdm import tqdm


def get_player_name(full_img: Image):
    name_part = full_img.crop((790, 200, 1455, 250))
    name = pytesseract.image_to_string(name_part).strip()
    unaccented_string = unidecode.unidecode(name)
    return unaccented_string


def extract_csv(path) -> List[List[str]]:
    with open(path, 'r', encoding='UTF-8') as csvfile:
        lines = csvfile.readlines()
        lines = [l.split(',') for l in lines]
    return lines


if __name__ == '__main__':
    lines = extract_csv('data-cleaned.csv')
    names = [line[1].lower() for line in lines[1:]]
    ids = [line[0] for line in lines[1:]]
    print(len(names), len(ids))

    hit = 0
    for pic_path in tqdm(glob.glob(r'.\captures\*.jpg')):
        name = get_player_name(Image.open(pic_path)).lower()
        if name in names:
            occurences = [i for i, x in enumerate(names) if x == name]
            if len(occurences) == 1:
                img = Image.open(pic_path)
                face_crop = img.crop((1456, 120, 1920, 620))
                face_crop.save(fr'.\named\{ids[names.index(name)]}.jpg')
                # shutil.copy(pic_path, fr'.\named\{ids[names.index(name)]}.jpg')
                hit += 1
            else:
                print(name, len(occurences))
    # print hit percentage
    print(hit, len(glob.glob(r'.\captures\*.jpg')), round(hit / len(glob.glob(r'.\captures\*.jpg')) * 100, 2))
