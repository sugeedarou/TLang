from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from csv import DictReader

train_val_in = Path('../data/raw/train_val_eliminated.csv')
train_val_out = Path('../data/train_val')
test_in = Path('../data/raw/test_eliminated.csv')
test_out = Path('../data/test')
fontsize = 16
max_characters = 100
font = ImageFont.truetype(f'{Path.home()}/AppData/Local/Microsoft/Windows/Fonts/GoNotoCurrent.ttf', fontsize)

def save_text_to_image(text, save_path):
    W = int(font.getlength(text))
    # W = fontsize * max_characters
    H = int(fontsize*1.5)
    image = Image.new('1', (W, H), 255)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, fill='black', font=font)
    image.save(save_path)

def text_ds_to_images(in_path, out_dir):
    out_dir.mkdir(exist_ok=True)
    i = 0
    with open(in_path, 'r', encoding='utf-8', newline='') as f:
        reader = DictReader(f, delimiter=',')
        for row in reader:
            id = row['id']
            text = row['text']
            save_path = out_dir / f'{id}.png'
            save_text_to_image(text, save_path)
            
            # if i > 10:
            #     break
            # i += 1

text_ds_to_images(train_val_in, train_val_out)
text_ds_to_images(test_in, test_out)
print('done')

