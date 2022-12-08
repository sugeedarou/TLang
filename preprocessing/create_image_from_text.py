import os
from PIL import Image, ImageDraw, ImageFont

# txt = '香港では旧正月がトラフィックの谷になるのか。地域ならではのトラフィック傾向で面白い。'
txt = '香港では旧正月がトラフィックの谷になるのか。地域ならではのトラフィック傾向で面。香港では旧正月がトラフィックの谷になるのか。地域ならではのトラフィック傾向で面。ラフィック傾向で面。ラフィック傾向で面。'
txt.encode("utf-8")
fontsize = 16
font = ImageFont.truetype('%WINDIR%/FontsGoNotoCurrent.ttf', fontsize)

h = fontsize
w = font.getlength(txt)
W = int(w)
H = int(h * 1.5)

image = Image.new('RGBA', (W, H), (255,255,255))
draw = ImageDraw.Draw(image)

draw.text((fontsize // 4,0), txt, fill='black', font=font)

save_location = os.getcwd()
image.save(save_location + '/sample.png')
