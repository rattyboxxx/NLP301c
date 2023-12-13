import re

text = "Such as *+2+3*8 4+7*2, 12+2*6+"

res = re.findall(r"(?:[0-9+]+[0-9+*?]+)+[0-9]+", text)

print(res)
