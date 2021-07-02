import re

class Cleaner:
    decoration = re.compile(r"(［[^［］]*］)|(《[^《》]*》)|[｜\n]")
    def __init__(self, filename):
        self.filename = filename
        # 青空文庫はShift-JISなので
        with open(filename, "r", encoding="shift-jis") as afile:
            self.whole_str = afile.read()
        paragraphs = self.whole_str.splitlines()
        # 最後の3行の空白行以降のコメント行を除く
        c = 0
        position = 0
        for (i, u) in enumerate(reversed(paragraphs)):
            if len(u) != 0:
                c = 0
            else:
                c += 1
                if c >= 3:
                    position = i
                    break
        if position != 0:
            paragraphs = paragraphs[:-(position+1)]
 
        # 先頭の----行で囲まれたコメント領域の行を除く
        newparagraphs = []
        addswitch = True
        for u in paragraphs:
            if u[:2] != '--':
                if addswitch:
                    newparagraphs.append(u)
            else:
                addswitch = not addswitch
 
        self.cleanedparagraphs = []
        for u in newparagraphs:
            v = re.sub(self.decoration, '', u)
            self.cleanedparagraphs.append(v)
 
    def read(self):
        return self.cleanedparagraphs