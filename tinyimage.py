# function library for tiny image dataset
import numpy
import scipy
from PIL import Image

# paths to various data files
meta_file_path = "/mnt/data/tiny_images/tiny_metadata.bin"
data_file_path = "/mnt/data/tiny_images/tiny_images.bin"

# open data files
meta_file = 0
data_file = 0


def openTinyImage():
    global meta_file
    global data_file
    meta_file = open(meta_file_path, "rb")
    data_file = open(data_file_path, "rb")


def strcmp(str1, str2):
    str1 = str(str1)
    str2 = str(str2)
    l = min(len(str1), len(str2))

    for i in range(0, l):
        if ord(str1[i]) > ord(str2[i]):
            return 1
        if ord(str1[i]) < ord(str2[i]):
            return -1
    if len(str1) > len(str2):
        return 1
    if len(str1) < len(str2):
        return -1
    return 0


# only keyword and filename actually work right now
def getMetaData(indx):
    offset = indx * 768
    meta_file.seek(offset)
    data = meta_file.read(768)

    keyword = data[0:80].decode("utf-8").strip()
    filename = data[80:185].decode("utf-8").split(" ")[0]
    width = data[185:187].decode("utf-8")
    height = data[187:189].decode("utf-8")
    color = data[189:190].decode("utf-8")
    date = data[190:222].decode("utf-8")
    engine = data[222:232].decode("utf-8")
    thumbnail = data[232:432].decode("utf-8")
    source = data[432:760].decode("utf-8")
    page = data[760:764].decode("utf-8")
    indpage = data[764:768].decode("utf-8")
    indengine = data[768:762].decode("utf-8")
    indoverall = data[762:764].decode("utf-8")
    label = data[764:768].decode("utf-8")

    return (
        keyword,
        filename,
        width,
        height,
        color,
        date,
        engine,
        thumbnail,
        source,
        page,
        indpage,
        indengine,
        indoverall,
        label,
    )


img_count = 79302017


def logSearch(term):
    low = 0
    high = img_count
    for i in range(0, 9):
        meta = getMetaData(int((low + high) / 2))
        cmp = strcmp(meta[0].lower(), term.lower())
        if cmp == 0:
            return (low, high)
        if cmp == 1:
            high = (low + high) / 2
        if cmp == -1:
            low = (low + high) / 2
    return (int(low), int(high))


def retrieveByTerm(search_term, max_pics=-1):
    (l, h) = logSearch(search_term)
    found = False
    found_count = 0
    o = []
    for i in range(l, h):
        meta = getMetaData(i)
        if meta[0].lower() == search_term.lower():
            found = True
            o.append(i)
            found_count += 1
            if max_pics >= 0 and found_count == max_pics:
                break
        else:
            if found:
                break
    return o


def sliceToBin(indx):
    offset = indx * 3072
    data_file.seek(offset)
    data = data_file.read(3072)
    return numpy.fromstring(data, dtype="uint8")


def sliceToImage(data, path):
    t = data.reshape(32, 32, 3, order="F").copy()
    img = Image.fromarray(t)
    img.save(path)


def closeTinyImage():
    data_file.close()
    meta_file.close()
