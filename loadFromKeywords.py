import sys
import os
import tinyimage

if len(sys.argv) < 4:
    print(
        "usage: python loadFromKeywords <keyword,keyword,keyword,...> <max> <output path>"
    )
    sys.exit(1)

keywords = sys.argv[1].split(",")
max_pics = int(sys.argv[2])
output_path = sys.argv[3]

tinyimage.openTinyImage()

for keyword in keywords:
    keyword_output_dir = output_path + "/" + keyword
    if not os.path.exists(keyword_output_dir):
        os.mkdir(keyword_output_dir)

    indexes = tinyimage.retrieveByTerm(keyword, max_pics)
    for i in indexes:
        tinyimage.sliceToImage(
            tinyimage.sliceToBin(i), keyword_output_dir + "/" + str(i) + ".png"
        )

tinyimage.closeTinyImage()
