from PIL import Image
import os

inputPath="lung/normal/test"
outputPath="lung/normal/test"

for filename in os.listdir(inputPath):
  imageFile=inputPath+"/"+filename
  image = Image.open(imageFile)
  sunset_resized = image.resize((224,224))
  outputFile=outputPath+"/"+filename
  sunset_resized.save(outputFile)