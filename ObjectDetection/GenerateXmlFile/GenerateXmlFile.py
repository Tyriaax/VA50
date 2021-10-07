from xml.dom import minidom
import os 
  

class Annotation:
  def __init__(self, folder, filename, path, source, sizes, segmented, object):
      self.folder = folder
      self.filename = filename
      self.path = path
      self.source = self.Source(source)
      self.size = self.Size(sizes)
      self.segmented = str(segmented)
      self.object = self.Object(object)

  class Source:
    def __init__(self, database) -> None:
        self.database = database

  class Size: 
    def __init__(self, sizes):
      self.width = str(sizes["width"])
      self.height = str(sizes["height"])
      self.depth = str(sizes["depth"])

  class Object:
    def __init__(self, object) -> None:
        self.name = object["name"]
        self.pose = object["pose"]
        self.truncated = str(object["truncated"])
        self.difficult = object["difficult"]
        self.bndbox = self.Bndbox(object["bndbox"])

    class Bndbox:
      def __init__(self, bndbox):
        self.xmin = bndbox["xmin"]
        self.ymin = bndbox["ymin"]
        self.xmax = bndbox["xmax"]
        self.ymax = bndbox["ymax"]


  def publishXmlAnnotationFile(self, PATH_TO_SAVE, nameOfTheFile):
    if nameOfTheFile.lower().endswith(".xml"):
      with open(PATH_TO_SAVE + '/' + nameOfTheFile, "w") as XMLfile: 
        XMLfile.write(nameOfTheFile) 

  def generateAnnotationFile(self):

    root = minidom.Document()
  
    xml = root.createElement('root') 
    root.appendChild(xml)
      
    productChild = root.createElement('product')
    productChild.setAttribute('name', 'Geeks for Geeks')
      
    xml.appendChild(productChild)
      
    xml_str = root.toprettyxml(indent ="\t") 

    save_path_file = os.path.abspath(os.path.dirname( __file__ ))
    self.publishXmlAnnotationFile(save_path_file, "test.xml")


sourceDict = {
  "database" : "Unknown"
}

sizeDict = {
  "width" : 3456,
  "height" : 4608,
  "depth" : 3
}

bndBoxDict = {
  "xmin" : 741,
  "ymin" : 2717,
  "xmax" : 12,
  "ymax" : 21
}

objectDict = {
  "name" : "name",
  "pose" : "pose",
  "truncated" : 0,
  "difficult" : "difficult",
  "bndbox" : bndBoxDict
}

annotation = Annotation("nomask","filename","path",source=sourceDict,sizes= sizeDict, segmented = 0, object=objectDict)
annotation.generateAnnotationFile()

  
