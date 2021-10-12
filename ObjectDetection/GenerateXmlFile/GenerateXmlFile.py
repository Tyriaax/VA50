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
        self.database = database["database"]

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
        self.difficult = str(object["difficult"])
        self.bndbox = self.Bndbox(object["bndbox"])

    class Bndbox:
      def __init__(self, bndbox):
        self.xmin = str(bndbox["xmin"])
        self.ymin = str(bndbox["ymin"])
        self.xmax = str(bndbox["xmax"])
        self.ymax = str(bndbox["ymax"])

  def publishXmlAnnotationFile(self, PATH_TO_SAVE, nameOfTheFile, content):
    # if nameOfTheFile.lower().endswith(".xml"):
    with open(PATH_TO_SAVE + '/' + nameOfTheFile + '.xml', "w") as XMLfile:
      XMLfile.write(content)
  
  def addText(self, root, upper, element, text):
    element.appendChild(root.createTextNode(text))
    upper.appendChild(element)

  def generateAnnotationFile(self):
    root = minidom.Document()
  
    #------------Annotation---------------
    xml = root.createElement("annotation") 
    root.appendChild(xml)
    
    #------------folder---------------
    folder = root.createElement("folder")
    self.addText(root,xml, folder, self.folder)

    #------------folder---------------
    filename = root.createElement("filename")
    self.addText(root, xml, filename, self.filename + '.jpg')

    #------------path---------------
    path = root.createElement("path")
    self.addText(root, xml, path, self.path)

    #------------source---------------
    source = root.createElement("source")
    database = root.createElement("database")
    self.addText(root, source, database, self.source.database)
    xml.appendChild(source)

    #------------size---------------
    size = root.createElement("size")

    width = root.createElement("width")
    self.addText(root, size, width, self.size.width)
    height = root.createElement("height")
    self.addText(root, size, height, self.size.height)
    depth = root.createElement("depth")
    self.addText(root, size, depth, self.size.depth)

    xml.appendChild(size)

    #------------path---------------
    segmented = root.createElement("segmented")
    self.addText(root, xml, segmented, self.segmented)

    #------------object---------------
    object = root.createElement("object")

    name = root.createElement("name")
    self.addText(root, object, name, self.object.name)
    pose = root.createElement("pose")
    self.addText(root, object, pose, self.object.name)
    truncated = root.createElement("truncated")
    self.addText(root, object, truncated, self.object.truncated)
    difficult = root.createElement("difficult")
    self.addText(root, object, difficult, self.object.difficult)

    bndbox = root.createElement("bndbox")
    xmin = root.createElement("xmin")
    self.addText(root, bndbox, xmin, self.object.bndbox.xmin)
    ymin = root.createElement("ymin")
    self.addText(root, bndbox, ymin, self.object.bndbox.ymin)
    xmax = root.createElement("xmax")
    self.addText(root, bndbox, xmax, self.object.bndbox.xmax)
    ymax = root.createElement("xmin")
    self.addText(root, bndbox, ymax, self.object.bndbox.ymax)
    object.appendChild(bndbox)

    xml.appendChild(object)

    #---------------presentation----------
    xml_str = root.toprettyxml(indent ="\t") 

    #---------------publish---------------
    # save_path_file = os.path.abspath(os.path.dirname( __file__ ))
    self.publishXmlAnnotationFile(self.path, self.filename, xml_str)


sourceDict = {
  "database" : "Unknown"
}

""""
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

"""
