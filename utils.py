
import os

# Required for making xml-tree
import xml.etree.cElementTree as ET
import xml.dom.minidom
from xml.dom import minidom
import xml.etree.ElementTree as xml
from xml.etree.ElementTree import tostring

VALID_FORMAT = ('.BMP', '.GIF', '.JPG', '.JPEG', '.PNG', '.PBM', '.PGM', '.PPM', '.TIFF', '.XBM')  # Image formats supported by Qt

def getImages(folder):
    ''' Get the names and paths of all the images in a directory. '''
    image_list = []
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            if file.upper().endswith(VALID_FORMAT):
                im_path = os.path.join(folder, file)
                image_obj = {'name': file, 'path': im_path }
                image_list.append(image_obj)
    return image_list


# Will be used to create a directory if it doesn't exist already
def is_dir_or_create(dir):
    path = os.path.join(os.getcwd(),dir)
    if not os.path.exists(path):
        os.mkdir(path)


def name_modify_img(fname,identifier, prefered_ext = None,directory=None):
    try:
        name, ext = fname.split(".")  
    except: 
        name = fname
        ext = prefered_ext or "png" 
        
    extension = prefered_ext or ext
    directory = directory or ""
    
    new_fname = directory+name+identifier+"."+extension # Original Name appending -pr1
    return new_fname

def save_image(data,fname,identifier):
    # Splitting the file name
    dir_name = "saved_img"
    is_dir_or_create(dir_name)
    fname = name_modify_img(fname,identifier)
    #new_image = Image.fromarray(data,'RGB')

    new_image = data
    if new_image != None:
        new_image.save(dir_name+'/'+fname)
    return True,fname


def create_save_xml_data(fname, identifier, s1,s2,s3,s4):
    dir_name = "saved_xmls"
    is_dir_or_create(dir_name)
    image_data = {}
    image_data['name'] = fname
    image_data["s1"] = str(s1)
    image_data["s2"] = str(s2)
    image_data["s3"] = str(s3)
    image_data["s4"] = str(s4)
    
    def create_xml():
        root = ET.Element("data")
        image = ET.SubElement(root, "image")
        ET.SubElement(image,"name").text = image_data["name"]
        coordinates = ET.SubElement(image,"coordinates")

        ET.SubElement(coordinates,"S1").text = image_data["s1"]
        ET.SubElement(coordinates,"S2").text = image_data["s2"]
        ET.SubElement(coordinates,"S3").text = image_data["s3"]
        ET.SubElement(coordinates,"S4").text = image_data["s4"]
        
        return root
    
    def write_xml_file(fname,root):
        is_dir_or_create(dir_name)
        fname = name_modify_img(fname, identifier=identifier,prefered_ext= "xml")
      
        tree = xml.ElementTree(root)
     
        #xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
        
        with open("./"+ dir_name +"/"+fname,"wb") as files:
            tree.write(files)

        return True
    
    
    root = create_xml()
    write_xml_file(fname,root)
    
    return True